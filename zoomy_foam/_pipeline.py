"""In-process run entry for the OpenFOAM (zoomyFoam) backend — REQ-93.

Foam is not drivable in-process: it is codegen -> apptainer ``wmake`` ->
OpenFOAM ``polyMesh`` + ``0/`` fields -> ``zoomyFoam`` -> VTK, with the DOF count
baked ``constexpr`` per model.  This module wraps that whole pipeline behind one
call so the server's ``FoamAdapter`` (and anyone) can drive the *shared*
folder-case format exactly like the numpy/jax/dmplex/amrex backends::

    from zoomy_foam import run_case
    h5 = run_case(model, settings, output_dir, on_progress=None)   # -> HDF5 path

It mirrors, generically, the by-hand flow in
``tools/macdonald_friction_verification.py``:

  (a) code-gen ``Model.H`` / ``NumericsKernels.H`` from the resolved model
      (``FoamSystemModelPrinter`` / ``FoamNumericsPrinter``);
  (b) ``wmake`` ``zoomyFoam`` in the OpenFOAM-13 apptainer — cached by a hash of
      the emitted headers (the DOF + physics are baked in, so identical headers
      reuse the binary);
  (c) build the OpenFOAM case: structured ``blockMeshDict`` from the mesh, ``0/Qi``
      from ``model.initial_conditions`` at the cell centres, ``controlDict`` from
      ``settings`` (endTime / maxCo / reconstructionOrder / timeScheme / params);
  (d) run ``zoomyFoam``; parse ``Time = `` lines for ``on_progress``;
  (e) ``foamToVTK`` -> ``zoomy_prepost.vtk_to_hdf5`` so the server serves the
      shared HDF5 format.

Env: ``ZOOMY_OF_SIF`` overrides the apptainer image (default
``~/of_build/zoomy_openfoam.sif``).
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# The wmake tree (Model.H, zoomyFoam.C, Make/) is the package's PARENT dir.
FOAM_ROOT = Path(__file__).resolve().parent.parent
SIF = Path(os.environ.get("ZOOMY_OF_SIF", str(Path.home() / "of_build" / "zoomy_openfoam.sif")))
_BINCACHE = FOAM_ROOT / ".bincache"
# where `wmake` installs the freshly built binary inside the apptainer
_OF_BIN = "$HOME/OpenFOAM/$(whoami)-13/platforms/linux64GccDPInt32Opt/bin/zoomyFoam"


# ── apptainer helper ────────────────────────────────────────────────────────
def _bind_args(binds):
    """--bind each real path so it is visible inside the container.

    ``Path.resolve()`` canonicalises through the host's symlinks (e.g.
    ``/mnt/userdrive/...``); the container only auto-mounts ``$HOME``/CWD, so any
    other real path (the wmake tree, a scratch output dir) must be bound."""
    out = []
    for b in sorted({str(FOAM_ROOT), *(str(x) for x in binds)}):
        out += ["--bind", f"{b}:{b}"]
    return out


def _apptainer_cmd(script, binds=()):
    return ["apptainer", "exec", *_bind_args(binds), str(SIF), "bash", "-lc",
            "source /opt/openfoam13/etc/bashrc 2>/dev/null; " + script]


def _apptainer(script, binds=(), **kw):
    if not SIF.exists():
        raise RuntimeError(
            f"OpenFOAM apptainer image not found at {SIF} (set ZOOMY_OF_SIF). "
            "zoomy_foam.run_case needs the OF-13 container to wmake + run zoomyFoam.")
    return subprocess.run(_apptainer_cmd(script, binds), **kw)


# ── (a) codegen ─────────────────────────────────────────────────────────────
def _codegen(model):
    """Emit Model.H / NumericsKernels.H into the wmake tree; return the SystemModel.

    Uses ``SystemModel.from_model`` (NOT ``model.system_model``) so a resolved
    case model carries its model-level baked BCs into the kernel — the same
    coercion the numpy adapter path uses."""
    from zoomy_core.systemmodel import SystemModel
    from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
    from zoomy_core.transformation.to_openfoam import (
        FoamSystemModelPrinter, FoamNumericsPrinter)
    sm = SystemModel.from_model(model)
    FoamSystemModelPrinter.write_code(sm, FOAM_ROOT / "Model.H")
    FoamNumericsPrinter.write_code(
        PositiveNonconservativeRusanov(model=sm), FOAM_ROOT / "NumericsKernels.H")
    return sm


# ── (b) build (hash-cached) ─────────────────────────────────────────────────
def _headers_hash():
    h = hashlib.sha256()
    for name in ("Model.H", "NumericsKernels.H", "zoomyFoam.C"):
        h.update((FOAM_ROOT / name).read_bytes())
    return h.hexdigest()[:16]


def _wmake_cached():
    """wmake zoomyFoam for the currently-emitted headers; cache by header hash.

    The physics + DOF are baked into Model.H/NumericsKernels.H, so identical
    headers reuse the cached binary.  Returns the absolute path to the binary."""
    _BINCACHE.mkdir(exist_ok=True)
    cached = _BINCACHE / f"zoomyFoam_{_headers_hash()}"
    if cached.exists():
        return cached
    r = _apptainer(f"cd {FOAM_ROOT}; wclean >/dev/null 2>&1; wmake 2>&1 | tail -4 && "
                   f"cp {_OF_BIN} '{cached}'",
                   capture_output=True, text=True)
    if r.returncode != 0 or not cached.exists():
        raise RuntimeError(f"zoomyFoam wmake failed:\n{r.stdout}\n{r.stderr}")
    return cached


# ── (c) case build ──────────────────────────────────────────────────────────
def _mesh_geometry(mesh):
    """(x0, x1, n) for a uniform 1-D interval mesh from an LSQMesh.

    Only structured 1-D is handled here (the shared SWE/SME channel cases); a
    genuine 2-D/unstructured mesh should come in as a gmsh ``.msh`` and go
    through ``gmshToFoam`` — raised as NotImplementedError until wired."""
    nc = int(mesh.n_inner_cells)
    cc = np.asarray(mesh.cell_centers)[:, :nc]
    xc = cc[0]
    y_flat = np.allclose(cc[1], cc[1][0]) if cc.shape[0] > 1 else True
    z_flat = np.allclose(cc[2], cc[2][0]) if cc.shape[0] > 2 else True
    if not (y_flat and z_flat):
        raise NotImplementedError(
            "zoomy_foam.run_case currently builds structured 1-D blockMesh only; "
            "a 2-D/unstructured case must be supplied as a gmsh .msh for gmshToFoam.")
    order = np.argsort(xc)
    xc = xc[order]
    dx = float(np.mean(np.diff(xc))) if nc > 1 else 1.0
    return float(xc.min() - dx / 2), float(xc.max() + dx / 2), nc, order


def _write_grid_system(case, x0, x1, n):
    """Structured 1-D blockMeshDict + minimal fvSchemes/fvSolution (shared by the
    explicit and Chorin case builders)."""
    (case / "system" / "blockMeshDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\n"
        "convertToMeters 1;\n"
        f"vertices ( ({x0} 0 0)({x1} 0 0)({x1} 1 0)({x0} 1 0)"
        f"({x0} 0 1)({x1} 0 1)({x1} 1 1)({x0} 1 1) );\n"
        f"blocks ( hex (0 1 2 3 4 5 6 7) ({n} 1 1) simpleGrading (1 1 1) );\n"
        "edges (); boundary (\n"
        "  left  { type patch; faces ( (0 4 7 3) ); }\n"
        "  right { type patch; faces ( (1 2 6 5) ); }\n"
        "  frontAndBack { type empty; faces ( (0 1 5 4) (3 7 6 2) ); }\n"
        "  topAndBottom { type empty; faces ( (0 3 2 1) (4 5 6 7) ); }\n"
        "); mergePatchPairs ();\n")
    (case / "system" / "fvSchemes").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n"
        "ddtSchemes { default none; } gradSchemes { default Gauss linear; }\n"
        "divSchemes { default none; } laplacianSchemes { default none; }\n"
        "interpolationSchemes { default linear; } snGradSchemes { default corrected; }\n")
    (case / "system" / "fvSolution").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers {}\n")


def _field_file(case, name, vals, patches):
    body = "\n".join(f"{v:.10g}" for v in np.atleast_1d(vals))
    bf = "\n".join(f"  {p} {{ type {t}; }}" for p, t in patches.items())
    (case / "0" / name).write_text(
        f"FoamFile {{ version 2.0; format ascii; class volScalarField; object {name}; }}\n"
        f"dimensions [0 0 0 0 0 0 0];\ninternalField nonuniform List<scalar>\n"
        f"{len(np.atleast_1d(vals))}\n(\n{body}\n);\nboundaryField {{\n{bf}\n}}\n")


def _model_parameters(model, sm):
    """{name: value} for the controlDict modelParameters, from the model."""
    out = {}
    vals = getattr(model, "parameter_values", None)
    if vals is not None:
        for k in getattr(vals, "keys", lambda: [])():
            try:
                out[str(k)] = float(getattr(vals, k))
            except Exception:
                pass
    # fall back / fill from the system model's declared parameter symbols
    for s in getattr(sm, "parameters", []) or []:
        nm = str(s)
        if nm not in out:
            dv = getattr(s, "default", None)
            out[nm] = float(dv) if dv is not None else 0.0
    return out


def _build_case(case, mesh, model, sm, settings, binary):
    if case.exists():
        shutil.rmtree(case)
    (case / "0").mkdir(parents=True); (case / "system").mkdir(); (case / "constant").mkdir()
    x0, x1, n, order = _mesh_geometry(mesh)
    _write_grid_system(case, x0, x1, n)

    t_end = float(settings.get("time_end", 1.0))
    n_snap = int(settings.get("output_snapshots", 20)) or 1
    order_recon = int(settings.get("reconstruction_order", 1))
    scheme = settings.get("time_scheme", "explicit")
    maxco = float(settings.get("cfl", 0.4))
    dt0 = float(settings.get("min_dt", 1e-3)) or 1e-3
    params = _model_parameters(model, sm)
    param_str = " ".join(f"{k} {v:g};" for k, v in params.items())
    imex = ("imexTableau ars232; imexMaxIter 20; imexTol 1e-12;"
            if scheme.lower().startswith("imex") else "")
    (case / "system" / "controlDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
        "application zoomyFoam;\n"
        f"startFrom startTime; startTime 0; stopAt endTime; endTime {t_end}; deltaT {dt0};\n"
        f"writeControl adjustableRunTime; writeInterval {t_end / n_snap:g}; purgeWrite 0;\n"
        f"maxCo {maxco}; reconstructionOrder {order_recon}; timeScheme {scheme}; {imex}\n"
        f"modelParameters {{ {param_str} }}\n")

    # 0/Qi from the model's initial conditions at the (ordered) inner cell centres
    nc = int(mesh.n_inner_cells)
    cc = np.asarray(mesh.cell_centers)[:, :nc]
    Q = np.zeros((len(sm.state), nc))
    Q = model.initial_conditions.apply(cc, Q)
    Q = np.asarray(Q)[:, order]
    patches = {"left": "zeroGradient", "right": "zeroGradient",
               "frontAndBack": "empty", "topAndBottom": "empty"}
    for i in range(len(sm.state)):
        _field_file(case, f"Q{i}", Q[i], patches)
    return binary


# ── (d) run ─────────────────────────────────────────────────────────────────
def _run_stream(case, binary, on_progress):
    _apptainer(f"cd {case}; blockMesh > log.blockMesh 2>&1", binds=[case], check=True)
    p = subprocess.Popen(
        _apptainer_cmd(f"cd {case}; '{binary}' -case {case}", binds=[case]),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    log = (case / "run.log").open("w")
    it, prev_t = 0, 0.0
    # zoomyFoam prints `Time = <t>s` per step (adaptive dt, no separate deltaT
    # line) — derive dt from consecutive report times.
    time_re = re.compile(r"^Time = ([-\d.eE+]+)")
    for line in p.stdout:
        log.write(line)
        m = time_re.match(line.strip())
        if m:
            t = float(m.group(1)); it += 1
            dt = t - prev_t; prev_t = t
            if on_progress:
                try:
                    on_progress(it, t, dt)
                except Exception:
                    pass
    p.wait(); log.close()
    if p.returncode != 0:
        raise RuntimeError(f"zoomyFoam failed (rc={p.returncode}); see {case/'run.log'}")


# ── (e) VTK -> HDF5 ─────────────────────────────────────────────────────────
def _time_dirs(case):
    return sorted((d for d in case.iterdir()
                   if d.is_dir() and re.fullmatch(r"[0-9]+(\.[0-9]+)?", d.name)),
                  key=lambda d: float(d.name))


def _strip_nonstate(case, n_state):
    """Keep only the state fields ``Q0..Q{n-1}`` in each written time dir, so the
    exported HDF5 ``Q`` is exactly the model state — not zoomyFoam's internal
    reconstruction diagnostics (``Dm*``/``Dp*``) or aux (``Qaux*``)."""
    keep = {f"Q{i}" for i in range(n_state)} | {"uniform"}
    for d in _time_dirs(case):
        for fp in d.iterdir():
            if fp.name not in keep:
                (fp.unlink() if fp.is_file() else shutil.rmtree(fp, ignore_errors=True))


def _write_pvd(path, time_files):
    ds = "\n".join(f'    <DataSet timestep="{t:g}" file="{f.name}"/>' for t, f in time_files)
    path.write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
                    f'  <Collection>\n{ds}\n  </Collection>\n</VTKFile>\n')


def _vtk_field_names(vtk_path):
    """Field names of a VTK frame in the SAME order ``zoomy_prepost`` packs Q."""
    import meshio
    m = meshio.read(vtk_path)
    names = []
    for name, blocks in (m.cell_data or {}).items():
        arr = np.asarray(blocks[0])
        names += [name] if arr.ndim == 1 else [f"{name}_{c}" for c in range(arr.shape[1])]
    if not names:
        for name, arr in (m.point_data or {}).items():
            arr = np.asarray(arr)
            names += [name] if arr.ndim == 1 else [f"{name}_{c}" for c in range(arr.shape[1])]
    return names


def _keep_state_rows(h5_path, n_state, sample_vtk):
    """Drop foamToVTK's synthetic ``cellID`` (and any non-state field): keep the
    ``Q0..Q{n-1}`` rows, in state order, in every frame's ``Q``."""
    import h5py
    names = _vtk_field_names(sample_vtk)
    try:
        idx = [names.index(f"Q{i}") for i in range(n_state)]
    except ValueError:
        return  # unexpected field naming — leave Q untouched rather than corrupt it
    if idx == list(range(n_state)):
        return
    with h5py.File(h5_path, "a") as f:
        for k in f["fields"]:
            g = f["fields"][k]
            Q = g["Q"][:]
            del g["Q"]
            g.create_dataset("Q", data=Q[idx])


def _to_vtk(case, n_state):
    """Strip to the state fields, `foamToVTK`, and write a `.pvd` collection with
    physical OF times. Returns ``(source, vtks)`` — ``source`` is the ``.pvd``
    path (or an ordered frame list on a count mismatch), ``vtks`` the sorted
    frame files."""
    _strip_nonstate(case, n_state)
    _apptainer(f"cd {case}; foamToVTK > log.foamToVTK 2>&1", binds=[case], check=True)
    vtkdir = case / "VTK"
    vtks = sorted(vtkdir.glob("*.vtk"),
                  key=lambda p: int(re.search(r"_(\d+)\.vtk$", p.name).group(1)))
    times = [float(d.name) for d in _time_dirs(case)]
    if not vtks:
        raise RuntimeError(f"foamToVTK produced no VTK frames in {vtkdir}")
    if len(vtks) == len(times):
        pvd = vtkdir / "series.pvd"
        _write_pvd(pvd, list(zip(times, vtks)))
        return str(pvd), vtks
    return [str(p) for p in vtks], vtks   # count mismatch → numeric-index order


def _to_hdf5(case, output_dir, n_state):
    source, vtks = _to_vtk(case, n_state)
    from zoomy_prepost import vtk_to_hdf5
    out = vtk_to_hdf5(source, str(Path(output_dir) / "simulation.h5"))
    _keep_state_rows(out, n_state, str(vtks[0]))
    return out


# ── public entry ────────────────────────────────────────────────────────────
def run_case(model, settings, output_dir, on_progress=None):
    """Run a shared folder-case on the zoomyFoam backend; return the HDF5 path.

    Parameters
    ----------
    model : zoomy_core Model
        Resolved case model (IC/BC baked); coerced with ``SystemModel.from_model``.
    settings : dict
        The case ``settings.json`` (``mesh``, ``time_end``, ``cfl``,
        ``output_snapshots``, ``reconstruction_order``, optional ``time_scheme``,
        ``min_dt``).  ``mesh`` is resolved relative to ``settings["_case_dir"]`` /
        the current directory when not absolute.
    output_dir : str | Path
        Where the OpenFOAM case + ``simulation.h5`` are written.
    on_progress : callable(iteration, time, dt) | None
    """
    case, sm = _run_pipeline(model, settings, output_dir, on_progress)
    return _to_hdf5(case, Path(output_dir), len(sm.state))


def _run_pipeline(model, settings, output_dir, on_progress):
    """Shared codegen → wmake → case → run prefix. Returns ``(case_dir, sm)``."""
    from zoomy_core.mesh.lsq_mesh import LSQMesh
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    mp = settings.get("mesh", "mesh.h5")
    if not os.path.isabs(mp):
        mp = os.path.join(settings.get("_case_dir", os.getcwd()), mp)
    mesh = LSQMesh.from_hdf5(mp)

    sm = _codegen(model)
    binary = _wmake_cached()
    case = output_dir / "foam_case"
    _build_case(case, mesh, model, sm, settings, binary)
    _run_stream(case, binary, on_progress)
    return case, sm


def run_to_vtk(model, settings, output_dir, on_progress=None):
    """Same pipeline as :func:`run_case` but stops at the VTK series (no HDF5) —
    the shape the gui solver wrappers use (gui converts VTK→h5 with
    ``zoomy_prepost``). Returns the ``.pvd`` collection path."""
    case, sm = _run_pipeline(model, settings, output_dir, on_progress)
    return _to_vtk(case, len(sm.state))[0]


# ── Chorin split pipeline (non-hydrostatic VAM — the chorinFoam app) ─────────
_OF_CHORIN_BIN = "$HOME/OpenFOAM/$(whoami)-13/platforms/linux64GccDPInt32Opt/bin/chorinFoam"


def _codegen_chorin(model):
    """Emit the Chorin split headers from a model with ``chorin_split``:
    predictor ``Model.H`` + ``NumericsKernels.H`` + ``Pressure.H`` (ChorinPressure)
    + ``Corrector.H`` (ChorinCorrector) + ``ChorinState.H`` (full n_state).
    Mirrors ``create_model.py`` but from the model OBJECT (baked BCs).  Returns
    ``(full_sm, n_state)``."""
    import sympy as sp
    from zoomy_core.systemmodel import SystemModel
    from zoomy_core.transformation.to_openfoam import (
        FoamSystemModelPrinter, FoamNumericsPrinter)
    from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
    # SystemModel.from_model (NOT model.system_model): the latter currently trips
    # over VAM's dict-shaped interpolate_to_3d (core regression); from_model is the
    # same coercion the explicit path uses and carries the baked BCs.
    full = SystemModel.from_model(model)
    n_state = len(full.state)
    dt = sp.Symbol("dt", positive=True)
    split = model.chorin_split(dt, system_model=full)
    FoamSystemModelPrinter.write_code(split.SM_pred, FOAM_ROOT / "Model.H",
                                      namespace_name="Model")
    FoamNumericsPrinter.write_code(
        PositiveNonconservativeRusanov(model=split.SM_pred),
        FOAM_ROOT / "NumericsKernels.H")
    FoamSystemModelPrinter.write_code(split.SM_press, FOAM_ROOT / "Pressure.H",
                                      namespace_name="ChorinPressure", dt_symbol=dt)
    FoamSystemModelPrinter.write_code(split.SM_corr, FOAM_ROOT / "Corrector.H",
                                      namespace_name="ChorinCorrector")
    (FOAM_ROOT / "ChorinState.H").write_text(
        "#pragma once\n"
        f"namespace Model {{ constexpr int n_state = {n_state}; }}\n")
    return full, n_state


def _wmake_chorin_cached():
    """wmake the ``chorin_app`` (chorinFoam) for the current headers; cache by a
    hash of the split headers + driver.  Returns the cached binary path."""
    _BINCACHE.mkdir(exist_ok=True)
    h = hashlib.sha256()
    for name in ("Model.H", "Pressure.H", "Corrector.H", "ChorinState.H",
                 "NumericsKernels.H", "chorin_app/chorinFoam.C"):
        h.update((FOAM_ROOT / name).read_bytes())
    cached = _BINCACHE / f"chorinFoam_{h.hexdigest()[:16]}"
    if cached.exists():
        return cached
    r = _apptainer(
        f"cd {FOAM_ROOT}/chorin_app; wclean >/dev/null 2>&1; wmake 2>&1 | tail -4 && "
        f"cp {_OF_CHORIN_BIN} '{cached}'", capture_output=True, text=True)
    if r.returncode != 0 or not cached.exists():
        raise RuntimeError(f"chorinFoam wmake failed:\n{r.stdout}\n{r.stderr}")
    return cached


def _build_chorin_case(case, mesh, model, sm, settings):
    if case.exists():
        shutil.rmtree(case)
    (case / "0").mkdir(parents=True); (case / "system").mkdir(); (case / "constant").mkdir()
    x0, x1, n, order = _mesh_geometry(mesh)
    _write_grid_system(case, x0, x1, n)

    t_end = float(settings.get("time_end", 1.0))
    n_snap = int(settings.get("output_snapshots", 20)) or 1
    maxco = float(settings.get("cfl", 0.3))
    dt0 = float(settings.get("min_dt", 1e-3)) or 1e-3
    ptol = float(settings.get("pressure_tol", 1e-8))
    pmaxit = int(settings.get("pressure_maxit", 2000))
    params = _model_parameters(model, sm)
    param_str = " ".join(f"{k} {v:g};" for k, v in params.items())
    (case / "system" / "controlDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
        "application chorinFoam;\n"
        f"startFrom startTime; startTime 0; stopAt endTime; endTime {t_end}; deltaT {dt0};\n"
        f"writeControl adjustableRunTime; writeInterval {t_end / n_snap:g}; purgeWrite 0;\n"
        f"maxCo {maxco};\n"
        f"pressureSolver bicgstab; pressureTol {ptol:g}; pressureMaxIter {pmaxit};\n"
        f"modelParameters {{ {param_str} }}\n")

    nc = int(mesh.n_inner_cells)
    cc = np.asarray(mesh.cell_centers)[:, :nc]
    Q = np.zeros((len(sm.state), nc))
    Q = np.asarray(model.initial_conditions.apply(cc, Q))[:, order]
    patches = {"left": "zeroGradient", "right": "zeroGradient",
               "frontAndBack": "empty", "topAndBottom": "empty"}
    for i in range(len(sm.state)):
        _field_file(case, f"Q{i}", Q[i], patches)


def run_chorin_to_vtk(model, settings, output_dir, on_progress=None):
    """Chorin (pressure-projection) analog of :func:`run_to_vtk` — split codegen →
    wmake chorin_app → case (8-state VAM) → chorinFoam → VTK series.  Returns the
    ``.pvd`` path.  ``model`` must expose ``chorin_split`` (e.g. VAM)."""
    from zoomy_core.mesh.lsq_mesh import LSQMesh
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    mp = settings.get("mesh", "mesh.h5")
    if not os.path.isabs(mp):
        mp = os.path.join(settings.get("_case_dir", os.getcwd()), mp)
    mesh = LSQMesh.from_hdf5(mp)

    full, n_state = _codegen_chorin(model)
    binary = _wmake_chorin_cached()
    case = output_dir / "foam_case"
    _build_chorin_case(case, mesh, model, full, settings)
    _run_stream(case, binary, on_progress)
    return _to_vtk(case, n_state)[0]
