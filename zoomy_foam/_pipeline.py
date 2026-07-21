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
import time
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
    from zoomy_foam._constants import write_march_constants
    sm = SystemModel.from_model(model)
    # Emission order kept as it was (Model.H before the Riemann solver is
    # constructed).  Checked, not assumed: constructing
    # ``PositiveNonconservativeRusanov(model=sm)`` first and printing after
    # yields a BYTE-IDENTICAL Model.H, so the order is not load-bearing — it is
    # preserved only to keep this diff to the constants emission.
    FoamSystemModelPrinter.write_code(sm, FOAM_ROOT / "Model.H")
    nsm = PositiveNonconservativeRusanov(model=sm)
    FoamNumericsPrinter.write_code(nsm, FOAM_ROOT / "NumericsKernels.H")
    # Mandate 6a: the MOOD detector bound is RESOLVED BY CORE, not restated as a
    # literal in zoomyFoam.C.  Emitted on EVERY codegen so the header can never
    # lag the model the binary is compiled against.
    write_march_constants(nsm, FOAM_ROOT / "MarchConstants.H")
    return sm


# ── (b) build (hash-cached) ─────────────────────────────────────────────────
def _headers_hash():
    # Hash EVERY source that goes into the zoomyFoam binary, not just the
    # generated pair — the hand-written headers (numerics*.H, init.H,
    # UserFunctions.H, Numerics.H) change the binary too, and omitting them meant
    # an edit there silently reused a stale cached binary.
    h = hashlib.sha256()
    # MarchConstants.H is generated and carries the MOOD detector bound: leaving
    # it out would let a changed bound reuse a stale binary, which is the exact
    # failure mode this hash exists to prevent.
    for name in ("Model.H", "NumericsKernels.H", "MarchConstants.H",
                 "Numerics.H", "numerics.H",
                 "numerics_o2.H", "init.H", "UserFunctions.H", "zoomyFoam.C"):
        f = FOAM_ROOT / name
        if f.exists():
            h.update(name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _wmake_script(build_dir, of_bin, cached):
    """Shell for one cached wmake: build in ``build_dir``, and copy ``of_bin``
    to ``cached`` ONLY if the compile actually succeeded.

    Two guards, both load-bearing — this used to read

        wmake 2>&1 | tail -4 && cp <of_bin> <cached>

    where ``&&`` binds to the PIPELINE's status, i.e. ``tail``'s, which is
    always 0.  A FAILED wmake therefore still copied whatever stale binary
    happened to sit in the install dir into .bincache under the NEW header
    hash; ``cached.exists()`` then went true and the RuntimeError never fired.
    Observed consequence: a "clean 16.84 s build" that ran a stale 4-state
    binary against a freshly generated 3-state Model.H — silent, and it would
    corrupt every reference produced from that cache entry.

      1. ``rm -f`` the install target first, so a stale binary cannot be
         mistaken for a fresh one even if the status check is subverted.
      2. capture wmake's OWN status via a log file rather than a pipe.  Done
         with a plain redirect + ``$?`` instead of ``set -o pipefail`` /
         ``$PIPESTATUS`` because the container shell is not guaranteed to be
         bash; this form is POSIX and works in dash too.

    The trailing ``[ $rc -eq 0 ]`` makes a failed compile exit non-zero, so
    the caller's ``returncode`` check raises instead of caching junk."""
    log = "wmake.$$.log"
    return (
        f"cd {build_dir}; wclean >/dev/null 2>&1; rm -f {of_bin}; "
        # tail -60, not -4: a template-heavy OpenFOAM compile error is many lines
        # of instantiation context, and the LAST 4 are the `make: *** Error 1`
        # trailer -- i.e. the old width reliably hid the actual diagnostic and
        # reported only that something failed.
        f"wmake > {log} 2>&1; rc=$?; tail -60 {log}; rm -f {log}; "
        f"[ $rc -eq 0 ] && cp {of_bin} '{cached}'"
    )


def _wmake_cached():
    """wmake zoomyFoam for the currently-emitted headers; cache by header hash.

    The physics + DOF are baked into Model.H/NumericsKernels.H, so identical
    headers reuse the cached binary.  Returns the absolute path to the binary."""
    _BINCACHE.mkdir(exist_ok=True)
    cached = _BINCACHE / f"zoomyFoam_{_headers_hash()}"
    if cached.exists():
        return cached
    r = _apptainer(_wmake_script(FOAM_ROOT, _OF_BIN, cached),
                   capture_output=True, text=True)
    if r.returncode != 0 or not cached.exists():
        raise RuntimeError(f"zoomyFoam wmake failed:\n{r.stdout}\n{r.stderr}")
    return cached


# ── (c) case build ──────────────────────────────────────────────────────────
def _default_face_names(dim):
    """Side-patch names in axis order (x-lo, x-hi[, y-lo, y-hi]).  1-D keeps the
    historical left/right; 2-D uses the compass tags the shared cases already
    tag their BCs with (mesh.FACE_NAMES = West/East/South/North)."""
    return ("left", "right") if dim == 1 else ("West", "East", "South", "North")


def _mesh_geometry(mesh):
    """Structured-grid geometry for a uniform 1-D or 2-D LSQMesh.

    Returns ``(lo, hi, n, order, dim)``: ``lo/hi/n`` are per-active-axis lists
    (x, then y), ``order`` maps the LSQMesh cell order to OpenFOAM's blockMesh
    order (x fastest, y slowest — ``hex (nx ny 1)``), ``dim`` in {1, 2}.  The
    foam SOLVER is dimension-agnostic (it reads the full face normal); only this
    structured-grid builder is bounded — a 3-D or unstructured case must arrive
    as a gmsh ``.msh`` and go through ``gmshToFoam``."""
    nc = int(mesh.n_inner_cells)
    cc = np.asarray(mesh.cell_centers)[:, :nc]
    active = [ax for ax in range(min(cc.shape[0], 3))
              if not np.allclose(cc[ax], cc[ax][0])] or [0]
    if active != list(range(len(active))):
        raise NotImplementedError(
            "zoomy_foam builds structured grids with contiguous active axes "
            "(1-D in x, 2-D in x–y); supply a gmsh .msh for anything else.")
    dim = len(active)
    if dim > 2:
        raise NotImplementedError(
            "zoomy_foam structured build handles 1-D and 2-D; a 3-D or "
            "unstructured case must come as a gmsh .msh for gmshToFoam.")
    lo, hi, n = [], [], []
    for ax in active:
        u = np.unique(np.round(cc[ax], 9))
        d = float(np.mean(np.diff(u))) if len(u) > 1 else 1.0
        lo.append(float(u.min() - d / 2))
        hi.append(float(u.max() + d / 2))
        n.append(int(len(u)))
    # OF cell order for hex (nx ny 1) is x-fastest, y-slowest.  lexsort's LAST
    # key is primary, so keys (x, y) sort by y then x — exactly that order.
    order = np.lexsort(tuple(cc[ax] for ax in active))
    return lo, hi, n, order, dim


def _write_grid_system(case, lo, hi, n, dim=1, face_names=None):
    """Structured blockMeshDict (1-D interval or 2-D quad grid) + minimal
    fvSchemes/fvSolution, shared by the explicit and Chorin builders.

    The mesh box is ``[lo, hi]`` on each active axis; the y (1-D only) and z (all)
    directions are a single ``empty`` cell so OpenFOAM's operators run in the
    intended dimension.  ``face_names`` names the ``2*dim`` side patches in axis
    order — they MUST match the model's BC tags, else the solver's name-based BC
    dispatch (init.H) leaves them at the field default (zeroGradient)."""
    fn = tuple(face_names) if face_names else _default_face_names(dim)
    x0, x1 = lo[0], hi[0]
    if dim >= 2:
        y0, y1, nx, ny = lo[1], hi[1], n[0], n[1]
    else:
        y0, y1, nx, ny = 0.0, 1.0, n[0], 1
    # 8 hex vertices of [x0,x1]×[y0,y1]×[0,1].
    verts = [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0),
             (x0, y0, 1), (x1, y0, 1), (x1, y1, 1), (x0, y1, 1)]
    vtxt = "".join(f"({vx} {vy} {vz})" for vx, vy, vz in verts)
    # Hex face vertex-loops: x-lo/x-hi, y-lo/y-hi, z-lo/z-hi.
    xlo, xhi = "(0 4 7 3)", "(1 2 6 5)"
    ylo, yhi = "(0 1 5 4)", "(3 7 6 2)"
    zfaces = "(0 3 2 1) (4 5 6 7)"
    patches = [f"  {fn[0]} {{ type patch; faces ( {xlo} ); }}",
               f"  {fn[1]} {{ type patch; faces ( {xhi} ); }}"]
    if dim >= 2:
        patches += [f"  {fn[2]} {{ type patch; faces ( {ylo} ); }}",
                    f"  {fn[3]} {{ type patch; faces ( {yhi} ); }}",
                    f"  frontAndBack {{ type empty; faces ( {zfaces} ); }}"]
    else:
        patches += [f"  frontAndBack {{ type empty; faces ( {ylo} {yhi} ); }}",
                    f"  topAndBottom {{ type empty; faces ( {zfaces} ); }}"]
    (case / "system" / "blockMeshDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }\n"
        "convertToMeters 1;\n"
        f"vertices ( {vtxt} );\n"
        f"blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 1 1) );\n"
        "edges (); boundary (\n" + "\n".join(patches) + "\n); mergePatchPairs ();\n")
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
    """{name: value} for the controlDict ``modelParameters`` override.

    Source of truth is ``sm.parameter_values`` (the resolved Zstruct the model
    baked at derivation — e.g. VAM g=9.81, rho=1); the model-level
    ``parameter_values`` (user override) wins where present.  NEVER fabricate a
    0.0 for an unresolved name: the emitted ``Model::default_parameters()``
    already carries the baked defaults, and a zero override for g/rho SIGFPEs
    the run (division by rho in every pressure term)."""
    out = {}
    for src in (getattr(sm, "parameter_values", None),
                getattr(model, "parameter_values", None)):
        if src is None:
            continue
        for k in getattr(src, "keys", lambda: [])():
            try:
                out[str(k)] = float(getattr(src, k))
            except Exception:
                pass
    return out


def _build_case(case, mesh, model, sm, settings, binary):
    if case.exists():
        shutil.rmtree(case)
    (case / "0").mkdir(parents=True); (case / "system").mkdir(); (case / "constant").mkdir()
    lo, hi, n, order, dim = _mesh_geometry(mesh)
    face_names = settings.get("face_names") or _default_face_names(dim)
    _write_grid_system(case, lo, hi, n, dim, face_names)

    t_end = float(settings.get("time_end", 1.0))
    n_snap = int(settings.get("output_snapshots", 20)) or 1
    order_recon = int(settings.get("reconstruction_order", 1))
    scheme = settings.get("time_scheme", "explicit")
    maxco = float(settings.get("cfl", 0.4))
    dt0 = float(settings.get("min_dt", 1e-3)) or 1e-3
    # a-posteriori positivity: "mood" enables the local-MOOD wet/dry limiter in
    # the order-2 explicit path; "none" (default) leaves it off.
    positivity = str(settings.get("positivity", "none"))
    params = _model_parameters(model, sm)
    param_str = " ".join(f"{k} {v:g};" for k, v in params.items())
    imex = ("imexTableau ars232; imexMaxIter 20; imexTol 1e-12;"
            if scheme.lower().startswith("imex") else "")
    (case / "system" / "controlDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
        "application zoomyFoam;\n"
        f"startFrom startTime; startTime 0; stopAt endTime; endTime {t_end}; deltaT {dt0};\n"
        f"writeControl adjustableRunTime; writeInterval {t_end / n_snap:g}; purgeWrite 0;\n"
        # OpenFOAM's DEFAULT writePrecision is 6 significant digits, which caps
        # every exported field at ~3e-07 absolute on an O(0.3) value.  Measured
        # consequence: the lake-at-rest surface deviation read back as exactly
        # 4.8429e-07 at order 1 AND order 2, at t=1 s AND t=10 s -- a constant,
        # time- and order-independent floor that looks exactly like a lost
        # well-balancing but is pure output truncation (|u| meanwhile decayed to
        # 1e-15, i.e. the lake really was at rest).  Every reference in the test
        # suite is only as good as this number, so it is raised to full double.
        f"writePrecision 15;\n"
        # spaceDimension: the 1/d factor of the CFL law lives INSIDE
        # numerics::compute_dt (same formula as core), so maxCo is a pure
        # safety factor and 0.9 is the law in 1-D and 2-D alike.  The mesh
        # cannot supply d — OpenFOAM meshes a 1-D case as 3-D with `empty`
        # directions — so it is written here from _mesh_geometry.
        f"spaceDimension {dim};\n"
        f"maxCo {maxco}; reconstructionOrder {order_recon}; timeScheme {scheme}; "
        f"positivity {positivity}; {imex}\n"
        f"modelParameters {{ {param_str} }}\n")

    # 0/Qi from the model's initial conditions at the (ordered) inner cell centres
    nc = int(mesh.n_inner_cells)
    cc = np.asarray(mesh.cell_centers)[:, :nc]
    Q = np.zeros((len(sm.state), nc))
    Q = model.initial_conditions.apply(cc, Q)
    Q = np.asarray(Q)[:, order]
    patches = {**{fnm: "zeroGradient" for fnm in face_names}, "frontAndBack": "empty"}
    if dim == 1:
        patches["topAndBottom"] = "empty"
    for i in range(len(sm.state)):
        _field_file(case, f"Q{i}", Q[i], patches)
    return binary


# ── (d) run ─────────────────────────────────────────────────────────────────
def _assert_binary_matches_model(binary, reported_dof, expected_dof, case):
    """Hard-fail if the binary that just ran was not built from the Model.H we
    just generated.

    This is the second half of the stale-binary defence.  The first half stops a
    failed ``wmake`` from POISONING the cache (see :func:`_wmake_script`); this
    one catches a mismatch however it arose — a hand-copied binary, a cache entry
    predating a hash-input change, a partially-bound container.

    The check is cheap because the binary reports its own compiled-in DOF count
    at startup (``zoomy: n_dof_q = N`` in zoomyFoam.C).  That is the ONLY
    trustworthy source: the DOF count is baked in at compile time, so it cannot
    be faked by regenerating headers.  Comparing against Model.H on disk would
    prove nothing — Model.H is exactly the file the stale binary disagrees with.

    Why this must RAISE rather than warn: a stale binary is silent by
    construction.  It reads Q0..Qn-1 fine, writes back its OWN number of fields,
    and ``_strip_nonstate`` then deletes the surplus — so a 4-state binary run
    against a 3-state model exports a perfectly well-formed 3-state result that
    is physically wrong.  Nothing downstream can detect it."""
    if reported_dof is None:
        raise RuntimeError(
            f"zoomyFoam did not print its `zoomy: n_dof_q` build fingerprint — "
            f"the cached binary at {binary} predates the fingerprint banner and "
            f"cannot be verified against the generated Model.H (expected "
            f"n_dof_q = {expected_dof}). Delete {_BINCACHE} and rebuild; see "
            f"{case / 'run.log'}.")
    if reported_dof != expected_dof:
        raise RuntimeError(
            f"STALE BINARY: {binary} was compiled with Model::n_dof_q = "
            f"{reported_dof}, but the Model.H just generated for this run has "
            f"n_dof_q = {expected_dof}. The results in {case} are from the WRONG "
            f"model and must not be used. Delete {_BINCACHE} and rebuild.")


def _run_stream(case, binary, on_progress, nprocs=1, n_dof_q=None):
    """blockMesh → (optionally decomposePar → mpirun -parallel → reconstructPar)
    → stream the solver's per-step ``Time`` lines to ``on_progress``.

    With ``nprocs > 1`` the case is decomposed (scotch, so it works for any mesh
    dimension), run under ``mpirun -np N <binary> -parallel``, and reconstructed
    back to the serial time dirs the VTK/HDF5 export consumes.  The solver is
    dimension- and rank-agnostic (global dt via returnReduce, processor-patch
    fluxes via patchNeighbourField), so a decomposed run reproduces serial.

    ``n_dof_q`` is the state size of the model we just generated headers for.
    When given, the binary's SELF-REPORTED ``zoomy: n_dof_q = N`` banner is
    checked against it — see :func:`_assert_binary_matches_model`."""
    nprocs = int(nprocs or 1)
    _apptainer(f"cd {case}; blockMesh > log.blockMesh 2>&1", binds=[case], check=True)
    if nprocs > 1:
        (case / "system" / "decomposeParDict").write_text(
            "FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }\n"
            f"numberOfSubdomains {nprocs};\nmethod scotch;\n")
        _apptainer(f"cd {case}; decomposePar -force > log.decomposePar 2>&1",
                   binds=[case], check=True)
        run_cmd = f"cd {case}; mpirun -np {nprocs} '{binary}' -parallel -case {case}"
    else:
        run_cmd = f"cd {case}; '{binary}' -case {case}"
    _t_solver = time.perf_counter()
    p = subprocess.Popen(
        _apptainer_cmd(run_cmd, binds=[case]),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    log = (case / "run.log").open("w")
    it, prev_t = 0, 0.0
    # zoomyFoam prints `Time = <t>s` per step (adaptive dt, no separate deltaT
    # line) — derive dt from consecutive report times.  Under mpirun only the
    # master rank prints Info, unprefixed, so the same regex matches.
    time_re = re.compile(r"^Time = ([-\d.eE+]+)")
    # Both drivers self-report the state size COMPILED INTO them: zoomyFoam via
    # `zoomy: n_dof_q = N`, chorinFoam via its pre-existing
    # `chorinFoam: n_state=N` (the full shared state, which is what the chorin
    # pipeline generates fields for and exports).
    banner_re = re.compile(r"^(?:zoomy: n_dof_q = |chorinFoam: n_state=)(\d+)")
    reported_dof = None
    for line in p.stdout:
        log.write(line)
        b = banner_re.match(line.strip())
        if b:
            reported_dof = int(b.group(1))
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
    # The SOLVER's own wall time, excluding apptainer startup, blockMesh,
    # foamToVTK and the HDF5 pack.  The test suite budgets against THIS rather
    # than the whole-pipeline time: the pipeline is five subprocesses whose
    # scheduling swamps the measurement (measured 23% run-to-run spread with no
    # code change), while the solver loop is the number that actually tracks
    # scheme cost.  Written to a file rather than returned so no caller
    # signature changes.
    (case / "solver_wall.txt").write_text(f"{time.perf_counter() - _t_solver:.6f}\n")
    if p.returncode != 0:
        raise RuntimeError(f"zoomyFoam failed (rc={p.returncode}); see {case/'run.log'}")
    if n_dof_q is not None:
        _assert_binary_matches_model(binary, reported_dof, int(n_dof_q), case)
    if nprocs > 1:
        # Reconstruct the decomposed time dirs so the VTK/HDF5 export is identical
        # to a serial run.
        _apptainer(f"cd {case}; reconstructPar > log.reconstructPar 2>&1",
                   binds=[case], check=True)


# ── (e) VTK -> HDF5 ─────────────────────────────────────────────────────────
def _time_dirs(case):
    return sorted((d for d in case.iterdir()
                   if d.is_dir() and re.fullmatch(r"[0-9]+(\.[0-9]+)?", d.name)),
                  key=lambda d: float(d.name))


def _strip_nonstate(case, n_state, n_aux=0):
    """Keep only the state fields ``Q0..Q{n-1}`` in each written time dir, so the
    exported HDF5 ``Q`` is exactly the model state — not zoomyFoam's internal
    reconstruction diagnostics (``Dm*``/``Dp*``).

    ``n_aux > 0`` additionally keeps ``Qaux0..Qaux{n_aux-1}``, which the export
    then splits into a separate ``Qaux`` dataset.  Aux is dropped by DEFAULT
    because the server's shared-HDF5 contract is state-only; the test suite opts
    in, because a reference that pins Q alone cannot see a broken
    ``update_aux_variables`` (the auxes carry the reconstruction gradients and
    the bed slope, so an aux defect shows up in Q only after it has already
    corrupted the physics)."""
    keep = ({f"Q{i}" for i in range(n_state)}
            | {f"Qaux{i}" for i in range(n_aux)} | {"uniform"})
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


def _keep_state_rows(h5_path, n_state, vtks, n_aux=0):
    """Drop foamToVTK's synthetic ``cellID`` (and any non-state field): keep the
    ``Q0..Q{n-1}`` rows, in state order, in every frame's ``Q``.

    With ``n_aux > 0`` the ``Qaux0..Qaux{n_aux-1}`` rows are split off into a
    sibling ``Qaux`` dataset in the same frame group.

    Selection is BY NAME, never by position, and is resolved PER FRAME.

    Both of those matter and neither is theoretical:

    * foamToVTK does not emit cell_data in declaration order (observed:
      ``cellID, Qaux3, Q2, Qaux1, Qaux4, Qaux0, Q1, Qaux2, Q0``), so a
      positional slice would interleave aux rows into the state.
    * FRAMES ARE NOT HOMOGENEOUS.  Under ``nprocs > 1`` the reconstructed ``0/``
      time dir carries only ``Q0..Qn-1`` — ``reconstructPar`` does not backfill
      the aux into the pre-existing serial ``0/`` that ``decomposePar`` was
      seeded from, while every later time dir has the full set.  Resolving the
      index list once from frame 0 and applying it to every frame would
      therefore mis-index every later frame.
    """
    import h5py
    vtks = [str(v) for v in (vtks if isinstance(vtks, (list, tuple)) else [vtks])]
    with h5py.File(h5_path, "a") as f:
        keys = sorted(f["fields"], key=lambda k: int(k.split("_")[1]))
        for j, k in enumerate(keys):
            names = _vtk_field_names(vtks[min(j, len(vtks) - 1)])
            try:
                idx = [names.index(f"Q{i}") for i in range(n_state)]
            except ValueError:
                continue  # unexpected naming — leave this frame's Q untouched
            g = f["fields"][k]
            Q = g["Q"][:]
            del g["Q"]
            g.create_dataset("Q", data=Q[idx])
            if not n_aux:
                continue
            try:
                aux_idx = [names.index(f"Qaux{i}") for i in range(n_aux)]
            except ValueError:
                continue  # this frame has no aux (see the nprocs>1 note above)
            if "Qaux" in g:
                del g["Qaux"]
            g.create_dataset("Qaux", data=Q[aux_idx])


def _drop_auxless_times(case, n_aux):
    """Drop time dirs that do not carry the full aux set, so every exported frame
    has the SAME field list.

    Why this is necessary rather than tidy: ``zoomy_prepost.vtk_to_hdf5`` fixes
    the packed field schema from the FIRST frame, so one short frame silently
    truncates ``Q`` for the whole series (measured: every frame came back with 4
    rows instead of 9).

    Only the t=0 dir is ever short, and only under ``nprocs > 1``: it is the IC
    that ``_build_case`` wrote (state only), the solver's own startup aux write
    went to ``processor*/0/``, and ``reconstructPar`` does not overwrite an
    existing serial time dir.  Deleting ``0/`` and re-reconstructing is not an
    option — ``reconstructPar`` then fails outright (exit 1).
    """
    need = {f"Qaux{i}" for i in range(n_aux)}
    dirs = _time_dirs(case)
    for d in dirs[:-1]:                      # never drop the final frame
        if not need.issubset({f.name for f in d.iterdir()}):
            shutil.rmtree(d, ignore_errors=True)
    if not need.issubset({f.name for f in dirs[-1].iterdir()}):
        raise RuntimeError(
            f"the final time dir {dirs[-1]} carries no aux fields; cannot export "
            f"with_aux (looked for {sorted(need)})")


def _to_vtk(case, n_state, n_aux=0):
    """Strip to the state fields, `foamToVTK`, and write a `.pvd` collection with
    physical OF times. Returns ``(source, vtks)`` — ``source`` is the ``.pvd``
    path (or an ordered frame list on a count mismatch), ``vtks`` the sorted
    frame files."""
    _strip_nonstate(case, n_state, n_aux)
    if n_aux:
        _drop_auxless_times(case, n_aux)
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


def _to_hdf5(case, output_dir, n_state, n_aux=0):
    source, vtks = _to_vtk(case, n_state, n_aux)
    from zoomy_prepost import vtk_to_hdf5
    out = vtk_to_hdf5(source, str(Path(output_dir) / "simulation.h5"))
    _keep_state_rows(out, n_state, vtks, n_aux)
    return out


# ── public entry ────────────────────────────────────────────────────────────
def run_case(model, settings, output_dir, on_progress=None, with_aux=False):
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
    with_aux : bool
        Also export the model's aux state as a per-frame ``Qaux`` dataset.  Off
        by default so the server's state-only HDF5 contract is unchanged.
    """
    case, sm = _run_pipeline(model, settings, output_dir, on_progress)
    n_aux = len(sm.aux_state) if with_aux else 0
    return _to_hdf5(case, Path(output_dir), len(sm.state), n_aux)


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
    _run_stream(case, binary, on_progress, nprocs=int(settings.get("nprocs", 1)),
                n_dof_q=len(sm.state))
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
    ``(full_sm, n_state, n_pred_aux)``.

    ``n_pred_aux`` is the PREDICTOR's aux count (``Model::n_dof_qaux``), which is
    what chorinFoam writes as ``Qaux*``.  It is NOT ``len(full.aux_state)``: the
    predictor is a different sub-system with its own aux, and the printer appends
    the automatic ``hinv`` to whichever SystemModel it renders."""
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
    # AFTER write_code: the printer appends the automatic `hinv` aux in place, so
    # reading this before the call would under-count by one.
    return full, n_state, len(split.SM_pred.aux_state)


def _wmake_chorin_cached():
    """wmake the ``chorin_app`` (chorinFoam) for the current headers; cache by a
    hash of the split headers + driver.  Returns the cached binary path."""
    _BINCACHE.mkdir(exist_ok=True)
    h = hashlib.sha256()
    # ``numerics.H`` / ``Numerics.H`` are hashed too: chorinFoam.C #includes
    # numerics.H and calls ``numerics::compute_dt``, so an edit confined to
    # that header changes the chorin BINARY.  Omitting it meant a
    # numerics.H-only change silently reused a stale cached chorinFoam — the
    # same stale-binary hole ``_headers_hash`` was already hardened against
    # for zoomyFoam, never closed here.  Found while landing the CFL-law
    # change, which edits exactly numerics.H.
    for name in ("Model.H", "Pressure.H", "Corrector.H", "ChorinState.H",
                 "NumericsKernels.H", "Numerics.H", "numerics.H",
                 "chorin_app/chorinFoam.C"):
        f = FOAM_ROOT / name
        if f.exists():
            h.update(name.encode())
            h.update(f.read_bytes())
    cached = _BINCACHE / f"chorinFoam_{h.hexdigest()[:16]}"
    if cached.exists():
        return cached
    r = _apptainer(
        _wmake_script(f"{FOAM_ROOT}/chorin_app", _OF_CHORIN_BIN, cached),
        capture_output=True, text=True)
    if r.returncode != 0 or not cached.exists():
        raise RuntimeError(f"chorinFoam wmake failed:\n{r.stdout}\n{r.stderr}")
    return cached


def _build_chorin_case(case, mesh, model, sm, settings):
    if case.exists():
        shutil.rmtree(case)
    (case / "0").mkdir(parents=True); (case / "system").mkdir(); (case / "constant").mkdir()
    lo, hi, n, order, dim = _mesh_geometry(mesh)
    face_names = _default_face_names(dim)
    _write_grid_system(case, lo, hi, n, dim, face_names)

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
        f"writePrecision 15;\n"      # see the explicit builder: default 6 truncates at ~3e-07
        f"spaceDimension {dim};\n"      # see the explicit builder
        f"maxCo {maxco};\n"
        f"pressureSolver bicgstab; pressureTol {ptol:g}; pressureMaxIter {pmaxit};\n"
        f"modelParameters {{ {param_str} }}\n")

    nc = int(mesh.n_inner_cells)
    cc = np.asarray(mesh.cell_centers)[:, :nc]
    Q = np.zeros((len(sm.state), nc))
    Q = np.asarray(model.initial_conditions.apply(cc, Q))[:, order]
    patches = {**{fnm: "zeroGradient" for fnm in face_names}, "frontAndBack": "empty"}
    if dim == 1:
        patches["topAndBottom"] = "empty"
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

    full, n_state, _ = _codegen_chorin(model)
    binary = _wmake_chorin_cached()
    case = output_dir / "foam_case"
    _build_chorin_case(case, mesh, model, full, settings)
    _run_stream(case, binary, on_progress, nprocs=int(settings.get("nprocs", 1)),
                n_dof_q=n_state)
    return _to_vtk(case, n_state)[0]


def run_chorin_case(model, settings, output_dir, on_progress=None, with_aux=False):
    """Chorin analog of :func:`run_case`: same pipeline, exported to HDF5.

    ``run_chorin_to_vtk`` stops at the VTK series because that is the shape the
    gui wants.  The test suite needs the same ``(Q, Qaux)`` readback it gets for
    the hyperbolic path — otherwise the VAM/chorin pair would be the one pair
    compared on a weaker footing than every other test.
    """
    from zoomy_core.mesh.lsq_mesh import LSQMesh
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    mp = settings.get("mesh", "mesh.h5")
    if not os.path.isabs(mp):
        mp = os.path.join(settings.get("_case_dir", os.getcwd()), mp)
    mesh = LSQMesh.from_hdf5(mp)

    full, n_state, n_pred_aux = _codegen_chorin(model)
    binary = _wmake_chorin_cached()
    case = output_dir / "foam_case"
    _build_chorin_case(case, mesh, model, full, settings)
    _run_stream(case, binary, on_progress, nprocs=int(settings.get("nprocs", 1)),
                n_dof_q=n_state)
    n_aux = n_pred_aux if with_aux else 0
    return _to_hdf5(case, Path(output_dir), n_state, n_aux)
