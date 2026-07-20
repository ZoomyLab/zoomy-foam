"""Case content for the zoomy_foam test suite: ICs, BCs, the march helper and
the analytic comparisons.

Named ``foam_cases`` and not ``cases`` because ``tests/cases`` is already a
DIRECTORY of hand-written verification cases in this repo.

Two structural facts about the foam backend drive everything here:

1. **Boundary conditions are declared ON THE MODEL, before ``from_model``.**
   ``SystemModel.boundary_conditions`` is a compiled symbolic kernel built at
   ``SystemModel.from_model`` time, not a declaration slot.  ``models.py``
   therefore takes a hashable ``bc`` STRING (so its ``lru_cache`` keys) and
   builds the live objects here.  Initial conditions are likewise baked on the
   model, because foam writes ``0/Qi`` from ``model.initial_conditions`` — there
   is no post-build assignment hook the way jax has one.

2. **A foam march is a pipeline, not a call.**  codegen -> wmake (hash-cached)
   -> blockMesh -> zoomyFoam -> foamToVTK -> HDF5.  :func:`march` wraps it and
   returns the same ``(Q, Qaux)`` pair the jax helper returns, read back out of
   the exported HDF5 so the comparison is on the REAL round-tripped data rather
   than on an in-memory array the exporter never touched.
"""
from __future__ import annotations

import csv
import functools
import pathlib
import re

import numpy as np

G = 9.81

# SWASHES depths — the exact configuration the hand-built SWE's wet_dry_eps=1e-2
# cap silently zeroed (cid=54).  Matching the cached analytic tables exactly.
ETA_L, ETA_R, DAM_X = 0.005, 0.001, 5.0
SWASHES_DOMAIN = (0.0, 10.0)
SWASHES_T_END = 6.0

# The cached SWASHES analytic tables live in the thesis case; zoomy_foam always
# sits inside the superrepo, so this relative hop is stable.
SWASHES_REF_DIR = (pathlib.Path(__file__).resolve().parents[3]
                   / "thesis" / "cases" / "swe_swashes_verification"
                   / "reference")

# Escalante / bump (VAM chorin) geometry.
ESC_DOMAIN, ESC_NCELLS = (-1.5, 1.5), 60

# Order-2 dry-front negativity — a MEASURED FINDING, not a tuned tolerance.
#
# The order-2 reconstruction at the ritter dry front undershoots h below zero.
# Measured on the 20-cell twin (t = 0.5 s, CFL 0.9, ETA_L = 5e-3):
#
#     positivity none :  min h = -5.146e-07   (MOOD fired 0x)
#     positivity mood :  min h = -4.969e-12   (MOOD fired 2x)
#
# So the sanctioned a-posteriori limiter (REQ-175) removes five orders of
# magnitude but does NOT deliver exact non-negativity: a residual undershoot at
# ~1e-12 survives, which is 1e-9 RELATIVE to the initial depth, i.e. roundoff
# scale.  Without MOOD the undershoot is 1e-4 relative — genuinely negative
# water, not roundoff.
#
# The order-2 dry tests therefore run WITH mood and assert against this bound.
# This is NOT a floor and NOT a clip: h is never modified: the bound is an
# assertion threshold, and the exact value is stored in the reference so any
# drift fails the comparison.  Order 1 keeps the strict ``h >= 0``.
DRY_NEG_TOL = 1e-10


# ── boundary conditions ─────────────────────────────────────────────────────
def bcs_for(kind: str, dimension: int):
    """Live BC objects for a hashable kind string."""
    from zoomy_core.model import boundary_conditions as BC

    # The tags MUST match the patch names zoomy_foam gives the structured grid
    # (``_default_face_names``): 1-D keeps the historical left/right, 2-D uses the
    # compass tags.  A mismatch is not a soft failure — the solver's name-based
    # BC dispatch leaves the patch unbound and the first ``fvc::grad`` in
    # ``update_aux_variables`` SEGFAULTS (measured: rc=-11 with left/right/top/
    # bottom on a 2-D case).
    tags = ["left", "right"] if dimension == 2 else ["West", "East", "South", "North"]
    if kind in ("extrapolation", "swashes", "bump"):
        return BC.BoundaryConditions([BC.Extrapolation(tag=t) for t in tags])
    if kind == "wall":
        # FINDING (reported, not worked around silently): BC.Wall raises
        # sympy ShapeError "Dimensions incorrect for dot product: (2, 1), (1, 1)"
        # at SystemModel.from_model time for SME(level=0, dimension=2) — i.e. the
        # 1-D case — with or without on="momentum".  The SAME construction at
        # dimension=3 (2-D) builds fine, so the wall kernel's normal/momentum
        # contraction is wrong in 1-D specifically.  The 1-D lake-at-rest test
        # therefore uses "extrapolation": well-balancing is a property of the
        # bed-slope SOURCE term, not of the boundary, and on a lake at rest
        # nothing propagates to the boundary anyway — so the WB claim is
        # unaffected.  Wall is left reachable here for 2-D callers.
        return BC.BoundaryConditions([BC.Wall(tag=t) for t in tags])
    if kind == "periodic":
        return BC.BoundaryConditions(
            [BC.Periodic(tag="left", tag_partner="right"),
             BC.Periodic(tag="right", tag_partner="left")])
    raise KeyError(f"unknown BC kind {kind!r}")


# ── initial conditions ──────────────────────────────────────────────────────
# Foam applies ICs to the FULL cell-centre array at once (``model.initial_
# conditions.apply(cc, Q)``), so these are vectorised over x — unlike the jax
# per-point ICs.  The state layout is [b, h, q_0] for SME(level=0, dimension=2).
def stoker_ic(x):
    """Wet dam break (Stoker): both states wet, flat bed."""
    xc = np.asarray(x)[0]
    h = np.where(xc < DAM_X, ETA_L, ETA_R)
    return np.stack([np.zeros_like(h), h, np.zeros_like(h)])


def ritter_ic(x):
    """Dry dam break (Ritter): capless dry front, flat bed.  The dry side is
    EXACTLY zero — no floor anywhere (user law: never clip h)."""
    xc = np.asarray(x)[0]
    h = np.where(xc < DAM_X, ETA_L, 0.0)
    return np.stack([np.zeros_like(h), h, np.zeros_like(h)])


def lake_at_rest_ic(x):
    """Flat surface over a Gaussian bump — the topography gate.  Mass
    conservation is BLIND to well-balancing; this is not."""
    xc = np.asarray(x)[0]
    b = 0.1 * np.exp(-((xc - 5.0) ** 2) / 0.5)
    return np.stack([b, 0.3 - b, np.zeros_like(b)])


def gaussian_pulse_2d(x):
    """Radial pulse in a closed basin — exercises both horizontal dims.
    State layout for SME(level=0, dimension=3) is [b, h, q_0, q_1]."""
    xc = np.asarray(x)
    r2 = xc[0] ** 2 + xc[1] ** 2
    h = 0.3 + 0.05 * np.exp(-r2 / 0.05)
    z = np.zeros_like(h)
    return np.stack([z, h, z, z])


def bump_ic(x):
    """Subcritical bump for the VAM / chorin pair.

    VAM(level=1, dimension=2) carries the FULL 8-row state
    ``[b, h, q_0, q_1, r_0, r_1, P_0, P_1]`` — the chorin pipeline writes ``0/Qi``
    for every one of them, so an IC returning only ``[b, h, q]`` would leave the
    vertical-mode and pressure rows unwritten.  They start at rest.
    """
    xc = np.asarray(x)[0]
    b = np.where(np.abs(xc) < 0.5, 0.05 * np.cos(np.pi * xc) ** 2, 0.0)
    h = 0.34 - b
    z = np.zeros_like(h)
    return np.stack([b, h, z, z, z, z, z, z])


def ic_for(case: str):
    return {"stoker_wet": stoker_ic, "ritter_dry": ritter_ic}[case]


# ── the MANDATED pre-march operator print ───────────────────────────────────
def describe(sm, nsm=None) -> str:
    """Render the NSM operator matrices — the MANDATED pre-march print.

    Takes BOTH halves of the chain because they carry different things: the
    operator matrices live on the SystemModel, while the numerical model (the
    Riemann solver) wraps it as ``.model`` and contributes only the scheme
    identity.  Printing ``nsm.state`` would raise — the numerical model has no
    such attribute.
    """
    out = [
        "── NSM operator matrices (pre-march sanity print) ──",
        f"state: {list(sm.state)}",
        f"aux_state: {list(sm.aux_state)}",
        f"parameter_values: {getattr(sm, 'parameter_values', None)}",
        f"flux:\n{sm.flux}",
        f"nonconservative_matrix:\n{sm.nonconservative_matrix}",
        f"quasilinear_matrix:\n{getattr(sm, 'quasilinear_matrix', None)}",
        f"source:\n{sm.source}",
        f"eigenvalues:\n{getattr(sm, 'eigenvalues', None)}",
        f"update_variables (must be None — cap-free): "
        f"{getattr(sm, 'update_variables', None)}",
    ]
    if nsm is not None:
        out.append(f"numerical model: {type(nsm).__name__} "
                   f"(integration_order={getattr(nsm, 'integration_order', None)})")
    return "\n".join(out)


def chain(model):
    """Model -> SystemModel -> NumericalSystemModel, built EXPLICITLY.

    Requirement (3) of the design: every test shows the chain.  Foam's codegen
    builds its own copy internally (``_pipeline._codegen``), so what this
    returns is the chain the test PRINTS and asserts on; the run then regenerates
    it identically from the same model.  Returning both halves keeps the two
    assertions the design wants — ``sm.update_variables is None`` (cap-free) and
    the operator matrices — anchored to real objects rather than to a docstring.
    """
    from zoomy_core.systemmodel import SystemModel
    from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov

    sm = SystemModel.from_model(model)
    nsm = PositiveNonconservativeRusanov(model=sm)
    return sm, nsm


# ── the march ───────────────────────────────────────────────────────────────
def march(model, outdir, *, n_inner_cells, domain, t_end, cfl, order=1,
          nprocs=1, snapshots=2, dimension=1, extra_settings=None):
    """Run one foam case and read the FULL final state back out of the HDF5.

    Returns ``(Q, Qaux, info)`` where ``Q`` is ``(n_state, n_cells)``, ``Qaux``
    is ``(n_aux, n_cells)`` and ``info`` carries the solver-log step data.

    ``Qaux`` matters: a reference that pins Q alone cannot see a broken
    ``update_aux_variables``, because the auxes carry the reconstruction
    gradients and the bed slope and only reach Q after they have already
    corrupted the physics.  ``run_case(with_aux=True)`` is the opt-in export
    path added for exactly this.
    """
    from zoomy_foam import run_case
    import h5py

    mp = _write_mesh(outdir, domain, n_inner_cells, dimension)
    settings = {"mesh": str(mp), "time_end": t_end, "output_snapshots": snapshots,
                "cfl": cfl, "reconstruction_order": order, "nprocs": nprocs}
    settings.update(extra_settings or {})
    outdir = pathlib.Path(outdir)
    h5 = run_case(model, settings, str(outdir / "run"), with_aux=True)

    with h5py.File(h5, "r") as f:
        keys = sorted(f["fields"], key=lambda k: int(k.split("_")[1]))
        g = f["fields"][keys[-1]]
        Q = np.asarray(g["Q"], float)
        Qaux = np.asarray(g["Qaux"], float)

    info = solver_steps(outdir / "run" / "foam_case" / "run.log")
    info["case"] = outdir / "run" / "foam_case"
    return Q, Qaux, info


def read_raw_state(case, n_state, n_cells):
    """The FINAL state read straight from the OpenFOAM ASCII fields.

    WHY THIS EXISTS.  ``foamToVTK``'s legacy writer emits cell data as FLOAT32,
    so everything that comes back through the VTK -> HDF5 path is capped at ~1e-7
    RELATIVE.  That is fine for pinning references (``np.allclose`` defaults to
    rtol 1e-5) but it destroys any claim that needs machine precision.

    Measured on lake-at-rest: the surface deviation read back through HDF5 was a
    constant 4.84e-07 (OpenFOAM writePrecision 6), then 2.64e-08 after raising
    writePrecision to 15 (the float32 VTK floor) -- identical at order 1 and
    order 2, and identical at t = 1 s and t = 10 s.  A deviation that does not
    move with either the scheme order or the march length is not a scheme
    defect.  Read from the raw fields the SAME run gives 6.8e-12, i.e. the
    solver is well-balanced to machine precision and the rest was export
    truncation.

    Uses the library's own reader (``zoomy_plotting.column_plots.read_of_states``,
    which moved there from ``zoomy_core.postprocessing``).
    """
    from zoomy_plotting.column_plots import read_of_states

    T, Q = read_of_states(pathlib.Path(case), n_state, n_cells)
    return np.asarray(T, float), np.asarray(Q, float)[-1]


def _write_mesh(outdir, domain, n_inner_cells, dimension):
    """Write the case mesh; ``n_inner_cells`` is an int in 1-D, ``(nx, ny)`` in 2-D."""
    from zoomy_core.mesh.base_mesh import BaseMesh

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mp = outdir / "mesh.h5"
    if dimension == 1:
        m = BaseMesh.create_1d(domain=domain, n_inner_cells=int(n_inner_cells))
    else:
        nx, ny = n_inner_cells
        m = BaseMesh.create_2d(domain=domain, nx=int(nx), ny=int(ny))
    m.write_to_hdf5(str(mp))
    return mp


def march_chorin(model, outdir, *, n_inner_cells, domain, t_end, cfl,
                 snapshots=2, extra_settings=None):
    """Chorin-split (non-hydrostatic VAM) march — the chorinFoam app.

    Runs at the CASE-PROVEN CFL, not the hyperbolic law: VAM is measured stable
    only to ~0.15 and breaks at 0.20 on the dispersive modes.  That is a
    documented property of the model class, not a silent reduction of the law.
    """
    from zoomy_foam import run_chorin_case
    import h5py

    outdir = pathlib.Path(outdir)
    mp = _write_mesh(outdir, domain, n_inner_cells, 1)
    settings = {"mesh": str(mp), "time_end": t_end, "output_snapshots": snapshots,
                "cfl": cfl}
    settings.update(extra_settings or {})
    h5 = run_chorin_case(model, settings, str(outdir / "run"), with_aux=True)

    with h5py.File(h5, "r") as f:
        keys = sorted(f["fields"], key=lambda k: int(k.split("_")[1]))
        g = f["fields"][keys[-1]]
        Q = np.asarray(g["Q"], float)
        Qaux = np.asarray(g["Qaux"], float)
    return Q, Qaux, solver_steps(outdir / "run" / "foam_case" / "run.log")


def solver_steps(logpath) -> dict:
    """Step times and dt read from the SOLVER LOG.

    Deliberately NOT from the frame timestamps: consecutive HDF5 frames differ
    by the fixed ``writeInterval``, so a dt assertion on those would pass even
    for a solver taking garbage steps.
    """
    log = pathlib.Path(logpath).read_text()
    t = np.array([float(m.group(1))
                  for m in re.finditer(r"^Time = ([-\d.eE+]+)", log, re.M)])
    dts = np.diff(np.concatenate([[0.0], t])) if t.size else np.array([])
    dof = re.search(r"^zoomy: n_dof_q = (\d+)", log, re.M)
    wall = pathlib.Path(logpath).parent / "solver_wall.txt"
    return {"t": t, "dt": dts, "n_steps": int(t.size),
            "n_dof_q": int(dof.group(1)) if dof else None,
            "solver_wall": float(wall.read_text()) if wall.exists() else None}


def cell_centers(domain, n):
    lo, hi = domain
    dx = (hi - lo) / n
    return lo + dx * (np.arange(n) + 0.5), dx


# ── SWASHES analytic comparison ─────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def swashes_table(case: str) -> tuple:
    """The cached SWASHES analytic table (the library's own output).

    Generated by the ``swashes`` binary at t = 6 s over (0, 10) and cached in the
    thesis case.  We read the cache rather than shelling out, so the suite is
    reproducible without the binary — which is NOT installed on this machine
    (``which swashes`` -> not found), making the cache the only available truth.
    """
    path = SWASHES_REF_DIR / f"{case}.csv"
    if not path.exists():
        raise AssertionError(
            f"missing SWASHES analytic table {path} — regenerate it with "
            f"`python run_verification.py` in thesis/cases/"
            f"swe_swashes_verification (needs the `swashes` binary).")
    cols: dict = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return tuple(np.asarray(cols[k]) for k in ("x", "h"))


def l1_vs_analytic(Q, domain, case: str, t: float) -> float:
    """Mesh-normalized L1 error of h against the SWASHES analytic solution.

    ``t`` is ASSERTED against the table's time rather than used: the cached
    tables are t = 6 s only, and silently comparing a t = 1 s run against a
    t = 6 s table would manufacture a meaningless "error" that still converges.
    """
    assert abs(t - SWASHES_T_END) < 1e-12, (
        f"the cached SWASHES tables are t = {SWASHES_T_END} s only; got t = {t}. "
        f"Generate a new table before comparing at another time.")
    xr, hr = swashes_table(case)
    n = Q.shape[1]
    x, dx = cell_centers(domain, n)
    h_exact = np.interp(x, xr, hr)
    return float(np.sum(np.abs(np.asarray(Q[1], float) - h_exact) * dx)
                 / (dx * n))
