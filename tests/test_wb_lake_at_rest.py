"""Well-balancing over topography: lake at rest on a Gaussian bump.

BC NOTE: these use extrapolation, not wall.  ``BC.Wall`` raises a sympy
ShapeError in 1-D (see ``foam_cases.bcs_for``) — a reported core finding.  It
does not weaken the claim: well-balancing is a property of the bed-slope SOURCE
term, and on a lake at rest nothing ever propagates to the boundary.

MASS CONSERVATION IS BLIND TO WELL-BALANCING.  A flat-bed suite structurally
cannot see a lost bed-slope treatment — a lake can tear itself apart at 1e-16
mass drift.  The gate is the free SURFACE staying flat and the velocity staying
zero over a non-trivial bed, which is what these two tests assert.
"""
import time

import numpy as np
import pytest

import foam_models as models
import foam_refs as refs
import zoomy_foam._pipeline as rc
from conftest import CFL_1D
from foam_cases import chain, describe, lake_at_rest_ic, march, read_raw_state

pytestmark = pytest.mark.skipif(
    not rc.SIF.exists(), reason=f"OpenFOAM apptainer image not found at {rc.SIF}")

DOMAIN = (0.0, 10.0)


def _wb_metrics(Q):
    """Surface flatness and spurious velocity — the two WB observables."""
    b, h, q = Q[0], Q[1], Q[2]
    eta = b + h
    return float(np.abs(eta - eta[0]).max()), float(np.abs(q / h).max())


# The WB claim is asserted on the RAW OpenFOAM fields, not on the HDF5 readback.
# foamToVTK's legacy writer stores cell data as FLOAT32, which puts a ~1e-7
# relative floor under everything that comes back through the export path — and a
# well-balancing assertion is exactly the claim that floor would destroy.
#
# MEASURED on this case, same run, three readback paths:
#     HDF5, writePrecision 6   : d_eta 4.8429e-07   (OpenFOAM ASCII default)
#     HDF5, writePrecision 15  : d_eta 2.6414e-08   (the float32 VTK floor)
#     raw OpenFOAM fields      : d_eta 6.8249e-12   (the actual solver state)
# The first two are IDENTICAL at order 1 and order 2 and at t = 1 s and t = 10 s.
# A deviation that moves with neither the scheme order nor the march length is
# not a scheme defect, and the raw read confirms it: zoomyFoam IS well-balanced.
WB_ETA_TOL = 1e-10      # raw-field bound; measured 6.8e-12
WB_U_TOL = 1e-9         # spurious velocity; measured 6.7e-11 at t=1, 4.7e-15 at t=10


@pytest.mark.small
@pytest.mark.foam
def test_lake_at_rest_over_bump(overwrite, tmp_path, capsys):
    model = models.swe(dimension=2, bc="extrapolation", ic=lake_at_rest_ic)
    sm, nsm = chain(model)
    with capsys.disabled():
        print(describe(sm, nsm))
    assert sm.update_variables is None

    t0 = time.perf_counter()
    Q, Qaux, info = march(model, tmp_path, n_inner_cells=100, domain=DOMAIN,
                          t_end=1.0, cfl=CFL_1D, order=1)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all() and np.isfinite(Qaux).all()
    # The bed must actually be non-trivial, or the test is a flat-bed test in
    # disguise and proves nothing about well-balancing.
    assert Q[0].max() - Q[0].min() > 1e-3, "bed is flat — WB not exercised"

    _, Qraw = read_raw_state(info["case"], len(sm.state), 100)
    d_eta, u_max = _wb_metrics(Qraw)
    print(f"[wb] raw-field surface deviation {d_eta:.3e}, |u|max {u_max:.3e} "
          f"(HDF5 readback would show {_wb_metrics(Q)[0]:.3e} — float32 export floor)")
    assert d_eta < WB_ETA_TOL, f"lake tilted — WB lost (deviation {d_eta:.3e})"
    assert u_max < WB_U_TOL, f"spurious currents over the bed ({u_max:.3e})"

    refs.check("wb_lake", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("wb_lake", elapsed, overwrite)


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.foam
def test_wb_drift_long_march(overwrite, tmp_path, capsys):
    """The long march: WB defects that are invisible in 1 s accumulate.

    Order 2, 200 cells, t = 50 s — the regression twin of the small test above.
    """
    model = models.swe(dimension=2, bc="extrapolation", ic=lake_at_rest_ic)
    sm, nsm = chain(model)
    with capsys.disabled():
        print(describe(sm, nsm))

    t0 = time.perf_counter()
    Q, Qaux, info = march(model, tmp_path, n_inner_cells=200, domain=DOMAIN,
                          t_end=50.0, cfl=CFL_1D, order=2, snapshots=10)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all() and np.isfinite(Qaux).all()
    _, Qraw = read_raw_state(info["case"], len(sm.state), 200)
    d_eta, u_max = _wb_metrics(Qraw)
    print(f"[wb-long] t=50 s raw-field: surface deviation {d_eta:.3e}, "
          f"|u|max {u_max:.3e}, {info['n_steps']} steps")
    # The load-bearing claim over a long march is that the deviation does not
    # GROW: a genuine WB defect accumulates with every step, and 50 s at order 2
    # is ~2000 steps.
    assert d_eta < WB_ETA_TOL, f"surface drift {d_eta:.2e} at t = 50 s"
    assert u_max < WB_U_TOL, f"spurious currents {u_max:.2e} at t = 50 s"

    refs.check("wb_long", overwrite, Q=Q, Qaux=Qaux,
               drift=np.array([d_eta]), umax=np.array([u_max]))
    refs.check_time("wb_long", elapsed, overwrite)
