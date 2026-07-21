"""SWASHES ritter (DRY dam break) on the foam backend — order 1 and its order-2
small twin.

The dry side is EXACTLY zero. No floor, no clip, no wet/dry cap (user law): the
only permitted intervention is the automatic KP ``hinv`` sweep, which is an aux,
not a modification of h.
"""
import time

import numpy as np
import pytest

import foam_models as models
import foam_refs as refs
import zoomy_foam._pipeline as rc
from conftest import CFL
from foam_cases import (SWASHES_DOMAIN, ETA_L, assert_cfl_sets_dt,
                        cfl_witness, chain, describe, march, ritter_ic)

pytestmark = pytest.mark.skipif(
    not rc.SIF.exists(), reason=f"OpenFOAM apptainer image not found at {rc.SIF}")


@pytest.mark.small
@pytest.mark.foam
def test_ritter_dry(overwrite, tmp_path, capsys):
    model = models.swe(dimension=2, bc="swashes", ic=ritter_ic)
    sm, nsm = chain(model)
    with capsys.disabled():
        print(describe(sm, nsm))
    assert sm.update_variables is None

    t0 = time.perf_counter()
    Q, Qaux, info = march(model, tmp_path, n_inner_cells=100,
                          domain=SWASHES_DOMAIN, t_end=1.0, cfl=CFL, order=1)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all() and np.isfinite(Qaux).all()
    assert Q[1].min() >= 0.0, "negative depth — and NO floor is permitted"
    assert Q[1].max() <= ETA_L + 1e-12, "new maximum over the dry dam break"
    assert info["n_steps"] >= 2 and np.all(info["dt"] > 0.0)

    refs.check("ritter_dry", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("ritter_dry", elapsed, overwrite)


@pytest.mark.small
@pytest.mark.foam
def test_ritter_dry_o2_small(overwrite, tmp_path, capsys):
    """Small twin of the order-2 ritter convergence regression: same model, same
    reconstruction, 20 cells, full state stored.

    ``t_end``/``snapshots`` are chosen so the CFL law actually SETS dt.  At the
    original ``t_end=0.5, snapshots=2`` the writeInterval was 0.25 s while
    dt_CFL was 2.04 s, so every step clamped to the writer and this twin came
    back bit-identical across the CFL=0.9 change (see ``assert_cfl_sets_dt``).
    With ``t_end=8, snapshots=1`` the writeInterval is 8 s and dt is CFL-set:
    measured 24 steps, dt in [0.105, 2.0].  Halving the CFL to 0.45 gives 12
    steps and dt[0]=1.0, so the witness below discriminates.
    """
    T_END, SNAPS = 8.0, 1
    model = models.swe(dimension=2, bc="swashes", ic=ritter_ic)
    sm, nsm = chain(model)
    with capsys.disabled():
        print(describe(sm, nsm))
    assert sm.update_variables is None

    t0 = time.perf_counter()
    Q, Qaux, info = march(model, tmp_path, n_inner_cells=20,
                          domain=SWASHES_DOMAIN, t_end=T_END, cfl=CFL, order=2,
                          snapshots=SNAPS, extra_settings={"positivity": "mood"})
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all() and np.isfinite(Qaux).all()
    # THE CONTRACT, not the implementation: h >= 0, no tolerance.  This used to
    # assert against DRY_NEG_TOL = 1e-10, the SAME number as the MOOD detector's
    # own dead band — so the test could not see anything the detector was
    # already blind to.  The detector is strict now (bound emitted by core as
    # c_mood_h_bound = 0.0), so the test asserts the physics.  h is still never
    # floored: MOOD recomputes troubled cells from Q^n, it does not clip them.
    print(f"[dry-front] order-2 min h = {Q[1].min():.6e} (mood)")
    assert Q[1].min() >= 0.0, (
        f"negative depth {Q[1].min():.3e} at the order-2 dry front — no floor "
        "is permitted, and a strict MOOD detector is meant to catch this")
    # The CFL — not the output writer — must be what chose dt, and the achieved
    # step count / dt spread is pinned in the reference so a future CFL change
    # cannot come back bit-identical the way the 0.9 change did.
    assert_cfl_sets_dt(info, t_end=T_END, snapshots=SNAPS, label="ritter_o2_small")

    refs.check("ritter_dry_o2_small", overwrite, Q=Q, Qaux=Qaux,
               min_h=np.array([Q[1].min()]), **cfl_witness(info))
    refs.check_time("ritter_dry_o2_small", elapsed, overwrite)
