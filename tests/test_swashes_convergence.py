"""SWASHES convergence on the foam backend — stoker-wet and ritter-dry at
order 1 and order 2.

THE RATES ARE RECORDED AND COMPARED, NOT FLOORED AT THEIR NOMINAL ORDER.  That
is a property of the PROBLEMS, not of the solver:

  * ``stoker_wet`` carries a SHOCK.  Its intermediate state is subcritical
    (measured Fr = 0.81), so u* - c* < 0 and the O(dx) error made at the shock
    travels UPSTREAM along C- through the plateau and into the fan
    (Engquist-Sjogreen / Casper-Carpenter shock pollution).  Region
    decomposition proves the cap is GLOBAL: excising a 1 m band around the shock
    still gives 1.06, and the smooth fan interior alone gives 0.88.  Order 2 is
    therefore capped near 1.0 — the textbook L1 ceiling for a shocked problem.
  * ``ritter_dry`` has a dry front, which caps it for its own reason.

Second order is ASSERTED only on smooth problems.  Here a collapse floor
(rate > 0.4) catches a real regression without pretending a shocked case can
reach 2, and the fitted rate itself is stored in the reference so any DRIFT in
the rate fails the comparison even though the rate is not floored at 2.
"""
import time

import numpy as np
import pytest

import foam_models as models
import foam_refs as refs
import zoomy_foam._pipeline as rc
from conftest import CFL, fit_order
from foam_cases import (SWASHES_DOMAIN, SWASHES_T_END, chain,
                        describe, ic_for, l1_vs_analytic, march)

pytestmark = pytest.mark.skipif(
    not rc.SIF.exists(), reason=f"OpenFOAM apptainer image not found at {rc.SIF}")

# N, 2N, 4N.  The cached SWASHES tables are t = 6 s only, so the march time is
# fixed by the reference data, not chosen.
SIZES = [100, 200, 400]


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.foam
@pytest.mark.parametrize("case", ["stoker_wet", "ritter_dry"])
@pytest.mark.parametrize("order", [1, 2])
def test_swashes_order(overwrite, tmp_path, capsys, case, order):
    model = models.swe(dimension=2, bc="swashes", ic=ic_for(case))
    sm, nsm = chain(model)
    with capsys.disabled():
        print(f"\n=== SWASHES {case} order {order} ===")
        print(describe(sm, nsm))
    assert sm.update_variables is None

    # Order 2 + dry front undershoots h; MOOD (REQ-175) is the sanctioned
    # a-posteriori limiter and is a proven no-op on healthy wet flow, so it is
    # enabled only where it is needed and never silently for everything.
    dry_o2 = (case == "ritter_dry" and order == 2)
    extra = {"positivity": "mood"} if dry_o2 else None
    # THE CONTRACT, in every case: h >= 0, no tolerance.  The dry-order-2 branch
    # used to be exempted to -DRY_NEG_TOL (1e-10), which was exactly the MOOD
    # detector's old dead band — the test agreed with the detector's blind spot
    # instead of checking the physics.  Detector is strict now (c_mood_h_bound,
    # emitted by core), so the exemption is gone.
    h_floor = 0.0

    # ── OPEN FINDING (2026-07-21), reported, NOT worked around ───────────────
    # `[2-ritter_dry]` ABORTS at n = 400 with SIGFPE (rc = -8) since the MOOD
    # detector became strict.  It is a pre-existing defect that the dead band
    # was HIDING, not a regression introduced by the strict bound:
    #
    #   * pre-change (dead band -1e-10) the same march completes, rate 1.004;
    #   * post-change it aborts, and the solver log shows the mechanism:
    #         [MOOD] troubled = 9
    #         [MOOD] WARNING still troubled = 5 after override    <- "should
    #         [MOOD] troubled = 11                                    never fire"
    #       then powf64 raises FE_INVALID inside libm.
    #
    # Two separate defects, neither of them in this test:
    #   1. zoomyFoam.C's apply_mood assumes ONE pass suffices ("it cannot seed
    #      a new one").  The WARNING firing proves that assumption false — the
    #      O1 override does not restore positivity here, so negative h survives
    #      into Q^n.
    #   2. the emitted eigenvalue slot evaluates sqrt(g)*sqrt(h**5)/h**2, which
    #      is NaN/FE_INVALID for h < 0.  That expression comes from zoomy_core
    #      (regularize_pow / the eigenvalue slot), not from this backend.
    #
    # Deliberately NOT "fixed" here by widening the bound back (that IS the
    # defect), by flooring h (forbidden by the user law), or by xfail (that
    # would hide a live physics defect behind a green suite).  The abort is the
    # honest signal.  Fixing (2) needs the zoomy_core owner.

    errs, Q, Qaux = [], None, None
    t0 = time.perf_counter()
    for n in SIZES:
        Q, Qaux, info = march(model, tmp_path / f"n{n}", n_inner_cells=n,
                              domain=SWASHES_DOMAIN, t_end=SWASHES_T_END,
                              cfl=CFL, order=order, extra_settings=extra)
        assert np.isfinite(Q).all(), f"non-finite state at n = {n}"
        assert Q[1].min() >= h_floor, (
            f"depth {Q[1].min():.3e} at n = {n} below the permitted bound "
            f"{h_floor:.1e} — h is never floored, this is an assertion")
        errs.append(l1_vs_analytic(Q, SWASHES_DOMAIN, case, t=SWASHES_T_END))
    elapsed = time.perf_counter() - t0

    rate = fit_order(SIZES, errs)
    print(f"[convergence] {case} order {order}: N {SIZES}, "
          f"L1 {['%.4e' % e for e in errs]}, fitted rate {rate:.3f}")

    assert rate > 0.4, f"{case} order {order}: rate {rate:.3f} — collapsed"
    refs.check(f"swashes_{case}_o{order}", overwrite, Q=Q, Qaux=Qaux,
               N=np.array(SIZES), l1=np.array(errs), rate=np.array([rate]))
    refs.check_time(f"swashes_{case}_o{order}", elapsed, overwrite)
