"""SME(i) <-> SME(j) preCICE coupling, AT TWO WINDOWS.

Three pairs, all through the canonical ``[b,h,u,v,w,p]`` interface:

    SME(1) <-> SME(1)   the user's "trivial" self-coupling
    SME(1) <-> SME(2)   adjacent levels
    SME(0) <-> SME(1)   inter-level

NAMING, because it has bitten before: the trivial case is ``sme0_sme1`` driven
with ``LEVEL1 = LEVEL2 = 1`` (the committed reference is ``snap_L1L1``).  It is
NOT the ``sme_self`` case — that is an older control which misses two fixes.

WHY TWO WINDOWS: window 1 is the preCICE start-up window and exercises nothing
(no peer data -> ghost falls back to own state -> zero interface flux -> both
participants come back bit-identical to their IC, and mass is conserved
trivially).  A one-window test cannot fail.  See ``foam_coupling`` for the
measured numbers and for why the coupled/monolithic comparison must be offset
by that one window.

Each pair asserts, at window 2:
  * ``preciceGhost characteristic;`` present in BOTH run artifacts (mandatory);
  * the start-up window is inert (Q and Qaux bit-identical to the IC);
  * the coupled pair reproduces the monolithic first step BIT-EXACTLY;
  * comparing at equal wall-clock time instead gives the spurious 8.8e-4 —
    pinned, so nobody "fixes" the offset back out;
  * total mass ``sum(h)*dx`` over BOTH participants = 15.0 exactly, drift 0,
    at every window;
  * full Q and Qaux of both participants against a blessed reference.

The small twin of each pair is the same march at ONE window: it is the control
that demonstrates the structural claim above (inert, and therefore not a
sufficient test), and it costs one window to run.
"""
import time

import numpy as np
import pytest

import foam_coupling as cpl
import foam_refs as refs
from foam_cases import describe

_ok, _why = cpl.available()
pytestmark = pytest.mark.skipif(not _ok, reason=_why)

#: (label, level of part1, level of part2)
PAIRS = [("L1L1", 1, 1), ("L1L2", 1, 2), ("L0L1", 0, 1)]


def _print_chain(capsys, level1, level2):
    """Model -> SystemModel -> NumericalSystemModel, EXPLICITLY, for both
    participants — the suite's standard shape."""
    with capsys.disabled():
        for tag, lvl in (("part1", level1), ("part2", level2)):
            model, sm, nsm = cpl.participant_chain(lvl)
            print(f"\n=== {tag}: {type(model).__name__}(level={lvl}) ===")
            print(describe(sm, nsm))
            assert sm.update_variables is None, "cap-free (user law)"
            cpl.assert_chain_matches_emitted(lvl, sm)


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.foam
@pytest.mark.parametrize("label,level1,level2", PAIRS)
def test_sme_level_coupling_two_windows(overwrite, tmp_path, capsys,
                                        label, level1, level2):
    _print_chain(capsys, level1, level2)

    case = cpl.stage(tmp_path, windows=2)
    t0 = time.perf_counter()
    cpl.run_pair(case, level1, level2)
    elapsed = time.perf_counter() - t0

    # The guard that must never be skipped: the exact documented failure was a
    # whole matrix silently run with the wrong ghost.
    cpl.assert_ghost_is_characteristic(case)

    # Window 1 exercises nothing — asserted, so the "two windows" claim stays
    # honest if preCICE ever changes.
    cpl.assert_startup_window_is_inert(case)

    # Window 2: the coupled pair IS the monolithic first step.
    off = cpl.compare_offset(case)
    same = cpl.spurious_same_clock(case)
    print(f"[coupling {label}] window2 vs mono step1: {off:.3e}   "
          f"(same wall-clock instead: {same:.3e})")
    assert off == 0.0, (
        f"{label}: coupled window 2 differs from the monolithic first step by "
        f"{off:.3e} — with characteristic ghosts this is bit-exact")
    assert same > 1e-5, (
        f"{label}: the same-wall-clock comparison came back {same:.3e}. That "
        "comparison is supposed to be WRONG by one start-up window; if it is "
        "now ~0 the offset has changed and compare_offset needs revisiting")

    # Mass over BOTH participants, at every window, exactly.
    for wdw in (0, 1, 2):
        m = cpl.total_mass(case, wdw)
        print(f"[mass {label}] window {wdw}: {m:.15f}  drift {m - cpl.MASS_BASELINE:.3e}")
        assert m == cpl.MASS_BASELINE, (
            f"{label} window {wdw}: total mass {m:.15f} != "
            f"{cpl.MASS_BASELINE} (drift {m - cpl.MASS_BASELINE:.3e})")

    # Full state of BOTH participants — Q AND Qaux.  Qaux matters: it carries
    # the reconstruction gradients and the bed slope, and a broken
    # update_aux_variables reaches Q only after it has already corrupted them.
    q1, a1 = cpl.state(case, "part1", cpl.tstr(2))
    q2, a2 = cpl.state(case, "part2", cpl.tstr(2))
    refs.check(f"coupling_{label}_w2", overwrite,
               Q_part1=q1, Qaux_part1=a1, Q_part2=q2, Qaux_part2=a2,
               mass=np.array([cpl.total_mass(case, 2)]))
    refs.check_time(f"coupling_{label}_w2", elapsed, overwrite)


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.foam
@pytest.mark.parametrize("label,level1,level2", PAIRS)
def test_sme_level_coupling_one_window_twin(tmp_path, capsys, label, level1, level2):
    """SMALL TWIN — and a NEGATIVE control, which is the whole point.

    One window is the preCICE start-up window: it returns both participants to
    their IC bit-identically and conserves mass trivially.  This twin asserts
    exactly that, so the suite CONTAINS the evidence that one window is not a
    test of the coupling — rather than only asserting it in a comment.
    """
    case = cpl.stage(tmp_path, windows=1)
    cpl.run_pair(case, level1, level2)

    cpl.assert_ghost_is_characteristic(case)
    cpl.assert_startup_window_is_inert(case)

    m0, m1 = cpl.total_mass(case, 0), cpl.total_mass(case, 1)
    print(f"[twin {label}] one window: mass {m0:.15f} -> {m1:.15f}")
    assert m0 == m1 == cpl.MASS_BASELINE

    # The claim, stated as an assertion: after ONE window the joined solution is
    # still exactly the initial condition, so this march cannot distinguish a
    # working interface from a broken one.
    assert np.array_equal(cpl.joined(case, 1), cpl.joined(case, 0)), (
        f"{label}: window 1 changed the joined solution — the start-up window "
        "is no longer inert, and the two-window rationale must be re-derived")
