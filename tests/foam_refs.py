"""Reference data: one .npz per test, one timings.json.

    pytest                       # compare
    pytest --overwrite-results   # rewrite the references it touches

NAMED ``foam_refs`` rather than the design's ``refs``: ``zoomy_jax/tests/refs.py``
already exists, and with no ``__init__.py`` both would import as the top-level
module ``refs``.  Under a superrepo-wide pytest run the first one imported would
win and ``DIR`` would silently resolve to the OTHER backend's reference folder —
so the suites would compare against each other's data.  Same reasoning for
``foam_models`` and ``foam_cases``.

Otherwise ported verbatim from the user-approved jax design
(``2026-07-20-jax-test-suite-code.md``) — the framework is deliberately
identical across backends, so a foam reference and a jax reference are read the
same way.

One deliberate choice worth naming: a MISSING reference is written rather than
failing.  It keeps the code trivial, and the review gate is the git diff — a new
``refs/*.npz`` in a commit is visible and must be justified.  The risk it
accepts is that a first run of a broken test becomes its own reference, so new
references are always reviewed before commit.
"""
import json, pathlib, numpy as np

DIR = pathlib.Path(__file__).parent / "refs"
TIMES = DIR / "timings.json"
SLOWER_OK = 1.25          # user ruling 2026-07-21: 10% sat below this shared box's noise floor (23% spread measured on identical code); faster ratchets down

# ── PARKED, WITH MEASURED NUMBERS: the 10% budget is below foam's noise floor ──
#
# SLOWER_OK is KEPT at the user-approved 1.10 rather than quietly widened, but on
# THIS backend it does not fit, and the numbers say so.  Five back-to-back runs
# of test_stoker_wet with NO code change between them:
#
#     11.11, 9.71, 9.79, 9.04, 9.18  s   ->  min 9.04, median 9.71, spread 23%
#
# The ratchet records the MINIMUM ever seen (9.04) and then fails anything above
# 9.94, so a perfectly healthy run at the median trips it.  Observed live:
# "stoker_wet: 10.96s vs 9.07s (+21%)".
#
# WHY foam differs from jax, which this constant was calibrated on: a jax march is
# timed in-process, whereas every foam march pays apptainer container startup +
# blockMesh + the solver + foamToVTK + vtk_to_hdf5, i.e. five subprocesses whose
# scheduling dominates the measurement.  The variance is in the HARNESS, not in
# the physics the budget exists to protect.
#
# TIMING ONLY THE SOLVER DOES NOT FIX IT — I measured that too rather than
# assuming.  ``_run_stream`` now records the solver subprocess's own wall time
# (``solver_wall.txt``, surfaced as ``info["solver_wall"]``), excluding apptainer
# startup, blockMesh, foamToVTK and the HDF5 pack.  Five runs of the same case:
#
#     3.646, 3.659, 4.410, 4.263, 4.387  s   ->  spread 21%
#
# So the variance is ENVIRONMENTAL, not harness-shaped: this box is shared, and
# other agents were running zoomy_jax / zoomy_amrex workloads throughout. No
# choice of measurand gets under 10% while the machine is contended.
#
# A full compare-mode run is green (38 passed / 1 skipped / 1 xfailed, every DATA
# reference matching); an immediate rerun then failed 6 tests on timing alone
# (+10% to +32%) with zero physics failures. So the flakiness is confined to this
# assertion and never touches a correctness claim.
#
# RESOLVED 2026-07-21 (user ruling), and it is NOT option (a) alone. Two changes:
#   * SLOWER_OK raised 1.10 -> 1.25, above this box's noise floor;
#   * the ASSERTION is SCOPED to the `regression` / `large` tiers (see
#     `ASSERT_TIME` and conftest's `_timing_tier` fixture).
# Everything still MEASURES, PRINTS and RATCHETS everywhere — the user wants
# every wall time reported — but only the heavy tiers, whose runtime is a
# deliberate claim, can fail on it. This kills the failure mode where the
# ratchet drives the reference to the fastest run ever seen and a healthy
# median-speed run then trips the budget with identical physics.


def check(name, overwrite=False, **arrays):
    """Compare the given arrays against refs/<name>.npz, or write it."""
    p = DIR / f"{name}.npz"
    if overwrite or not p.exists():
        DIR.mkdir(exist_ok=True)
        np.savez_compressed(p, **arrays)
        print(f"[refs] wrote {p.name}")
        return
    ref = np.load(p)
    for k, v in arrays.items():
        assert np.allclose(v, ref[k]), \
            f"{name}.{k}: max|diff| {np.abs(v - ref[k]).max():.3e}"


#: Whether :func:`check_time` ASSERTS or merely records.  Set per test by the
#: ``_timing_tier`` autouse fixture in conftest: True only for the
#: ``regression`` / ``large`` tiers (user ruling 2026-07-21).  Default False so
#: a direct call outside pytest records rather than fails.
ASSERT_TIME = False


def check_time(name, seconds, overwrite=False):
    """Record the wall time, ratchet the reference down, assert on heavy tiers.

    MEASURE EVERYWHERE, FAIL NARROWLY (user ruling 2026-07-21).  Every test
    still prints its wall time and still ratchets the stored reference — the
    user wants every time reported.  What is scoped is the ASSERTION: only
    ``regression`` / ``large`` tests fail on a slowdown, because those are the
    ones whose runtime is a deliberate claim.  On the small gate,
    ratchet-to-minimum plus this box's measured 23% run-to-run spread makes the
    budget fire on unchanged physics — noise, not signal (see SLOWER_OK above).
    """
    db = json.loads(TIMES.read_text()) if TIMES.exists() else {}
    ref = db.get(name)
    gate = "assert" if ASSERT_TIME else "record-only"
    print(f"[time] {name}: {seconds:.2f} s (ref {ref}) [{gate}]")
    if overwrite or ref is None or seconds < ref:
        db[name] = round(seconds, 3)
        DIR.mkdir(exist_ok=True)
        TIMES.write_text(json.dumps(dict(sorted(db.items())), indent=1))
        return
    if not ASSERT_TIME:
        # Still SURFACE the regression — it is just not fatal on this tier.
        if seconds > ref * SLOWER_OK:
            print(f"[time] NOTE {name}: {seconds:.2f}s vs {ref:.2f}s "
                  f"(+{100*(seconds/ref-1):.0f}%) — not asserted on this tier")
        return
    assert seconds <= ref * SLOWER_OK, \
        f"{name}: {seconds:.2f}s vs {ref:.2f}s (+{100*(seconds/ref-1):.0f}%)"
