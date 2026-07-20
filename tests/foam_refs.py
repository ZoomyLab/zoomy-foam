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
SLOWER_OK = 1.10          # a test may get 10% slower; faster ratchets down

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
# This is a change to an approved design and is NOT taken unilaterally here, so
# SLOWER_OK stays at 1.10 and the mechanism is untouched. Options for the user,
# in preference order: (a) raise SLOWER_OK for foam to ~1.35, above the measured
# floor; (b) record a median of N runs instead of ratcheting to the minimum;
# (c) re-bless on an idle box and treat timing failures as advisory. Until one is
# chosen, expect timing failures whenever the box is under concurrent load.


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


def check_time(name, seconds, overwrite=False):
    """Fail if >10% slower than the recorded time; lower it if faster."""
    db = json.loads(TIMES.read_text()) if TIMES.exists() else {}
    ref = db.get(name)
    print(f"[time] {name}: {seconds:.2f} s (ref {ref})")
    if overwrite or ref is None or seconds < ref:
        db[name] = round(seconds, 3)
        DIR.mkdir(exist_ok=True)
        TIMES.write_text(json.dumps(dict(sorted(db.items())), indent=1))
        return
    assert seconds <= ref * SLOWER_OK, \
        f"{name}: {seconds:.2f}s vs {ref:.2f}s (+{100*(seconds/ref-1):.0f}%)"
