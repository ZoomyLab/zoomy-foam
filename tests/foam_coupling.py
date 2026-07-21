"""Driver for the SME(i) <-> SME(j) preCICE coupling cases.

The three cases LIVE IN THE THESIS REPO
(``thesis/notebooks/coupling/cases/sme0_sme1``) and are read-only from here:
this module copies the case into ``tmp_path`` and patches only the march
length.  Nothing under ``thesis/`` is ever written.

WHY TWO WINDOWS IS THE MINIMUM — this is structural, not a margin
-----------------------------------------------------------------
Window 1 is the preCICE START-UP window.  Neither participant has peer data
yet, so the coupling ghost falls back to the participant's own state and the
interface flux is IDENTICALLY ZERO.  Both participants therefore come back
BIT-IDENTICAL to their initial condition, and mass is conserved trivially — a
one-window test passes even if the interface treatment is completely broken,
because the interface was never exercised.  Measured, all three pairs:
``max|h(window 1) - h(IC)| = 0.000e+00``.

At window 2 the exchanged data is real, and the coupled pair reproduces the
MONOLITHIC FIRST STEP bit-exactly.

AND THE OFFSET IS PART OF THE CLAIM.  Coupled window 2 corresponds to
monolithic step 1, not to monolithic ``t = 2*dt``: the start-up window costs
exactly one window of physical progress.  Comparing at equal wall-clock time
instead measures that offset and reports a spurious 8.824e-04 — which is a
property of the comparison, not of the coupling.  :func:`compare_offset` is
the correct comparison and :func:`spurious_same_clock` exists so a test can
PIN the trap rather than merely avoid it.

BUILD PATH: participant binaries come from ``library/zoomy_foam/create_model.py``
+ ``wmake`` (what the case's own ``run.sh`` does), NOT from
``cases/compile_sme.sh`` — that script is bit-rotted, its ``ROOT`` points at a
directory with no ``create_model.py``.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

import zoomy_foam._pipeline as rc

#: The case source, in the THESIS repo.  Read-only from this suite.
THESIS_CASE = (Path(rc.FOAM_ROOT).parent.parent
               / "thesis/notebooks/coupling/cases/sme0_sme1")

#: The case's own discretisation (generate.py): 200 cells over [0, 50].
N_TOTAL, X_MIN, X_MAX = 200, 0.0, 50.0
DX = (X_MAX - X_MIN) / N_TOTAL
N_PART = N_TOTAL // 2
DT = 5e-4

#: Dam: h = 0.5 on [0, 25], h = 0.1 on [25, 50]  ->  0.5*25 + 0.1*25 = 15.0.
#: The mass baseline is therefore EXACT in binary (both products are exact),
#: which is why the test can demand drift 0 rather than a tolerance.
MASS_BASELINE = 15.0

NCELLS = {"part1": N_PART, "part2": N_PART, "mono": N_TOTAL, "mono_l1": N_TOTAL}

_PYBIN = os.environ.get(
    "ZOOMY_COUPLING_PY",
    "/mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/micromamba/envs/zoomy/bin/python")


def participant_chain(level: int):
    """``(model, sm, nsm)`` for one participant, built EXPLICITLY.

    Mirrors what ``create_model.py --level L --closure newtonian`` emits for
    the coupling BC set, which is the binary ``run.sh`` actually builds — and
    :func:`assert_chain_matches_emitted` checks that claim rather than trusting
    this copy to stay in step.
    """
    from zoomy_core.model.models import SME
    from zoomy_core.model.models import closures as C
    from zoomy_core.model.boundary_conditions import (
        BoundaryConditions, Extrapolation, Coupled)
    from zoomy_core.systemmodel import SystemModel
    from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov

    model = SME(
        level=level, project_nz=40,
        closures=[C.Newtonian(), C.NavierSlip(), C.StressFree()],
        boundary_conditions=BoundaryConditions([
            Extrapolation(tag="outer"),
            Coupled(tag="coupled", mesh_name="interface")]))
    sm = SystemModel.from_model(model)
    return model, sm, PositiveNonconservativeRusanov(model=sm)


def assert_chain_matches_emitted(level: int, sm):
    """The chain the test prints must BE the participant's chain.

    ``create_model.build_system_model`` is the function ``run.sh`` calls, so
    comparing against it is what stops this module drifting into printing a
    model the coupled run never used.
    """
    import sys
    sys.path.insert(0, str(rc.FOAM_ROOT))
    from create_model import build_system_model

    emitted = build_system_model(level, closure="newtonian", bcs="coupling")
    assert [str(s) for s in sm.state] == [str(s) for s in emitted.state], (
        f"level {level}: the printed chain's state {[str(s) for s in sm.state]} "
        f"is not what create_model.py emits {[str(s) for s in emitted.state]}")


def available() -> tuple[bool, str]:
    """Can this suite run the coupling pair at all?"""
    if not rc.SIF.exists():
        return False, f"OpenFOAM apptainer image not found at {rc.SIF}"
    if not THESIS_CASE.is_dir():
        return False, f"thesis coupling case not found at {THESIS_CASE}"
    if not Path(_PYBIN).exists():
        return False, f"zoomy python not found at {_PYBIN} (set ZOOMY_COUPLING_PY)"
    return True, ""


# ── staging ─────────────────────────────────────────────────────────────────
def stage(tmp_path: Path, *, windows: int = 2) -> Path:
    """Copy the thesis case into ``tmp_path`` and shorten the march.

    Only ``T_END`` and ``WRITE`` are patched — the physics, the BCs, the ghost
    and the coupling scheme are the case's own.  ``WRITE = DT`` so every window
    is written and the per-window comparison has data to read.
    """
    dst = tmp_path / "sme_levels"
    shutil.copytree(THESIS_CASE, dst,
                    ignore=shutil.ignore_patterns(
                        "bin", "precice-run", "results_*", "snap_*", "*.gif",
                        "__pycache__", "mono", "mono_l1", "part1", "part2"))
    gen = dst / "generate.py"
    src = gen.read_text()
    patched = src.replace("T_END, DT = 1.0, 5e-4",
                          f"T_END, DT = {windows * DT!r}, {DT!r}")
    assert patched != src, "generate.py no longer declares `T_END, DT = 1.0, 5e-4`"
    src, patched = patched, patched.replace("WRITE = 0.02", f"WRITE = {DT!r}")
    assert patched != src, "generate.py no longer declares `WRITE = 0.02`"
    gen.write_text(patched)
    return dst


def run_pair(case: Path, level1: int, level2: int, *, timeout: int = 2400):
    """Emit both levels, wmake both participants, mesh, run mono + the pair."""
    script = (f"ZOOMY_FOAM={rc.FOAM_ROOT} ZOOMY_PY={_PYBIN} "
              f"bash run.sh parallel-explicit {level2} {level1}")
    r = subprocess.run(
        ["apptainer", "exec",
         "--bind", "/mnt/userdrive:/mnt/userdrive", "--bind", "/tmp:/tmp",
         "--bind", f"{rc.FOAM_ROOT}:{rc.FOAM_ROOT}",
         str(rc.SIF), "bash", "-lc",
         f"source /opt/openfoam13/etc/bashrc 2>/dev/null; cd {case}; {script}"],
        capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(
            f"coupling run L{level1}<->L{level2} failed (rc={r.returncode})\n"
            f"{r.stdout[-4000:]}\n{r.stderr[-2000:]}")
    return r


# ── reading OpenFOAM ascii fields ───────────────────────────────────────────
def read_field(path: Path, n: int) -> np.ndarray:
    """One ascii ``volScalarField`` internal field, ALWAYS length ``n``.

    The ``uniform`` form carries ONE number for the whole mesh; it must be
    expanded to ``n`` cells or any sum over it is silently wrong.  (Measured
    while writing this: leaving it un-expanded made the t=0 mass read 0.15
    instead of 15.0 — the mass assertion would have been meaningless.)
    """
    t = Path(path).read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n"
                  r"(\d+)\s*\n\((.*?)\n\)", t, re.S)
    if m:
        a = np.fromstring(m.group(2), sep="\n")
        assert a.size == int(m.group(1)) == n, (
            f"{path}: got {a.size} values, header says {m.group(1)}, expected {n}")
        return a
    m = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)", t)
    if m:
        return np.full(n, float(m.group(1)))
    raise ValueError(f"no internalField in {path}")


def _stack(case: Path, part: str, time: str, prefix: str) -> np.ndarray:
    d = case / part / time
    names = sorted((p.name for p in d.glob(f"{prefix}[0-9]*")),
                   key=lambda s: int(s[len(prefix):]))
    assert names, f"no {prefix}* fields in {d}"
    return np.array([read_field(d / nm, NCELLS[part]) for nm in names])


def state(case: Path, part: str, time: str):
    """``(Q, Qaux)`` for one participant at one written time."""
    return _stack(case, part, time, "Q"), _stack(case, part, time, "Qaux")


def tstr(window: int) -> str:
    """The written time-directory name for window ``k`` (0 = IC)."""
    return "0" if window == 0 else repr(round(window * DT, 12))


# ── the assertions ──────────────────────────────────────────────────────────
def assert_ghost_is_characteristic(case: Path):
    """MANDATORY GUARD.  ``preciceGhost characteristic;`` must be in the RUN
    ARTIFACT of both participants.

    Not a style check: running the pair with the wrong ghost is the exact
    documented failure that silently produced a whole matrix of results.  The
    assertion reads the generated controlDict, i.e. what the solver actually
    consumed, not the generator's source.
    """
    for part in ("part1", "part2"):
        cd = (case / part / "system/controlDict").read_text()
        assert "preciceGhost characteristic;" in cd, (
            f"{part}/system/controlDict does not carry "
            f"`preciceGhost characteristic;` — the coupled run is invalid")


def joined(case: Path, window: int) -> np.ndarray:
    """Depth over the joined domain, part1 ++ part2, at ``window``."""
    t = tstr(window)
    return np.concatenate([read_field(case / "part1" / t / "Q1", N_PART),
                           read_field(case / "part2" / t / "Q1", N_PART)])


def total_mass(case: Path, window: int) -> float:
    """``sum(h)*dx`` over BOTH participants."""
    return float(joined(case, window).sum() * DX)


def assert_startup_window_is_inert(case: Path):
    """Window 1 must return both participants' STATE to their IC, bit-identically.

    Asserted rather than merely documented, so the reason two windows are
    required stays true: if a future preCICE version delivers real data in
    window 1, this fails and the one-window argument must be revisited.

    ON Q ONLY, and the exclusion is measured, not convenient.  ``Qaux`` is
    written at ``t=0`` as the INITIAL PLACEHOLDER — the cases set
    ``aux_initial_conditions`` to zeros — and is first computed during step 1,
    so it changes in the start-up window for a reason that has nothing to do
    with coupling (measured: max|dQaux| = 2.0 on part1, 10.0 on part2, against
    an all-zero t=0 field, while max|dQ| = 0.000e+00 on both).  The all-zero
    precondition is asserted below so this stays an explained exclusion: if
    Qaux at t=0 ever becomes a real field, this trips and must be reconsidered.
    """
    for part in ("part1", "part2"):
        (q0, a0), (q1, _) = state(case, part, tstr(0)), state(case, part, tstr(1))
        assert np.array_equal(q0, q1), (
            f"{part}: Q changed during the preCICE start-up window "
            f"(max|dQ| = {np.abs(q0 - q1).max():.3e}) — window 1 is supposed "
            "to see no peer data and produce zero interface flux")
        assert np.all(a0 == 0.0), (
            f"{part}: Qaux at t=0 is no longer the all-zero placeholder, so it "
            "can no longer be excluded from the start-up-window comparison")


def compare_offset(case: Path):
    """Coupled window 2 vs the MONOLITHIC FIRST STEP.  Returns max|diff| on h.

    The one-window offset is the point: the start-up window costs exactly one
    window of physical progress, so coupled window k lines up with monolithic
    step k-1.
    """
    mono = read_field(case / "mono" / tstr(1) / "Q1", N_TOTAL)
    return float(np.abs(joined(case, 2) - mono).max())


def spurious_same_clock(case: Path):
    """The WRONG comparison, kept so a test can pin it: coupled window 2
    against monolithic ``t = 2*dt``.  Measured 8.824e-04 on all three pairs."""
    mono = read_field(case / "mono" / tstr(2) / "Q1", N_TOTAL)
    return float(np.abs(joined(case, 2) - mono).max())
