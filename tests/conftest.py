"""Test-tier plumbing for the zoomy_foam suite.

Port of the jax suite's conftest (``2026-07-20-jax-test-suite-code.md``).  The
SHAPE is fixed by that design; only the tech adapts:

  * jax marches in-process and can force 2 CPU devices before the first import.
    Foam cannot — parallelism is ``decomposePar`` + ``mpirun -np N`` inside the
    apptainer, so the "2 device" twin becomes a 2-RANK twin driven through
    ``settings["nprocs"]``.
  * jax's ``march(nsm, mesh, ...)`` returns arrays directly.  Foam's march is a
    whole pipeline (codegen -> wmake -> blockMesh -> solver -> foamToVTK ->
    HDF5), so the equivalent helper lives in ``foam_cases.march`` and returns the
    same ``(Q, Qaux)`` pair read back out of the exported HDF5.

The module is named ``foam_cases`` rather than ``cases`` because ``tests/cases``
is already a DIRECTORY of hand-written verification cases in this repo.
"""
from __future__ import annotations

import os
import sys
import pathlib

import numpy as np
import pytest

# Make the sibling helper modules importable however pytest was invoked (from
# this repo, from the superrepo root, or under importmode=importlib, none of
# which reliably put this directory on sys.path).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


# ── CFL law (user law — never silently reduced) ─────────────────────────────
CFL_1D, CFL_2D = 0.9, 0.45

# VAM / non-hydrostatic models carry their OWN documented limit: measured stable
# only to ~0.15, breaking at 0.20 on the dispersive modes.  They therefore run at
# the CASE-PROVEN CFL, and the law-CFL behaviour is REPORTED rather than silently
# accommodated.  This is not an augmentation of the hyperbolic law — it is a
# different model class with its own measured stability bound.
CFL_VAM = 0.15

# Smooth-problem order floors.  ONLY applied to smooth problems: the SWASHES
# rates are RECORDED and compared, never floored (see test_swashes_convergence).
ORDER_FLOOR = {1: 0.9, 2: 1.9}


def pytest_addoption(parser):
    g = parser.getgroup("zoomy test tiers")
    g.addoption("--overwrite-results", action="store_true", default=False,
                help="rewrite the reference .npz/timings the tests touch")
    g.addoption("--run-large", action="store_true", default=False,
                help="also run the `large` tier")


def pytest_configure(config):
    for m in ("small", "regression", "large", "foam"):
        config.addinivalue_line("markers", f"{m}: zoomy test tier / area tag")


def pytest_collection_modifyitems(config, items):
    """Default ``pytest`` runs the SMALL gate tier only.

    Implemented as "deselect the heavy tiers" rather than "select only
    ``small``" for one concrete reason: this repo already had 23 passing tests
    before this suite existed, and none of them carry a tier marker.  They are
    fast and they ARE the historic gate, so requiring an explicit ``small``
    marker on every one of them would either silently retire them from the
    default run or force a marker-only edit across ten unrelated files.
    Unmarked therefore means "small"; only ``regression`` / ``large`` opt out.

    ``-m regression`` selects the reference marches; ``large`` additionally
    needs ``--run-large`` so a bare ``-m regression`` cannot accidentally start
    a multi-minute march.
    """
    if not config.getoption("-m") and not config.getoption("--run-large"):
        skip = pytest.mark.skip(reason="heavy tier: use -m regression / --run-large")
        for it in items:
            if "regression" in it.keywords or "large" in it.keywords:
                it.add_marker(skip)
    if not config.getoption("--run-large"):
        skip_large = pytest.mark.skip(reason="large tier needs --run-large")
        for it in items:
            if "large" in it.keywords:
                it.add_marker(skip_large)


@pytest.fixture
def overwrite(request):
    return (request.config.getoption("--overwrite-results")
            or os.environ.get("ZOOMY_OVERWRITE_RESULTS") == "1")


def fit_order(sizes, errors):
    """Fitted convergence rate from a resolution sweep (least squares in log)."""
    return float(-np.polyfit(np.log(sizes), np.log(errors), 1)[0])


def restrict(fine):
    """Conservative fine -> coarse restriction; exact for cell averages."""
    return 0.5 * (fine[:, 0::2] + fine[:, 1::2])
