"""SWASHES stoker (WET dam break) on the foam backend — order 1 and its order-2
small twin.

CFL is the 1-D law value 0.9 and is NEVER silently reduced: instability at the
law CFL is a REPORTED FINDING, not a knob.
"""
import time

import numpy as np
import pytest

import foam_models as models
import foam_refs as refs
import zoomy_foam._pipeline as rc
from conftest import CFL
from foam_cases import (SWASHES_DOMAIN, ETA_L, ETA_R, assert_cfl_sets_dt,
                        cfl_witness, chain, describe, march, stoker_ic)

pytestmark = pytest.mark.skipif(
    not rc.SIF.exists(), reason=f"OpenFOAM apptainer image not found at {rc.SIF}")


@pytest.mark.small
@pytest.mark.foam
def test_stoker_wet(overwrite, tmp_path, capsys):
    model = models.swe(dimension=2, bc="swashes", ic=stoker_ic)  # Model
    sm, nsm = chain(model)                    # SystemModel -> NumericalSystemModel
    with capsys.disabled():
        print(describe(sm, nsm))
    assert sm.update_variables is None, "cap-free (cid=54) — see test_capless_sme0"

    t0 = time.perf_counter()
    Q, Qaux, info = march(model, tmp_path, n_inner_cells=100,
                          domain=SWASHES_DOMAIN, t_end=1.0, cfl=CFL, order=1)
    elapsed = time.perf_counter() - t0

    assert Q.shape[0] == len(sm.state)
    assert np.isfinite(Q).all() and np.isfinite(Qaux).all()
    assert Q[1].min() > 0.0, "wet dam break must stay wet"
    assert np.abs(Q[2]).max() > 0.0, "momentum is zero — the cap bug is back"
    # No new extrema on a flat bed before the wave reaches a boundary.
    assert ETA_R - 1e-12 <= Q[1].min() and Q[1].max() <= ETA_L + 1e-12
    assert info["n_steps"] >= 2 and np.all(info["dt"] > 0.0)

    refs.check("stoker_wet", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("stoker_wet", elapsed, overwrite)


@pytest.mark.small
@pytest.mark.foam
def test_stoker_wet_o2_small(overwrite, tmp_path, capsys):
    """Small twin of the order-2 convergence regression: same model, same
    reconstruction, 20 cells, a couple of steps, full state stored.

    It does NOT measure the convergence order — a rate needs a resolution sweep,
    which is what the regression twin is for.  What it catches is any change in
    the machinery that produces those rates.

    ADAPTATION: the jax design stops after exactly ``n_steps=2``.  zoomyFoam has
    no step-count stop — ``controlDict`` ends on ``endTime`` — so the twin uses a
    short ``t_end`` and ASSERTS the resulting step count.

    That step count must be CFL-SET to mean anything.  The original
    ``t_end=0.5, snapshots=2`` gave a 0.25 s writeInterval against a 2.04 s
    dt_CFL, so dt clamped to the writer, ``n_steps`` was always exactly 2, and
    this twin came back bit-identical across the CFL=0.9 change.  ``t_end=8,
    snapshots=1`` puts dt back under the CFL: measured 5 steps, dt[0]=2.0;
    at CFL 0.45 it is 10 steps, dt[0]=1.0.
    """
    T_END, SNAPS = 8.0, 1
    model = models.swe(dimension=2, bc="swashes", ic=stoker_ic)
    sm, nsm = chain(model)
    with capsys.disabled():
        print(describe(sm, nsm))
    assert sm.update_variables is None

    t0 = time.perf_counter()
    Q, Qaux, info = march(model, tmp_path, n_inner_cells=20,
                          domain=SWASHES_DOMAIN, t_end=T_END, cfl=CFL, order=2,
                          snapshots=SNAPS)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all() and np.isfinite(Qaux).all()
    assert Q[1].min() > 0.0
    assert_cfl_sets_dt(info, t_end=T_END, snapshots=SNAPS, label="stoker_o2_small")

    refs.check("stoker_wet_o2_small", overwrite, Q=Q, Qaux=Qaux, **cfl_witness(info))
    refs.check_time("stoker_wet_o2_small", elapsed, overwrite)
