"""One genuinely 2-D case at the 2-D law CFL 0.45.

Everything else in this suite is 1-D (SME(level=0, dimension=2) = one horizontal
direction).  This is the only test that exercises the second horizontal momentum
row, the 2-D flux assembly and a mesh whose faces are not all axis-aligned in a
single direction — and the only one that runs at CFL.
"""
import time

import numpy as np
import pytest

import foam_models as models
import foam_refs as refs
import zoomy_foam._pipeline as rc
from conftest import CFL
from foam_cases import chain, describe, gaussian_pulse_2d, march

pytestmark = pytest.mark.skipif(
    not rc.SIF.exists(), reason=f"OpenFOAM apptainer image not found at {rc.SIF}")


@pytest.mark.small
@pytest.mark.foam
def test_swe_2d_pulse(overwrite, tmp_path, capsys):
    # dimension=3 is the derivation's 2-D: state [b, h, q_0, q_1].
    model = models.swe(dimension=3, bc="extrapolation", ic=gaussian_pulse_2d)
    sm, nsm = chain(model)
    with capsys.disabled():
        print(describe(sm, nsm))
    assert sm.update_variables is None
    assert len(sm.state) == 4, f"expected [b,h,q_0,q_1], got {list(sm.state)}"

    t0 = time.perf_counter()
    # BaseMesh.create_2d takes a FLAT (x_min, x_max, y_min, y_max) domain.
    Q, Qaux, info = march(model, tmp_path, n_inner_cells=(24, 24),
                          domain=(-1.0, 1.0, -1.0, 1.0), t_end=0.1,
                          cfl=CFL, order=1, dimension=2)
    elapsed = time.perf_counter() - t0

    assert Q.shape[0] == 4 and Q.shape[1] == 24 * 24
    assert np.isfinite(Q).all() and np.isfinite(Qaux).all()
    assert Q[1].min() > 0.0, "basin went dry"
    # The pulse is radial, so BOTH momentum components must be excited: a 2-D
    # run that only ever moves q_0 would pass every 1-D assertion in this suite.
    assert np.abs(Q[2]).max() > 0.0, "q_0 never moved"
    assert np.abs(Q[3]).max() > 0.0, "q_1 never moved — 2-D flux not exercised"
    assert info["n_steps"] >= 2 and np.all(info["dt"] > 0.0)

    refs.check("swe_2d", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("swe_2d", elapsed, overwrite)
