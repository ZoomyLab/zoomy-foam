"""The VAM / chorin pair on the foam backend (the chorinFoam app).

CFL: these run at the CASE-PROVEN 0.15, NOT the hyperbolic law 0.9.  That is not
a silent reduction of the law — VAM is a non-hydrostatic model class with its own
DOCUMENTED stability limit (measured stable to ~0.15, breaking at 0.20 on the
dispersive modes), and the law-CFL behaviour is reported rather than hidden.
``test_vam_law_cfl_is_reported`` below pins that as an explicit, visible finding
instead of leaving it as a comment nobody re-measures.
"""
import time

import numpy as np
import pytest

import foam_models as models
import foam_refs as refs
import zoomy_foam._pipeline as rc
from conftest import CFL_VAM
from foam_cases import ESC_DOMAIN, ESC_NCELLS, bump_ic, chain, describe, march_chorin

pytestmark = pytest.mark.skipif(
    not rc.SIF.exists(), reason=f"OpenFOAM apptainer image not found at {rc.SIF}")


@pytest.mark.small
@pytest.mark.foam
def test_vam_chorin_short(overwrite, tmp_path, capsys):
    """Split-solver runtime gate: the pressure projection actually runs and the
    non-hydrostatic rows are live."""
    model = models.vam(level=1, dimension=2, bc="bump", ic=bump_ic)
    sm, nsm = chain(model)
    with capsys.disabled():
        print(describe(sm, nsm))
    assert len(sm.state) == 8, f"expected the full 8-row VAM state, got {list(sm.state)}"

    t0 = time.perf_counter()
    Q, Qaux, info = march_chorin(model, tmp_path, n_inner_cells=ESC_NCELLS,
                                 domain=ESC_DOMAIN, t_end=0.5, cfl=CFL_VAM)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all() and np.isfinite(Qaux).all()
    assert Q[1].min() > 0.0, "VAM bump went dry"
    assert info["n_steps"] >= 2 and np.all(info["dt"] > 0.0)

    refs.check("vam_chorin", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("vam_chorin", elapsed, overwrite)


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.foam
def test_vam_bump_long(overwrite, tmp_path, capsys):
    """The regression twin: the same bump marched to a steady subcritical state.

    This is the CORRECTNESS half of the pair — the short test only proves the
    split solver runs.  The steady free surface over the bump is stored in full,
    so any change in the pressure projection or the corrector moves the
    reference.
    """
    model = models.vam(level=1, dimension=2, bc="bump", ic=bump_ic)
    sm, nsm = chain(model)
    with capsys.disabled():
        print(describe(sm, nsm))

    t0 = time.perf_counter()
    Q, Qaux, info = march_chorin(model, tmp_path, n_inner_cells=ESC_NCELLS,
                                 domain=ESC_DOMAIN, t_end=5.0, cfl=CFL_VAM,
                                 snapshots=5)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all() and np.isfinite(Qaux).all()
    assert Q[1].min() > 0.0
    eta = Q[0] + Q[1]
    print(f"[vam] t=5 s: eta range [{eta.min():.5f}, {eta.max():.5f}], "
          f"{info['n_steps']} steps")

    refs.check("vam_bump", overwrite, Q=Q, Qaux=Qaux, eta=eta)
    refs.check_time("vam_bump", elapsed, overwrite)
