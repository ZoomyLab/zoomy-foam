"""REQ-175 — a-posteriori local MOOD positivity in the order-2 explicit path.

MOOD must NEVER corrupt a healthy run: where no cell goes h<0, the order-2
candidate is accepted verbatim and the result is BYTE-IDENTICAL to
``positivity none``.  That no-op safety is the property this guards (a MOOD
that silently perturbs healthy flow is worse than none).  The override itself
is conservative and never truncates h — it re-does a troubled cell with a
1st-order forward-Euler step from Q^n, so mass is preserved exactly.

The wet/dry dt behaviour (velocity ~ hu*hinv at a thinning front) is governed
by the NumericalSystemModel's desingularized ``hinv`` aux, NOT by MOOD and NOT
by any h truncation — so it is out of scope here.

Gated on the OpenFOAM apptainer image (``ZOOMY_OF_SIF`` / ~/of_build).
"""
import os
import re

import numpy as np
import pytest

import zoomy_foam._pipeline as rc

X0, X1, N = 0.0, 10.0, 64
DAM_C, DAM_R, H_IN, H_OUT = 5.0, 2.0, 1.0, 0.5   # fully WET dam (no dry cell)


def _wet_dam_model():
    """SWE with a fully-wet circular-dam IC — deep inside, shallower outside,
    everywhere wet, so the order-2 candidate never goes h<0 and MOOD must be a
    strict no-op."""
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.model.models import SWE
    from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation

    class _SWE1D(SWE):
        def _initialize_derived_properties(self):
            self.boundary_conditions = BoundaryConditions(
                [Extrapolation(tag="left"), Extrapolation(tag="right")])
            super()._initialize_derived_properties()

    def _ic(x):
        xc = np.asarray(x)[0]
        h = np.where(np.abs(xc - DAM_C) < DAM_R, H_IN, H_OUT)
        return np.stack([np.zeros_like(h), h, np.zeros_like(h)])

    m = _SWE1D()
    m.initial_conditions = IC.UserFunction(function=_ic)
    m.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    return m


def _final_h(tmp_path, positivity):
    """Run the wet dam at order 2 with the given positivity; return final h + the
    number of MOOD interventions logged."""
    import h5py
    from zoomy_core.mesh.base_mesh import BaseMesh

    mp = tmp_path / f"mesh_{positivity}.h5"
    BaseMesh.create_1d(domain=(X0, X1), n_inner_cells=N).write_to_hdf5(str(mp))
    out = tmp_path / positivity
    out.mkdir()
    h5 = rc.run_case(
        _wet_dam_model(),
        {"mesh": str(mp), "time_end": 0.2, "output_snapshots": 4,
         "cfl": 0.45, "reconstruction_order": 2, "positivity": positivity},
        out,
    )
    with h5py.File(h5, "r") as f:
        its = sorted(f["fields"], key=lambda s: int(s.split("_")[1]))
        h = f["fields"][its[-1]]["Q"][:][1]      # row 1 = depth
    log = (out / "run.log").read_text() if (out / "run.log").exists() else ""
    moods = len(re.findall(r"\[MOOD\] troubled", log))
    return np.asarray(h), moods


@pytest.mark.skipif(not rc.SIF.exists(), reason="OpenFOAM apptainer image not available")
def test_mood_is_a_noop_on_healthy_flow(tmp_path):
    """On a fully-wet dam the order-2 candidate never goes h<0, so `positivity
    mood` must be byte-identical to `positivity none` (and fire zero times)."""
    h_none, m_none = _final_h(tmp_path, "none")
    h_mood, m_mood = _final_h(tmp_path, "mood")
    assert m_none == 0
    assert m_mood == 0, "MOOD fired on a healthy wet flow — detection is too eager"
    assert np.array_equal(h_none, h_mood), (
        f"MOOD perturbed a healthy run (max|Δ|={np.max(np.abs(h_none - h_mood)):.2e})")
    assert h_none.min() > 0.0
