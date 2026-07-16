"""REQ-93: zoomy_foam.run_case — the importable in-process run entry.

Two tiers: (1) always-on unit checks of the pure-Python case-build logic
(codegen + blockMesh geometry + 0/Qi from the model IC + controlDict), which
need no container; (2) an end-to-end run gated on the OpenFOAM apptainer image
(``ZOOMY_OF_SIF`` / ``~/of_build/zoomy_openfoam.sif``).
"""
import os
import re

import numpy as np
import pytest

import zoomy_foam._pipeline as rc


# A lake-at-rest IC whose three state rows are all DISTINCT, so the HDF5
# round-trip can assert row IDENTITY, not just row count. bed=0.3 is a CONSTANT
# bed (db/dx=0), so g*h*db/dx=0 and this stays exactly at rest at every order —
# but 0.3 / 1.5 / 0.0 fingerprint b / h / hu uniquely.  This is the foam guard
# for REQ-158 (vtk_to_hdf5 maps fields positionally; a prepended synthetic field
# like foamToVTK's cellID would shift every row by one, and "ask for h, get b"
# passes a mass check silently).  See _keep_state_rows in _pipeline.py.
_IC_BED, _IC_DEPTH, _IC_HU = 0.3, 1.5, 0.0


def _swe_model():
    """A minimal SWE model instance with a constant-bed lake-at-rest IC.

    Enough to exercise codegen + case build without asserting physics (the
    dam-break physics is covered by the SIF-gated end-to-end test)."""
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.model.models import SWE
    from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation

    class _SWE1D(SWE):
        # set model-level BCs before the kernels build, so SystemModel.from_model
        # picks them up (the plain `boundary_conditions=` kwarg is stashed as
        # _coupling_bcs and bypassed on the from_model path — case_swe_1d gotcha).
        def _initialize_derived_properties(self):
            self.boundary_conditions = BoundaryConditions(
                [Extrapolation(tag="left"), Extrapolation(tag="right")])
            super()._initialize_derived_properties()

    m = _SWE1D()
    m.initial_conditions = IC.Constant(
        constants=lambda n: np.array([_IC_BED, _IC_DEPTH, _IC_HU]))
    m.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    return m


def test_import():
    from zoomy_foam import run_case
    assert callable(run_case)


def test_case_build_pure_python(tmp_path):
    """codegen + blockMesh geometry + 0/Qi + controlDict — no container."""
    from zoomy_core.mesh.base_mesh import BaseMesh
    from zoomy_core.mesh.lsq_mesh import LSQMesh

    model = _swe_model()
    sm = rc._codegen(model)
    assert [str(s) for s in sm.state] == ["b", "h", "hu"]
    assert (rc.FOAM_ROOT / "Model.H").exists() and (rc.FOAM_ROOT / "NumericsKernels.H").exists()

    mp = tmp_path / "mesh.h5"
    BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=40).write_to_hdf5(str(mp))
    mesh = LSQMesh.from_hdf5(str(mp))
    lo, hi, n, _order, dim = rc._mesh_geometry(mesh)
    assert dim == 1
    assert (round(lo[0]), round(hi[0]), n[0]) == (0, 10, 40)

    case = tmp_path / "foam_case"
    rc._build_case(case, mesh, model, sm,
                   {"time_end": 0.8, "output_snapshots": 20, "cfl": 0.45,
                    "reconstruction_order": 2}, "DUMMYBIN")
    cd = (case / "system" / "controlDict").read_text()
    assert "application zoomyFoam;" in cd and "endTime 0.8" in cd
    assert "reconstructionOrder 2" in cd and "maxCo 0.45" in cd and "g 9.81" in cd
    # one 0/Qi per state variable, each nc long, written in state order:
    # Q0=bed, Q1=depth, Q2=hu — the same order the HDF5 round-trip must preserve.
    assert sorted(os.listdir(case / "0")) == ["Q0", "Q1", "Q2"]

    def _field(name):
        return [float(v) for v in re.search(r"\(\s*(.*?)\s*\)",
                (case / "0" / name).read_text(), re.S).group(1).split()]

    assert all(abs(v - _IC_BED) < 1e-9 for v in _field("Q0"))
    h = _field("Q1")
    assert len(h) == 40 and all(abs(v - _IC_DEPTH) < 1e-9 for v in h)


@pytest.mark.skipif(not rc.SIF.exists(), reason="OpenFOAM apptainer image not available")
def test_run_case_end_to_end(tmp_path):
    """Full pipeline: import -> run_case -> HDF5 with clean [b,h,hu] frames."""
    from zoomy_foam import run_case
    import h5py
    from zoomy_core.mesh.base_mesh import BaseMesh

    model = _swe_model()
    mp = tmp_path / "mesh.h5"
    BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=80).write_to_hdf5(str(mp))
    prog = []
    h5 = run_case(model,
                  {"mesh": str(mp), "time_end": 0.2, "output_snapshots": 5,
                   "cfl": 0.45, "reconstruction_order": 1},
                  tmp_path, on_progress=lambda i, t, dt: prog.append((i, t, dt)))
    assert prog and prog[-1][1] == pytest.approx(0.2, abs=1e-6)
    with h5py.File(h5, "r") as f:
        its = sorted(f["fields"], key=lambda s: int(s.split("_")[1]))
        Q = f["fields"][its[-1]]["Q"][:]
        assert Q.shape[0] == 3            # [b, h, hu] only — no cellID/diagnostics
        # REQ-158 guard: rows must be [b, h, hu] BY IDENTITY, not just by count.
        # A positional shift (foamToVTK's cellID leaking as row 0) would put the
        # bed where h belongs; lake-at-rest keeps every row at its distinct IC.
        assert np.allclose(Q[0], _IC_BED,   atol=1e-6), "row 0 is not the bed"
        assert np.allclose(Q[1], _IC_DEPTH, atol=1e-6), "row 1 is not the depth"
        assert np.allclose(Q[2], _IC_HU,    atol=1e-6), "row 2 is not hu"
