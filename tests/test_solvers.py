"""REQ-133: zoomy_foam.solvers — param.Parameterized GUI solver wrappers.

Always-on: import + bounded params + structured-settings translation (writes the
1-D mesh.h5, maps settings.output) + SplitSolver guard.  SIF-gated: the full
HyperbolicSolver.solve -> VTK series.
"""
import os

import numpy as np
import pytest

import zoomy_foam._pipeline as rc
from zoomy_foam import solvers


def _swe_model():
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.model.models import SWE
    from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation

    class _SWE1D(SWE):
        def _initialize_derived_properties(self):
            self.boundary_conditions = BoundaryConditions(
                [Extrapolation(tag="left"), Extrapolation(tag="right")])
            super()._initialize_derived_properties()

    m = _SWE1D()
    m.initial_conditions = IC.RP(
        low=lambda n: np.array([0.0, 1.0, 0.0]),
        high=lambda n: np.array([0.0, 2.0, 0.0]), jump_position_x=5.0)
    m.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    return m


def test_import_and_params():
    hs = solvers.HyperbolicSolver(CFL=0.3, order=2)
    assert hs.CFL == 0.3 and hs.order == 2
    # bounded params (GUI widget generation) enforce ranges
    with pytest.raises(ValueError):
        solvers.HyperbolicSolver(order=3)          # bounds (1, 2)
    assert isinstance(solvers.SplitSolver().cfl, float)


def test_settings_translation(tmp_path):
    hs = solvers.HyperbolicSolver()
    out = str(tmp_path / "out")
    settings = {"output": {"directory": out, "snapshots": 7}, "time_end": 1.5}
    s = hs._foam_settings({"domain": [0.0, 10.0], "n_cells": [64]}, settings, hs._output_dir(settings))
    assert s["time_end"] == 1.5 and s["output_snapshots"] == 7
    assert os.path.exists(s["mesh"])               # the 1-D mesh.h5 was written
    from zoomy_core.mesh.lsq_mesh import LSQMesh
    assert LSQMesh.from_hdf5(s["mesh"]).n_inner_cells == 64


def test_split_guard_non_split_model():
    with pytest.raises(TypeError):                 # SWE has no chorin_split
        solvers.SplitSolver().solve(_swe_model(), {"domain": [0, 10], "n_cells": [8]},
                                    {"output": {"directory": "/tmp/x"}})


def test_2d_mesh_not_implemented():
    hs = solvers.HyperbolicSolver()
    with pytest.raises(NotImplementedError):
        hs._foam_settings({"domain": [0, 1, 0, 1], "n_cells": [4, 4]},
                          {"output": {"directory": "/tmp/x"}}, "/tmp/x")


@pytest.mark.skipif(not rc.SIF.exists(), reason="OpenFOAM apptainer image not available")
def test_hyperbolic_end_to_end(tmp_path):
    import meshio
    out = str(tmp_path / "out")
    settings = {"output": {"directory": out, "filename": "sim", "snapshots": 5},
                "time_end": 0.2}
    prog = []
    pvd = solvers.HyperbolicSolver(CFL=0.45, order=1).solve(
        _swe_model(), {"domain": [0.0, 10.0], "n_cells": [80]}, settings,
        on_progress=lambda i, t, dt: prog.append(t))
    assert os.path.exists(pvd) and pvd.endswith(".pvd")
    assert prog and prog[-1] == pytest.approx(0.2, abs=1e-6)
    frames = [f for f in os.listdir(os.path.dirname(pvd)) if f.endswith(".vtk")]
    assert frames
    m = meshio.read(os.path.join(os.path.dirname(pvd), sorted(frames)[0]))
    assert "Q1" in m.cell_data                     # state field h present, no HDF5 step


def test_split_solver_wired():
    """REQ-133 follow-up: SplitSolver drives the chorinFoam pipeline (no more
    NotImplementedError). Params are bounded; non-split models are rejected."""
    ss = solvers.SplitSolver(cfl=0.25, pressure_tol=1e-9)
    assert ss.cfl == 0.25 and ss.pressure_maxit == 2000
    from zoomy_foam._pipeline import run_chorin_to_vtk, _codegen_chorin  # noqa: F401
    import inspect
    src = inspect.getsource(solvers.SplitSolver.solve)
    assert "run_chorin_to_vtk" in src and "NotImplementedError" not in src
