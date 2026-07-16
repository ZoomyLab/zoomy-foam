"""REQ-155: the packaged foam builder is dimension-agnostic — a genuine 2-D SWE
run end-to-end.

The zoomyFoam SOLVER was always N-D (it reads the full face normal and the NSM
emits flux_x AND flux_y, projecting numerical_flux onto n); only the Python
structured-grid builder was 1-D.  This runs a 2-D circular-dam break on a fully
wet bed through ``run_case`` and asserts the result is a real 2-D field
[b,h,hu,hv] with the expected symmetry — proving the lift, not just that it runs.

Gated on the OpenFOAM apptainer image (``ZOOMY_OF_SIF`` / ~/of_build).
"""
import numpy as np
import pytest

import zoomy_foam._pipeline as rc

DOMAIN = (0.0, 10.0, 0.0, 10.0)
NX = NY = 24
CX, CY, R = 5.0, 5.0, 2.0
H_IN, H_OUT = 2.0, 0.5          # fully WET (avoids the REQ-180 dry-front issue)


def _swe2d_model():
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.model.models import SWE
    from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation

    class _SWE2D(SWE):
        def _initialize_derived_properties(self):
            # BC tags MUST match the builder's 2-D face names (West/East/South/North).
            self.boundary_conditions = BoundaryConditions(
                [Extrapolation(tag=t) for t in ("West", "East", "South", "North")])
            super()._initialize_derived_properties()

    def _ic(x):
        xc, yc = np.asarray(x)[0], np.asarray(x)[1]
        r = np.hypot(xc - CX, yc - CY)
        h = np.where(r < R, H_IN, H_OUT)
        z = np.zeros_like(h)
        return np.stack([z, h, z, z])          # [b, h, hu, hv]

    m = _SWE2D(dimension=2)
    m.initial_conditions = IC.UserFunction(function=_ic)
    m.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    return m


@pytest.mark.skipif(not rc.SIF.exists(), reason="OpenFOAM apptainer image not available")
def test_swe2d_dam_break_end_to_end(tmp_path):
    import h5py
    from zoomy_core.mesh.base_mesh import BaseMesh

    model = _swe2d_model()
    assert [str(s) for s in rc._codegen(model).state] == ["b", "h", "hu", "hv"]

    mp = tmp_path / "mesh2d.h5"
    BaseMesh.create_2d(domain=DOMAIN, nx=NX, ny=NY).write_to_hdf5(str(mp))
    prog = []
    h5 = rc.run_case(
        model,
        {"mesh": str(mp), "time_end": 0.3, "output_snapshots": 3,
         "cfl": 0.45, "reconstruction_order": 1},
        tmp_path, on_progress=lambda i, t, dt: prog.append((i, t, dt)))

    assert prog and prog[-1][1] == pytest.approx(0.3, abs=1e-6)
    with h5py.File(h5, "r") as f:
        its = sorted(f["fields"], key=lambda s: int(s.split("_")[1]))
        Q = f["fields"][its[-1]]["Q"][:]
    assert Q.shape[0] == 4                                   # [b, h, hu, hv]
    h = Q[1]
    assert np.isfinite(h).all()
    assert h.min() > 0.0                                    # wet everywhere, no h<0
    # genuinely 2-D: the field must vary in BOTH directions (not a 1-D extrusion)
    hg = h.reshape(NY, NX)                                  # OF order: y outer, x inner
    assert hg.std(axis=0).max() > 1e-3                      # varies along y
    assert hg.std(axis=1).max() > 1e-3                      # varies along x
    # radial symmetry of a circular dam: opposite quadrants match (bed flat, IC radial)
    assert np.allclose(hg, hg[::-1, :], atol=0.05)          # up/down mirror
    assert np.allclose(hg, hg[:, ::-1], atol=0.05)          # left/right mirror


def _run_h(tmp_path, nprocs):
    import h5py
    from zoomy_core.mesh.base_mesh import BaseMesh
    mp = tmp_path / f"mesh_{nprocs}.h5"
    BaseMesh.create_2d(domain=DOMAIN, nx=NX, ny=NY).write_to_hdf5(str(mp))
    out = tmp_path / f"np{nprocs}"
    out.mkdir()
    h5 = rc.run_case(
        _swe2d_model(),
        {"mesh": str(mp), "time_end": 0.15, "output_snapshots": 1,
         "cfl": 0.45, "reconstruction_order": 1, "nprocs": nprocs},
        out)
    with h5py.File(h5, "r") as f:
        its = sorted(f["fields"], key=lambda s: int(s.split("_")[1]))
        return f["fields"][its[-1]]["Q"][:]


@pytest.mark.skipif(not rc.SIF.exists(), reason="OpenFOAM apptainer image not available")
def test_swe2d_mpi_matches_serial(tmp_path):
    """The solver is rank-agnostic — a 2-processor decomposed run (scotch +
    mpirun -parallel + reconstructPar, driven by the `nprocs` knob) must
    reproduce the serial result BIT-for-BIT at order 1 (global dt via
    returnReduce, processor-face fluxes via patchNeighbourField)."""
    Q1 = _run_h(tmp_path, 1)
    Q2 = _run_h(tmp_path, 2)
    assert Q1.shape == Q2.shape
    assert np.max(np.abs(Q1 - Q2)) < 1e-10, (
        f"serial vs 2-proc differ by {np.max(np.abs(Q1 - Q2)):.2e}")
