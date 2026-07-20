"""Regression: a DERIVED, CAPLESS SME(level=0) must compile AND march in foam.

This pins two defects that together made foam unable to run any model built
under the standing user laws:

  BLOCKER 1 — zoomyFoam.C calls ``Model::update_variables`` unconditionally,
  but the printer used to emit that symbol only when the model HAD one.  With
  the wet/dry cap OFF by default (cid=54) a derived SME(level=0) has
  ``update_variables = None``, so the generated Model.H contained zero
  occurrences of it and the driver did not compile.  Core now always emits the
  kernel (identity when the slot is None); the ``update_variables is None``
  assertion below is what makes this test actually exercise that path -- if a
  future default re-enables the cap, the test must fail loudly rather than
  quietly stop testing the thing it was written for.

  BLOCKER 2 — ``numerical_flux`` used to carry an extra trailing row (the
  Rusanov max wave speed, REQ-212).  The v6 solver design moved eigenvalues to
  a dedicated dt pass, so the kernel is back to ``(n_state, 1)`` and foam's
  hand-written headers index a row's scalar as ``F[i][0]``.

Physics is asserted (finite, h >= 0, dt > 0, correct row count), never exit
status.  CFL is the 1-D law value 0.9 and is NOT reduced: if this ever goes
unstable that is a finding to report, not a knob to turn down.

Gated on the OpenFOAM apptainer image (``ZOOMY_OF_SIF`` / ~/of_build).
"""
import re

import numpy as np
import pytest

import zoomy_foam._pipeline as rc

X0, X1, N = 0.0, 10.0, 100
DAM_X, H_L, H_R = 5.0, 1.0, 0.5      # SWASHES stoker: WET dam break (h_R > 0)
CFL = 0.9                             # 1-D law. Never silently reduced.
T_END = 0.2

pytestmark = pytest.mark.skipif(
    not rc.SIF.exists(), reason=f"OpenFOAM apptainer image not found at {rc.SIF}")


def _model():
    """SWE as the DERIVATION produces it -- SME(level=0, dimension=2), i.e. one
    horizontal direction.  Never a hand-built SWE class."""
    from zoomy_core.model.models import SME
    from zoomy_core.model import boundary_conditions as BC
    from zoomy_core.model import initial_conditions as IC

    def ic(x):
        xc = np.asarray(x)[0]
        h = np.where(xc < DAM_X, H_L, H_R)
        return np.stack([np.zeros_like(h), h, np.zeros_like(h)])  # flat bed

    return SME(
        level=0, dimension=2,
        boundary_conditions=BC.BoundaryConditions(
            [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]),
        initial_conditions=IC.UserFunction(function=ic),
        aux_initial_conditions=IC.Constant(constants=lambda n: np.zeros(n)),
    )


def test_capless_sme0_stoker_wet_marches(tmp_path, capsys):
    import h5py
    from zoomy_core.systemmodel import SystemModel
    from zoomy_core.mesh.base_mesh import BaseMesh
    from zoomy_foam import run_case

    model = _model()
    sm = SystemModel.from_model(model)
    n_state = len(sm.state)

    # The cap must be OFF -- otherwise this test silently stops covering Blocker 1.
    assert sm.update_variables is None, (
        "wet/dry cap must be OFF by default (cid=54); with a cap present this "
        "test no longer exercises the unconditional-emit path it exists to pin")

    # NSM operator matrices BEFORE the first march (user law).
    with capsys.disabled():
        print(f"\nstate={list(sm.state)} aux={list(sm.aux_state)} dim={sm.dimension}")
        print("flux =", sm.flux)
        print("nonconservative_matrix =", sm.nonconservative_matrix)
        print("quasilinear_matrix =", sm.quasilinear_matrix)
        print("eigenvalues =", sm.eigenvalues)
        print("source =", sm.source)

    mp = tmp_path / "mesh.h5"
    BaseMesh.create_1d(domain=(X0, X1), n_inner_cells=N).write_to_hdf5(str(mp))
    h5 = run_case(model,
                  {"mesh": str(mp), "time_end": T_END, "output_snapshots": 5,
                   "cfl": CFL, "reconstruction_order": 1},
                  str(tmp_path / "run"))

    with h5py.File(h5, "r") as f:
        grp = f["fields"]
        keys = sorted(grp, key=lambda k: int(k.split("_")[1]))
        Q = np.stack([np.asarray(grp[k]["Q"]) for k in keys])

    # The stale-binary failure mode: a binary built for a different model reads
    # Q0..Qn-1 happily and the export layer drops the surplus, so the ROW COUNT
    # is the thing that catches it.
    assert Q.shape[1] == n_state, f"{Q.shape[1]} rows, model has {n_state}"
    assert np.all(np.isfinite(Q)), "non-finite state"

    h = Q[:, 1, :]
    assert np.all(h >= 0.0), f"negative depth: min h = {h.min():.3e}"
    # Wet dam break: h stays inside the initial bracket (no new extrema on a
    # flat bed before any wave reaches a boundary) and mass is conserved.
    assert H_R - 1e-9 <= h.min() and h.max() <= H_L + 1e-9, "h left [h_R, h_L]"
    m0, m1 = h[0].sum(), h[-1].sum()
    assert abs(m1 - m0) / m0 < 1e-6, f"mass drift {abs(m1-m0)/m0:.3e}"

    # dt must come from the SOLVER LOG, not the frame cadence -- consecutive
    # frames differ by the fixed writeInterval, so asserting on those would pass
    # even for a solver taking garbage steps.
    log = (tmp_path / "run" / "foam_case" / "run.log").read_text()
    solver_times = [float(m.group(1))
                    for m in re.finditer(r"^Time = ([-\d.eE+]+)s", log, re.M)]
    assert len(solver_times) >= 2, f"only {len(solver_times)} solver steps"
    dts = np.diff(np.array([0.0] + solver_times))
    assert np.all(dts > 0.0) and np.all(np.isfinite(dts)), "bad solver dt"

    # The build fingerprint the stale-binary guard keys on must be present and
    # agree -- run_case already raises on mismatch, this pins the banner itself.
    m = re.search(r"^zoomy: n_dof_q = (\d+)", log, re.M)
    assert m and int(m.group(1)) == n_state, "build-fingerprint banner missing/wrong"


@pytest.mark.parametrize("reported,expected", [(4, 3), (2, 3)])
def test_stale_binary_guard_raises_on_dof_mismatch(reported, expected, tmp_path):
    """A binary compiled for a different state size must HARD-FAIL the run.

    Verified end-to-end during development by copying a 3-state binary into the
    4-state cache slot: without this guard the run completes and exports a
    well-formed but physically wrong result, because the binary reads Q0..Q2,
    ignores Q3 and writes back its own field count."""
    with pytest.raises(RuntimeError, match="STALE BINARY"):
        rc._assert_binary_matches_model("/fake/bin", reported, expected, tmp_path)


def test_stale_binary_guard_raises_when_fingerprint_absent(tmp_path):
    """A cached binary predating the fingerprint banner cannot be verified, so
    it must be rejected rather than trusted."""
    with pytest.raises(RuntimeError, match="did not print"):
        rc._assert_binary_matches_model("/fake/bin", None, 3, tmp_path)


def test_stale_binary_guard_passes_on_match(tmp_path):
    rc._assert_binary_matches_model("/fake/bin", 3, 3, tmp_path)
