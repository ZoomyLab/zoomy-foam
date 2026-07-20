"""The preCICE coupling pair.

STATUS, measured this session:

  * ``swePreciceCoupling`` DID NOT BUILD.  It failed with the same flat-flux-
    indexing error the v6 ``(n_state, 1)`` ABI adoption fixed everywhere else::

        swePreciceCoupling.C:615:35: error: cannot convert
        'Foam::tmp<Foam::Field<double> >' to 'Foam::scalar' in assignment
            qStar = Fc[1] + Fl[1][1];

    ``Fc`` is a ``List<List<scalar>>``, so row 1 needs its scalar dereferenced.
    The nine other sites (numerics.H, numerics_o2.H, precice/PreciceManager.H)
    were adopted earlier; THIS one was missed because ``swePreciceCoupling`` is
    not built by the default ``wmake`` and no test exercised it.  Fixed, and
    ``test_swe_precice_coupling_builds`` below is what keeps it fixed.

  * preCICE itself is present in the image (3.4.1), so the build is genuinely
    exercised rather than skipped for a missing dependency.

  * The full COUPLED PAIR (a two-participant SME<->VOF run) is NOT executed
    here — see the named skip below for exactly why and what it would need.
"""
import subprocess

import pytest

import zoomy_foam._pipeline as rc

pytestmark = pytest.mark.skipif(
    not rc.SIF.exists(), reason=f"OpenFOAM apptainer image not found at {rc.SIF}")

COUPLING_SRC = rc.FOAM_ROOT / "swePreciceCoupling"


@pytest.mark.regression
@pytest.mark.large
@pytest.mark.foam
def test_swe_precice_coupling_builds():
    """The coupling function object must COMPILE against the current flux ABI.

    This is a build test, not a physics test, and it is deliberately so: the
    defect it pins was a compile error that sat undetected precisely because
    nothing built this target.  A physics assertion would not have caught it.
    """
    assert COUPLING_SRC.is_dir(), f"missing {COUPLING_SRC}"
    r = subprocess.run(
        rc._apptainer_cmd(f"cd {COUPLING_SRC}; wclean >/dev/null 2>&1; "
                          f"wmake 2>&1 | tail -40", binds=[rc.FOAM_ROOT]),
        capture_output=True, text=True, timeout=1800)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "Error" not in out, (
        f"swePreciceCoupling failed to build:\n{out[-3000:]}")
    assert "libswePreciceCoupling.so" in out, (
        f"wmake reported success but did not link the library:\n{out[-2000:]}")


@pytest.mark.skip(reason=(
    "NAMED SKIP, with evidence. The coupling LIBRARY builds and is covered by "
    "test_swe_precice_coupling_builds. A coupled PAIR is not run here because it "
    "needs a second participant process (the VOF/interFoam side) driven "
    "concurrently under a shared precice-config.xml, with both participants "
    "advancing in lockstep -- i.e. two mpirun processes and a coupling config, "
    "which is a case harness rather than a unit of this suite. The compare "
    "script for such a run reads its states via read_of_states, which moved from "
    "zoomy_core.postprocessing.column_plots to "
    "library/zoomy_plotting/zoomy_plotting/column_plots.py; foam_cases.read_raw_state "
    "already imports it from the new location, so the reader half is ready when "
    "the pair harness is built."))
@pytest.mark.regression
@pytest.mark.large
@pytest.mark.foam
def test_coupled_pair_physics():
    """Placeholder for the two-participant SME<->VOF coupled run."""
