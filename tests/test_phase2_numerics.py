"""Phase 2 — smoke-test the Foam Numerics printer.

Builds Rusanov over the SWE+bed+Manning test case, runs
:class:`FoamNumericsPrinter`, prints output, asserts that the three
expected kernels (``numerical_flux``, ``numerical_fluctuations``,
``local_max_abs_eigenvalue``) are emitted with Foam-typed signatures.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cases.swe_bed_friction_1d import make_test_case
from zoomy_core.fvm.riemann_solvers import Rusanov
from zoomy_core.transformation.to_openfoam import FoamNumericsPrinter


def test_foam_numerics_printer_emits_rusanov_kernels():
    sm = make_test_case()
    numerics = Rusanov(model=sm)
    code = FoamNumericsPrinter(numerics).create_code()

    for kernel in ("numerical_flux", "numerical_fluctuations",
                   "local_max_abs_eigenvalue"):
        assert f" {kernel}(" in code, f"missing kernel {kernel}"

    # Face-state symbols and parameter list must use Foam types.
    assert "const Foam::List<Foam::scalar>& Q_minus" in code
    assert "const Foam::List<Foam::scalar>& Q_plus" in code
    assert "const Foam::List<Foam::scalar>& p" in code
    assert "const Foam::vector& n" in code

    # The wave speed is the GENERATED kernel, not an opaque UserFunctions call:
    # core dropped max_wavespeed (4e3e1f9) for local_max_abs_eigenvalue.
    assert "numerics::max_wavespeed(" not in code
    # And NOT the inherited nested-max-abs expansion of GenericCppBase.
    assert "Foam::max(Foam::abs(" not in code


def test_swe_wave_speed_is_the_exact_analytic_spectrum():
    """SWE HAS a closed-form spectrum, so the printer must emit it — not the
    Gershgorin row-sum fallback that models without one get (REQ-167 GAP 1).

    Signature of the exact bound: sqrt(g*h**5)/h**2 == sqrt(g*h), i.e. a
    ``pow(..., 1.0/2.0)``.  The row-sum has no sqrt at all and would instead
    evaluate to max(1, 2*g*h) at rest — dimensionally m^2/s^2, ~6x too large
    at h=1 m.  If this ever regresses to a row-sum, dt silently drops ~6x.
    """
    code = FoamNumericsPrinter(Rusanov(model=make_test_case())).create_code()
    body = code.split("local_max_abs_eigenvalue", 1)[1].split("\n}", 1)[0]
    assert "1.0/2.0" in body, "SWE wave speed lost its sqrt -> row-sum fallback?"
