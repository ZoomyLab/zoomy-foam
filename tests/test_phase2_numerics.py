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


def main():
    sm = make_test_case()
    numerics = Rusanov(model=sm)
    code = FoamNumericsPrinter(numerics).create_code()
    print(code)

    for kernel in ("numerical_flux", "numerical_fluctuations",
                   "local_max_abs_eigenvalue"):
        assert f" {kernel}(" in code, f"missing kernel {kernel}"

    # Face-state symbols and parameter list must use Foam types.
    assert "const Foam::List<Foam::scalar>& Q_minus" in code
    assert "const Foam::List<Foam::scalar>& Q_plus" in code
    assert "const Foam::List<Foam::scalar>& p" in code
    assert "const Foam::vector& n" in code

    # max_wavespeed must be left opaque — the solver provides the
    # implementation in numerics.H (cf. numpy's "max_wavespeed: None").
    assert "numerics::max_wavespeed(" in code
    # And NOT the inherited nested-max-abs expansion of GenericCppBase.
    assert "Foam::max(Foam::abs(" not in code

    print("\nOK: FoamNumericsPrinter emits Rusanov kernels for SWE+bed+Manning.")


if __name__ == "__main__":
    sys.exit(main())
