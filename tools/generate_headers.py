"""Generate Model.H and Numerics.H for a chosen test case.

Run from anywhere with the zoomy conda env active.  Writes into the
zoomy_foam solver source tree so ``wmake`` can pick them up.

Usage:
    python tools/generate_headers.py             # SWE+bed+Manning + Rusanov
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the tests/cases/ package importable for case factories.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from cases.swe_bed_friction_1d import make_test_case
from zoomy_core.fvm.riemann_solvers import Rusanov
from zoomy_core.transformation.to_openfoam import (
    FoamNumericsPrinter,
    FoamSystemModelPrinter,
)


def main():
    sm = make_test_case()
    numerics = Rusanov(model=sm)

    model_h = ROOT / "Model.H"
    numerics_kernels_h = ROOT / "NumericsKernels.H"

    FoamSystemModelPrinter.write_code(
        sm, model_h, analytical_eigenvalues=True
    )
    FoamNumericsPrinter.write_code(numerics, numerics_kernels_h)

    print(f"wrote {model_h}")
    print(f"wrote {numerics_kernels_h}")


if __name__ == "__main__":
    sys.exit(main())
