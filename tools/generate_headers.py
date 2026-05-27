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

import argparse

from zoomy_core.fvm.riemann_solvers import Rusanov
from zoomy_core.transformation.to_openfoam import (
    FoamNumericsPrinter,
    FoamSystemModelPrinter,
    FoamUpdateAuxPrinter,
)


# Map a case-name CLI flag to the factory module under tests/cases/.
_CASE_FACTORIES = {
    "swe_bed_friction_1d": "cases.swe_bed_friction_1d:make_test_case",
    "swe_bed_as_aux_1d":   "cases.swe_bed_as_aux_1d:make_test_case",
}


def _load_factory(spec):
    import importlib
    mod_name, fn_name = spec.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case", default="swe_bed_friction_1d", choices=_CASE_FACTORIES,
    )
    args = ap.parse_args()
    sm = _load_factory(_CASE_FACTORIES[args.case])()
    numerics = Rusanov(model=sm)

    paths = {
        "Model.H":             FoamSystemModelPrinter,
        "NumericsKernels.H":   FoamNumericsPrinter,
        "UpdateAuxVariables.H": FoamUpdateAuxPrinter,
    }
    targets = {
        "Model.H":              (sm, {"analytical_eigenvalues": True}),
        "NumericsKernels.H":    (numerics, {}),
        "UpdateAuxVariables.H": (sm, {}),
    }

    for fname, cls in paths.items():
        out = ROOT / fname
        payload, opts = targets[fname]
        cls.write_code(payload, out, **opts)
        print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
