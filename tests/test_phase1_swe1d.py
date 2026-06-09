"""Phase 1 smoke test — Foam printer on SWE 1D.

Builds the smallest possible SystemModel (1D shallow water, h + hu)
and runs :class:`FoamSystemModelPrinter` to confirm that:

* the operator kernels (`flux_x`, `nonconservative_matrix_x`,
  `quasilinear_matrix_x`, `eigenvalues`, `source`) get emitted as
  syntactically-plausible Foam C++,
* the Q/Qaux/n/p symbol bindings resolve correctly,
* the per-direction split (`*_x`, `*_y`, `*_z`) honours the SystemModel's
  spatial dimension.

No OpenFOAM build is invoked; output is printed to stdout for inspection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sympy as sp
from sympy import Matrix

from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.misc.misc import ZArray
from zoomy_core.model import boundary_conditions as BC
from zoomy_core.model import initial_conditions as IC
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.transformation.to_openfoam import FoamSystemModelPrinter


class SWE1D(StructuredDerivativeModel):
    """Conservative 1D shallow water on a flat bed."""

    dimension = 1
    variables = ["h", "hu"]
    parameters = {"g": (9.81, "positive")}

    def flux(self):
        h = self.Q.h
        hu = self.Q.hu
        g = self.params.g
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[0, 0] = hu
        F[1, 0] = hu * hu / h + 0.5 * g * h * h
        return ZArray(F)

    def source(self):
        return ZArray.zeros(self.n_variables)


def main():
    bcs = BC.BoundaryConditions(
        [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]
    )

    def ic(x):
        Q = np.zeros(2, dtype=float)
        Q[0] = 0.5 if x[0] < 5.0 else 0.1
        Q[1] = 0.0
        return Q

    model = SWE1D(
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(function=ic),
    )
    sm = SystemModel.from_model(model)

    print("=" * 70)
    print(f"SystemModel: n_eq={sm.n_equations}  "
          f"n_state={len(sm.state)}  "
          f"n_aux={len(sm.aux_state)}  "
          f"dimension={sm.dimension}")
    print(f"state:       {list(sm.state)}")
    print(f"aux_state:   {list(sm.aux_state)}")
    print(f"parameters:  {list(sm.parameters.values())}")
    print(f"normal:      {list(sm.normal.values())}")
    print("=" * 70)

    code = FoamSystemModelPrinter(sm).create_code()
    print(code)


if __name__ == "__main__":
    sys.exit(main())
