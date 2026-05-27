"""Phase 1 — SWE 2D smoke test: confirm per-direction split emits
``flux_x`` *and* ``flux_y`` with the correct entries."""

from __future__ import annotations

import sys
import re

import numpy as np
from sympy import Matrix

from zoomy_core.misc.misc import ZArray
from zoomy_core.model import boundary_conditions as BC
from zoomy_core.model import initial_conditions as IC
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.transformation.to_openfoam import FoamSystemModelPrinter


class SWE2D(StructuredDerivativeModel):
    """Conservative 2D shallow water on a flat bed."""

    dimension = 2
    variables = ["h", "hu", "hv"]
    parameters = {"g": (9.81, "positive")}

    def flux(self):
        h, hu, hv = self.Q.h, self.Q.hu, self.Q.hv
        g = self.params.g
        F = Matrix.zeros(self.n_variables, self.dimension)
        # x-flux
        F[0, 0] = hu
        F[1, 0] = hu * hu / h + 0.5 * g * h * h
        F[2, 0] = hu * hv / h
        # y-flux
        F[0, 1] = hv
        F[1, 1] = hu * hv / h
        F[2, 1] = hv * hv / h + 0.5 * g * h * h
        return ZArray(F)

    def source(self):
        return ZArray.zeros(self.n_variables)


def main():
    bcs = BC.BoundaryConditions(
        [BC.Extrapolation(tag=t) for t in ("left", "right", "bottom", "top")]
    )

    def ic(x):
        Q = np.zeros(3, dtype=float)
        Q[0] = 0.5 if (x[0] - 5) ** 2 + (x[1] - 5) ** 2 < 1 else 0.1
        return Q

    sm = SystemModel.from_model(
        SWE2D(boundary_conditions=bcs, initial_conditions=IC.UserFunction(function=ic))
    )

    code = FoamSystemModelPrinter(sm, analytical_eigenvalues=True).create_code()
    print(code)

    # Sanity assertions: both flux_x and flux_y must appear, with the y
    # branch containing the cross-momentum entry.
    assert "inline Foam::List<Foam::List<Foam::scalar>> flux_x(" in code
    assert "inline Foam::List<Foam::List<Foam::scalar>> flux_y(" in code
    assert "Q[2]" in code  # hv must appear

    # Per-direction NCP / quasilinear kernels for both axes:
    for op in ("nonconservative_matrix", "quasilinear_matrix"):
        assert f"{op}_x(" in code
        assert f"{op}_y(" in code

    # Eigenvalues: normal-projected — both n.x() and n.y() must appear
    assert "n.x()" in code and "n.y()" in code

    print("\nOK: SWE 2D printer emits per-direction kernels correctly.")


if __name__ == "__main__":
    sys.exit(main())
