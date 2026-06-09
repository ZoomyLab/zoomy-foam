"""SWE 1D with bed b and Manning friction — reusable test case.

State (3):
    b   bed elevation (advected with zero flux — bed-slope sourced via NCP)
    h   water depth
    hu  depth-averaged momentum

Flux:
    F[0] = 0
    F[1] = hu
    F[2] = hu^2/h + g·h^2/2

Nonconservative product (bed-slope source):
    B[2, 0] · ∂_x b = g·h · ∂_x b

Source (Manning friction):
    S[2] = -g · n_m^2 · hu · |hu| / h^(7/3)

Boundary conditions: ``wall``, ``inflow``, ``outflow`` (Extrapolation
placeholders for Phase 1; refined in Phase 4 with the symbolic
Piecewise dispatch).

Once the OpenFOAM backend is end-to-end, this case is the natural target
for SWASHES-style comparison (Mac dam-break with friction).
"""

from __future__ import annotations

import numpy as np
from sympy import Matrix, Abs

from zoomy_core.misc.misc import ZArray
from zoomy_core.model import boundary_conditions as BC
from zoomy_core.model import initial_conditions as IC
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.systemmodel.system_model import SystemModel


class SWEBedFriction1D(StructuredDerivativeModel):
    """1D shallow water + bed + Manning friction."""

    dimension = 1
    variables = ["b", "h", "hu"]
    parameters = {"g": (9.81, "positive"), "n_m": (0.033, "positive")}

    def flux(self):
        h, hu = self.Q.h, self.Q.hu
        g = self.params.g
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[0, 0] = 0           # bed: no flux
        F[1, 0] = hu          # continuity
        F[2, 0] = hu * hu / h + 0.5 * g * h * h  # momentum
        return ZArray(F)

    def nonconservative_matrix(self):
        """B[i, j, d]: contribution to eq i from ∂_d state[j].

        Only nonzero entry: B[2, 0, 0] = g·h  (momentum eq picks up
        g·h·∂_x b from the bed-slope source written as an NCP).
        """
        h = self.Q.h
        g = self.params.g
        B = [[[0] * self.dimension for _ in range(self.n_variables)]
             for _ in range(self.n_variables)]
        B[2][0][0] = g * h
        return ZArray(B)

    def source(self):
        """Manning friction on the depth-averaged momentum row."""
        h, hu = self.Q.h, self.Q.hu
        g = self.params.g
        n_m = self.params.n_m
        S = Matrix.zeros(self.n_variables, 1)
        # τ_b = -g · n_m^2 · |u| · u / h^(1/3),  applied to momentum (h·u)
        # In conserved form: -g · n_m^2 · hu · |hu| / h^(7/3)
        S[2, 0] = -g * n_m * n_m * hu * Abs(hu) / h ** (7 / 3)
        return ZArray(S)


def make_test_case(domain=(0.0, 10.0), x0=5.0, h_L=0.5, h_R=0.01,
                   n_m=0.033, g=9.81):
    """Return a frozen :class:`SystemModel` for the SWASHES-style
    Mac dam-break-with-friction problem.

    Parameters mirror SWASHES' "dressler_friction_1D" / "mac_1D"
    families so the C++ solver output can be compared against the
    analytical (or quasi-analytical) reference once the OpenFOAM
    backend runs end-to-end.
    """
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="wall"),
        BC.Extrapolation(tag="inflow"),
        BC.Extrapolation(tag="outflow"),
    ])

    def ic(x):
        Q = np.zeros(3, dtype=float)
        Q[0] = 0.0                       # flat bed
        Q[1] = h_L if x[0] < x0 else h_R # dam-break depth
        Q[2] = 0.0                       # zero initial velocity
        return Q

    model = SWEBedFriction1D(
        parameters={"g": g, "n_m": n_m},
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(function=ic),
    )
    return SystemModel.from_model(model)
