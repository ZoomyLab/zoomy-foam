"""SWE 1D with bed b held as an *auxiliary* variable + Manning friction.

State (2):
    h, hu

Aux (1):
    b   bed elevation — supplied externally (pre-loaded by the case);
        SystemModel auto-registers ∂_x b in aux_registry once the
        source/NCP expressions reference it.

Flux (2):
    F[0] = hu
    F[1] = hu^2/h + g·h^2/2

Source (2):
    S[0] = 0
    S[1] = -g·h·∂_x b   ← bed-slope source via Derivative of aux
           -g·n_m^2·hu·|hu|/h^(7/3)    ← Manning friction

The Derivative(b, x) atom in S[1] triggers ``SystemModel`` to register
``b_x`` in ``aux_registry`` with ``kind="derivative"``, ``target_kind=
"aux"``, ``state_index=0`` (b is aux[0]), ``multi_index=(1,)``.

This is the natural test target for Phase 3 — the Foam backend must
populate the b_x entry every step by calling its ``compute_derivative``
helper against the bed field.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from sympy import Matrix, Abs

from zoomy_core.misc.misc import ZArray
from zoomy_core.model import boundary_conditions as BC
from zoomy_core.model import initial_conditions as IC
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.systemmodel.system_model import SystemModel


class SWEBedAsAux1D(StructuredDerivativeModel):
    """1D shallow water + Manning friction + bed as auxiliary.

    Note: ``StructuredDerivativeModel`` exposes user-supplied auxiliaries
    via ``user_aux_variables`` (separate from the auto-derivative-buffer
    auxiliaries).  Access is ``self.A.b`` (the ``A`` alias is set up by
    the parent's __init__).
    """

    dimension = 1
    variables = ["h", "hu"]
    user_aux_variables = ["b"]
    parameters = {"g": (9.81, "positive"), "n_m": (0.033, "positive")}

    def flux(self):
        h, hu = self.Q.h, self.Q.hu
        g = self.params.g
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[0, 0] = hu
        F[1, 0] = hu * hu / h + 0.5 * g * h * h
        return ZArray(F)

    def source(self):
        h, hu = self.Q.h, self.Q.hu
        b = self.A.b
        g, n_m = self.params.g, self.params.n_m
        x = self.position.x
        S = Matrix.zeros(self.n_variables, 1)
        # Bed-slope source — sp.Derivative with evaluate=False so the atom
        # survives into the SystemModel and triggers aux_registry.
        # (Plain sp.diff(Symbol, Symbol) → 0; we need the unevaluated form.)
        b_x = sp.Derivative(b, x, evaluate=False)
        S[1, 0] = -g * h * b_x
        # Manning friction (algebraic).
        S[1, 0] += -g * n_m * n_m * hu * Abs(hu) / h ** (7 / 3)
        return ZArray(S)


def make_test_case(h_L=0.5, h_R=0.01):
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="wall"),
        BC.Extrapolation(tag="inflow"),
        BC.Extrapolation(tag="outflow"),
    ])

    def ic(x):
        Q = np.zeros(2, dtype=float)
        Q[0] = h_L if x[0] < 5.0 else h_R
        Q[1] = 0.0
        return Q

    model = SWEBedAsAux1D(
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(function=ic),
    )
    return SystemModel.from_model(model)
