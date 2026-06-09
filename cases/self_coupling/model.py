"""Shared SWE+bed model for the preCICE self-coupling demo.

Both participants (domainA, domainB) and the single-domain reference all
build from THIS one model, so they share one Model.H + one zoomyFoam binary.
The only per-participant differences are runtime case files: the mesh, the
controlDict participant/mesh name, and the initial fields.

Patches are named symmetrically in both sub-domain meshes:

    outer     — transmissive (Extrapolation) outer boundary
    coupled   — the preCICE interface (Coupled BC; extrapolation fallback)

so the emitted Model.H is identical for domainA and domainB.  The preCICE
mesh name each participant provides (MeshA / MeshB) is supplied at runtime
via controlDict ``preciceMeshes``, not baked into the model.

The interface always exchanges the canonical interpolate_3d field set
[b, h, u, v, w, p]; project_3d is the SWE identity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sympy as sp
from sympy import Matrix

from zoomy_core.misc.misc import ZArray
from zoomy_core.model import boundary_conditions as BC
from zoomy_core.model import initial_conditions as IC
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.transformation.to_openfoam import (
    FoamNumericsPrinter,
    FoamSystemModelPrinter,
    FoamUpdateAuxPrinter,
)

FOAM_ROOT = Path(__file__).resolve().parents[2]   # library/zoomy_foam


class SWECoupled1D(StructuredDerivativeModel):
    """1D shallow water + bed, momentum flux split into advective +
    hydrostatic so PositiveNonconservativeRusanov's WB machinery applies."""

    dimension = 1
    variables = ["b", "h", "hu"]
    parameters = {"g": (9.81, "positive")}

    def flux(self):
        h, hu = self.Q.h, self.Q.hu
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[1, 0] = hu
        F[2, 0] = hu * hu / h
        return ZArray(F)

    def hydrostatic_pressure(self):
        h, g = self.Q.h, self.params.g
        P = Matrix.zeros(self.n_variables, self.dimension)
        P[2, 0] = 0.5 * g * h * h
        return ZArray(P)

    def nonconservative_matrix(self):
        h, g = self.Q.h, self.params.g
        B = [[[0] * self.dimension for _ in range(self.n_variables)]
             for _ in range(self.n_variables)]
        B[2][0][0] = g * h
        return ZArray(B)

    def reconstruction_variables(self):
        b, h, hu = self.Q.b, self.Q.h, self.Q.hu
        return ZArray([b, h + b, hu / h])

    # ── Phase-7 coupling projections ─────────────────────────────────
    def interpolate_3d(self):
        """Lift the depth-averaged state to the canonical 3D field set at
        height z.  v=w=0 (1D, derivative-free demo), p hydrostatic."""
        z = self.position[2]
        b, h, hu = self.Q.b, self.Q.h, self.Q.hu
        g = self.params.g
        eta = b + h
        u = hu / h
        return ZArray([b, h, u, sp.Integer(0), sp.Integer(0), g * (eta - z)])

    def project_3d(self):
        """SWE identity: recover [b, h, hu] from a depth-representative
        3D profile [b, h, u, v, w, p]."""
        b, h, u = sp.symbols("P3_b P3_h P3_u", real=True)
        return ZArray([b, h, h * u])


def build():
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="outer"),
        BC.Coupled(tag="coupled", mesh_name="interface"),
    ])
    ic = IC.UserFunction(function=lambda x: np.zeros(3))
    model = SWECoupled1D(boundary_conditions=bcs, initial_conditions=ic)
    return model, SystemModel.from_model(model)


def write_headers():
    """Emit Model.H + NumericsKernels.H + UpdateAuxVariables.H into the
    shared zoomy_foam tree."""
    model, sm = build()
    numerics = PositiveNonconservativeRusanov(model=sm)
    FoamSystemModelPrinter.write_code(
        sm, FOAM_ROOT / "Model.H",
        analytical_eigenvalues=True,
        project_3d=model.project_3d(),
    )
    FoamNumericsPrinter.write_code(numerics, FOAM_ROOT / "NumericsKernels.H")
    FoamUpdateAuxPrinter.write_code(sm, FOAM_ROOT / "UpdateAuxVariables.H")
    print(f"  → headers written to {FOAM_ROOT}")


if __name__ == "__main__":
    write_headers()
