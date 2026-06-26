#!/usr/bin/env python
"""Emit the declarative SME model (Kowalski–Torrilhon) at a given ``level`` to the
zoomyFoam C++ headers (``Model.H`` / ``NumericsKernels.H`` / ``UpdateAuxVariables.H``).

Clean path (zoomy_core ≥ 2ca1e69): the model owns EVERYTHING —
BCs via the constructor, the WB reconstruction (η=b+h, u_i=q_i/h) and the
interface projection ride the model's function-group slots. This script only
chooses the BC set for the coupling cases and points the printers at the slots.

level 0 = shallow water [b, h, q_0]; level N adds moments q_1 … q_N.
"""
import argparse
from pathlib import Path

from zoomy_core.model.models import SME
from zoomy_core.model.models import closures as C
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation, Coupled, FromModel, Dirichlet)
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.transformation.to_openfoam import (
    FoamSystemModelPrinter, FoamNumericsPrinter, FoamUpdateAuxPrinter)

HERE = Path(__file__).resolve().parent


def build_system_model(level, outer="extrapolation", closure="none", bcs="coupling",
                       q_in=1.0, h_out=1.0):
    # Boundary set.  "coupling" is the production preCICE pair (outer + coupled);
    # "open" is a plain two-sided extrapolation channel; "subcritical" is the
    # discharge-in / depth-out Dirichlet pair a well-posed steady subcritical
    # case needs (e.g. the MacDonald friction profile — transmissive ends let
    # the discharge drain off).  q_in fixes q_0 at the left (b,h extrapolated),
    # h_out fixes h at the right (b,q extrapolated).
    if bcs == "subcritical":
        boundary = BoundaryConditions([Dirichlet(tag="left", on="q_0", value=q_in),
                                       Dirichlet(tag="right", on="h", value=h_out)])
    elif bcs == "open":
        boundary = BoundaryConditions([Extrapolation(tag="left"),
                                       Extrapolation(tag="right")])
    else:
        outer_bc = (FromModel(tag="outer", definition="wall")
                    if outer == "wall" else Extrapolation(tag="outer"))
        boundary = BoundaryConditions([outer_bc,
                                       Coupled(tag="coupled", mesh_name="interface")])
    # Optional bottom-friction closure.  "manning" closes the bed stress into the
    # standard Manning term  −g·n²·q|q|/h^(7/3)  on the momentum row — the stiff
    # source the IMEX scheme treats implicitly (and exactly the SWASHES
    # MacDonald-case friction).  "none" = inviscid (production default).
    closures = [C.ManningFriction()] if closure == "manning" else []
    return SME(level=level, closures=closures,
               boundary_conditions=boundary).system_model


def emit(level, out=HERE, outer="extrapolation", closure="none", bcs="coupling",
         q_in=1.0, h_out=1.0):
    sm = build_system_model(level, outer=outer, closure=closure, bcs=bcs,
                            q_in=q_in, h_out=h_out)
    num = PositiveNonconservativeRusanov(model=sm)
    FoamSystemModelPrinter.write_code(
        sm, out / "Model.H",
        analytical_eigenvalues=False)   # numerical eigenvalues; projections read
                                        # from the model-owned slots (>= 1347b56)
    FoamNumericsPrinter.write_code(num, out / "NumericsKernels.H")
    FoamUpdateAuxPrinter.write_code(sm, out / "UpdateAuxVariables.H")
    print(f"emitted SME(level={level}, closure={closure}, bcs={bcs}) -> {out}  "
          f"state={[str(s) for s in sm.state]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--outer", choices=["extrapolation", "wall"],
                    default="extrapolation")
    ap.add_argument("--closure", choices=["none", "manning"], default="none")
    ap.add_argument("--bcs", choices=["coupling", "open", "subcritical"],
                    default="coupling")
    ap.add_argument("--q-in", type=float, default=1.0)
    ap.add_argument("--h-out", type=float, default=1.0)
    a = ap.parse_args()
    emit(a.level, a.out, outer=a.outer, closure=a.closure, bcs=a.bcs,
         q_in=a.q_in, h_out=a.h_out)
