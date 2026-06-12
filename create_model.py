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
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation, Coupled, FromModel)
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.transformation.to_openfoam import (
    FoamSystemModelPrinter, FoamNumericsPrinter, FoamUpdateAuxPrinter)

HERE = Path(__file__).resolve().parent


def build_system_model(level, outer="extrapolation"):
    outer_bc = (FromModel(tag="outer", definition="wall")
                if outer == "wall" else Extrapolation(tag="outer"))
    bcs = BoundaryConditions([outer_bc,
                              Coupled(tag="coupled", mesh_name="interface")])
    return SME(level=level, boundary_conditions=bcs).system_model


def emit(level, out=HERE, outer="extrapolation"):
    sm = build_system_model(level, outer=outer)
    num = PositiveNonconservativeRusanov(model=sm)
    FoamSystemModelPrinter.write_code(
        sm, out / "Model.H",
        analytical_eigenvalues=False)   # numerical eigenvalues; projections read
                                        # from the model-owned slots (>= 1347b56)
    FoamNumericsPrinter.write_code(num, out / "NumericsKernels.H")
    FoamUpdateAuxPrinter.write_code(sm, out / "UpdateAuxVariables.H")
    print(f"emitted SME(level={level}) -> {out}  state={[str(s) for s in sm.state]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--outer", choices=["extrapolation", "wall"],
                    default="extrapolation")
    a = ap.parse_args()
    emit(a.level, a.out, outer=a.outer)
