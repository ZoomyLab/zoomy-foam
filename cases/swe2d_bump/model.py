#!/usr/bin/env python
"""swe2d_bump — THE model derivation file (the only place the PDE lives).

Emits Model.H / NumericsKernels.H / UpdateAuxVariables.H for the 2D SWE
from the official ``zoomy_core.model.models.SWE`` class with the UNION
boundary-tag set (inflow, outflow, sides, coupled): one binary serves the
monolithic run and both coupled participants — the mesh's patch tags
select the behavior.  Change the model here, re-run compile.sh, rerun.

Usage: model.py [--out DIR] [--h-in H] [--q-in Q]
"""
import argparse
from pathlib import Path

from zoomy_core.model.models import SWE
from zoomy_core.model import boundary_conditions as BC
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.transformation.to_openfoam import (
    FoamSystemModelPrinter, FoamNumericsPrinter, FoamUpdateAuxPrinter)

HERE = Path(__file__).resolve().parent
FOAM_ROOT = HERE.parent.parent          # library/zoomy_foam


def build_system_model(h_in, q_in):
    bcs = BC.BoundaryConditions([
        BC.FromModel(tag="inflow", definition="inflow"),
        BC.Extrapolation(tag="outflow"),
        BC.FromModel(tag="sides", definition="wall_y"),
        BC.Coupled(tag="coupled", mesh_name="interface"),
    ])
    return SWE(dimension=2,
               parameters={"h_in": h_in, "q_in": q_in},
               boundary_conditions=bcs).system_model


def emit(out, h_in=0.2, q_in=0.1):
    sm = build_system_model(h_in, q_in)
    num = PositiveNonconservativeRusanov(model=sm)
    # ANALYTICAL eigenvalues are required in 2D: the numerical fallback in
    # UserFunctions::max_wavespeed is 1-D-only (A = n_x A_x), which gives
    # ZERO Rusanov dissipation on y-faces -> a 10%/step transverse
    # instability on any non-uniform background (measured on the ridge
    # control).  SWE has the closed form u.n +- sqrt(g h)|n|.
    FoamSystemModelPrinter.write_code(sm, out / "Model.H",
                                      analytical_eigenvalues=True)
    FoamNumericsPrinter.write_code(num, out / "NumericsKernels.H")
    FoamUpdateAuxPrinter.write_code(sm, out / "UpdateAuxVariables.H")
    print(f"emitted SWE(dim=2) -> {out}  state={[str(s) for s in sm.state]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=FOAM_ROOT)
    ap.add_argument("--h-in", type=float, default=0.2)
    ap.add_argument("--q-in", type=float, default=0.1)
    a = ap.parse_args()
    emit(a.out, a.h_in, a.q_in)
