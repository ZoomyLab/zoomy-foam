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

import sympy as sp

from zoomy_core.model.models import SME, VAM
from zoomy_core.model.models import closures as C
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Extrapolation, Coupled, FromModel, Dirichlet)
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.transformation.to_openfoam import (
    FoamSystemModelPrinter, FoamNumericsPrinter)

HERE = Path(__file__).resolve().parent


def emit_chorin(level=1, dim=2, out=HERE, bcs="open"):
    """Emit the Chorin pressure-split headers for the non-hydrostatic VAM solver.

    The predictor sub-system is emitted as the MAIN model (``namespace Model`` +
    a Riemann ``Numerics`` over it) so the explicit FV machinery runs it; the
    pressure and corrector sub-systems get their own namespaces.  A tiny
    ``ChorinState.H`` carries the FULL shared-state count (the rectangular
    sub-systems each expose only their own equation count, not the union).

      Model.H          namespace Model           predictor ops (pressure-zeroed)
      NumericsKernels.H namespace Numerics       Riemann flux over the predictor
      Pressure.H       namespace ChorinPressure  elliptic source (P,P_x,P_xx) + e2s
      Corrector.H      namespace ChorinCorrector update_variables(Q,Qaux,p,dt) + e2s
      ChorinState.H                              n_state (full 8-slot VAM state)
    """
    if bcs == "open":
        boundary = BoundaryConditions([Extrapolation(tag="left"),
                                       Extrapolation(tag="right")])
    else:
        boundary = BoundaryConditions([Extrapolation(tag="outer"),
                                       Coupled(tag="coupled", mesh_name="interface")])
    # Close the bulk/slip/surface stresses so σ̂ doesn't leak as a free aux
    # (unclosed VAM emits raw `\hat{\sigma}` symbols into the C++ source).
    m = VAM(level=level, dimension=dim,
            closures=[C.Newtonian(), C.NavierSlip(), C.StressFree()],
            boundary_conditions=boundary)
    full = m.system_model
    n_state = len(full.state)
    dt = sp.Symbol("dt", positive=True)
    split = m.chorin_split(dt)

    FoamSystemModelPrinter.write_code(split.SM_pred, out / "Model.H",
                                      namespace_name="Model")
    FoamNumericsPrinter.write_code(
        PositiveNonconservativeRusanov(model=split.SM_pred),
        out / "NumericsKernels.H")
    FoamSystemModelPrinter.write_code(split.SM_press, out / "Pressure.H",
                                      namespace_name="ChorinPressure", dt_symbol=dt)
    FoamSystemModelPrinter.write_code(split.SM_corr, out / "Corrector.H",
                                      namespace_name="ChorinCorrector")
    (out / "ChorinState.H").write_text(
        "#pragma once\n"
        "// Full shared VAM state count (the rectangular Chorin sub-systems each\n"
        "// expose only their own equation count). Emitted by create_model.py.\n"
        f"namespace Model {{ constexpr int n_state = {n_state}; }}\n")
    print(f"emitted Chorin VAM(level={level},dim={dim}) -> {out}  "
          f"n_state={n_state}  pred/press/corr e2s="
          f"{list(split.SM_pred.equation_to_state_index)}/"
          f"{list(split.SM_press.equation_to_state_index)}/"
          f"{list(split.SM_corr.equation_to_state_index)}")


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
    # Two-printer API (zoomy_core >= 75d8c76): the Model printer emits the full
    # model INCLUDING update_aux_variables (in `namespace Model`) — there is no
    # separate UpdateAuxVariables.H any more.  The Numerics printer emits the
    # Riemann kernels (local_max_abs_eigenvalue now comes from the model
    # eigenvalues; max_wavespeed was dropped).
    FoamSystemModelPrinter.write_code(sm, out / "Model.H")
    FoamNumericsPrinter.write_code(num, out / "NumericsKernels.H")
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
    ap.add_argument("--scheme", choices=["explicit", "chorin"], default="explicit")
    ap.add_argument("--dim", type=int, default=2, help="VAM dimension (Chorin)")
    a = ap.parse_args()
    if a.scheme == "chorin":
        emit_chorin(level=(a.level or 1), dim=a.dim, out=a.out,
                    bcs=("open" if a.bcs != "coupling" else "coupling"))
    else:
        emit(a.level, a.out, outer=a.outer, closure=a.closure, bcs=a.bcs,
             q_in=a.q_in, h_out=a.h_out)
