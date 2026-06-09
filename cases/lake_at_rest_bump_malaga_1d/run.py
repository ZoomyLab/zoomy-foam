#!/usr/bin/env python
"""Malaga-style formulation — lake-at-rest WB test.

Pressure moves OUT of flux + hydrostatic_pressure INTO the NCP.  Plain
``NonconservativeRusanov`` (no HR / Audusse hydrostatic reconstruction)
is used.  This is the user's suggested alternative formulation:

  state:                [b, h, hu]
  flux[1] = hu                       (continuity)
  flux[2] = hu*hu/h                  (advective momentum ONLY — no pressure)
  hydrostatic_pressure  = 0          (NO split for HR)
  B[2,0,0] = g*h                     (bed-slope NCP)
  B[2,1,0] = g*h                     (pressure NCP: ∂(½ g h²)/∂x = g·h·∂h/∂x)

For lake-at-rest (η = h+b = const, hu = 0) the NCP path integral collapses
to ``g·h·(η_R - η_L) = 0`` so the conservative + NCP contribution at the
face vanishes exactly.  The OPEN QUESTION: plain Rusanov dissipation
0.5·s·(Q_R - Q_L) is non-zero on the h component (h_L = η - b_owner
≠ η - b_nei = h_R), so the dissipation breaks WB unless suppressed.

This run measures how big that WB-violation is — informs whether the
Malaga path needs additional WB correction or whether HR is required.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from sympy import Matrix

HERE = Path(__file__).resolve().parent
FOAM_ROOT = HERE.parent.parent

from zoomy_core.misc.misc import ZArray
from zoomy_core.model import boundary_conditions as BC
from zoomy_core.model import initial_conditions as IC
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.fvm.riemann_solvers import WellBalancedNonconservativeRusanov
from zoomy_core.transformation.to_openfoam import (
    FoamNumericsPrinter,
    FoamSystemModelPrinter,
    FoamUpdateAuxPrinter,
)


# ── 1. Model: Malaga formulation ───────────────────────────────────────


class SWEBedMalaga1D(StructuredDerivativeModel):
    dimension = 1
    variables = ["b", "h", "hu"]
    parameters = {"g": (9.81, "positive")}

    def flux(self):
        hu = self.Q.hu
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[1, 0] = hu                      # continuity
        F[2, 0] = hu * hu / self.Q.h      # advective momentum — NO pressure
        return ZArray(F)

    # NO hydrostatic_pressure() override → uses base default (zero).

    def nonconservative_matrix(self):
        h = self.Q.h
        g = self.params.g
        B = [[[0] * self.dimension for _ in range(self.n_variables)]
             for _ in range(self.n_variables)]
        B[2][0][0] = g * h                 # bed-slope NCP
        B[2][1][0] = g * h                 # pressure as NCP
        return ZArray(B)

    def source(self):
        return ZArray.zeros(self.n_variables)

    def reconstruction_variables(self):
        b, h, hu = self.Q.b, self.Q.h, self.Q.hu
        return ZArray([b, h + b, hu / h])


def build_system_model():
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="wall"),
        BC.Extrapolation(tag="inflow"),
        BC.Extrapolation(tag="outflow"),
    ])
    ic = IC.UserFunction(function=lambda x: np.zeros(3))
    model = SWEBedMalaga1D(boundary_conditions=bcs, initial_conditions=ic)
    return SystemModel.from_model(model)


def write_headers():
    sm = build_system_model()
    numerics = WellBalancedNonconservativeRusanov(model=sm)
    FoamSystemModelPrinter.write_code(
        sm, FOAM_ROOT / "Model.H", analytical_eigenvalues=True
    )
    FoamNumericsPrinter.write_code(numerics, FOAM_ROOT / "NumericsKernels.H")
    FoamUpdateAuxPrinter.write_code(sm, FOAM_ROOT / "UpdateAuxVariables.H")
    print(f"  → headers written to {FOAM_ROOT}")


# ── 2. IC: same as the HR lake-at-rest case ────────────────────────────


ETA0, B_PEAK, B_CENT, B_SIGMA, H_MIN = 0.5, 0.2, 12.5, 1.0, 1e-3
X_MIN, X_MAX, N_CELLS = 0.0, 25.0, 200


def bed(x):
    return B_PEAK * np.exp(-((x - B_CENT) / B_SIGMA) ** 2)


def cell_centres():
    edges = np.linspace(X_MIN, X_MAX, N_CELLS + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _write_foam_field(path, name, vals):
    body = "\n".join(f"{v:.14e}" for v in vals)
    path.write_text(
        f"""FoamFile
{{ format ascii; class volScalarField; object {name}; }}
dimensions [0 0 0 0 0 0 0];
internalField nonuniform List<scalar>
{vals.size}
(
{body}
)
;
boundaryField {{
    inflow        {{ type zeroGradient; }}
    outflow       {{ type zeroGradient; }}
    sides         {{ type empty; }}
    topAndBottom  {{ type empty; }}
}}
"""
    )


def write_initial_fields():
    xc = cell_centres()
    b_f  = bed(xc)
    h_f  = np.maximum(ETA0 - b_f, H_MIN)
    hu_f = np.zeros_like(xc)
    z = HERE / "0"
    z.mkdir(exist_ok=True)
    _write_foam_field(z / "Q0", "Q0", b_f)
    _write_foam_field(z / "Q1", "Q1", h_f)
    _write_foam_field(z / "Q2", "Q2", hu_f)


BASHRC = "/opt/openfoam13/etc/bashrc"


def _run(cmd, cwd, log=None):
    if log:
        with open(log, "w") as f:
            r = subprocess.run(["bash", "-c", cmd], cwd=cwd,
                               stdout=f, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(["bash", "-c", cmd], cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"cmd failed (rc={r.returncode}); see {log}")


def driver():
    import shutil
    for d in HERE.glob("[0-9]*"):
        if d.is_dir(): shutil.rmtree(d)
    for d in (HERE / "constant").glob("polyMesh"):
        shutil.rmtree(d)
    for f in HERE.glob("log.*"): f.unlink()

    print("[1/4] wmake…")
    _run(f"source {BASHRC} && wmake", FOAM_ROOT, log=HERE / "log.wmake")
    print("[2/4] blockMesh…")
    _run(f"source {BASHRC} && blockMesh", HERE, log=HERE / "log.blockMesh")
    print("[3/4] Write IC…")
    write_initial_fields()
    print("[4/4] zoomyFoam…")
    _run(f"source {BASHRC} && unset FOAM_SIGFPE FOAM_SETNAN && zoomyFoam",
         HERE, log=HERE / "log.zoomyFoam")


def _read_internal(p):
    text = p.read_text()
    m = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(([^)]+)\)",
        text, re.DOTALL,
    )
    if m: return np.fromstring(m.group(2), sep="\n")
    m = re.search(r"internalField\s+uniform\s+([0-9eE.+\-]+)", text)
    return np.full(N_CELLS, float(m.group(1)))


def report():
    last = sorted(
        (float(d.name), d) for d in HERE.iterdir()
        if d.is_dir() and re.fullmatch(r"\d+(?:\.\d+)?", d.name)
        and (d / "Q1").exists()
    )[-1][1]
    h  = _read_internal(last / "Q1")
    hu = _read_internal(last / "Q2")
    b  = _read_internal(last / "Q0")
    u  = np.where(h > 1e-12, hu / h, 0.0)
    print(f"\n=== Malaga lake-at-rest result ===")
    print(f"  t                = {last.name}")
    print(f"  max |u|           = {np.max(np.abs(u)):.3e}  m/s")
    print(f"  max |η - {ETA0}|   = {np.max(np.abs(h + b - ETA0)):.3e}  m")
    print(f"  HR reference       max|u| = 2.7e-15 (machine eps)")
    print(f"  (rate)             pass if both ~ machine eps")


def main():
    write_headers()
    driver()
    report()


if __name__ == "__main__":
    main()
