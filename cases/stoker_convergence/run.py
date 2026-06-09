#!/usr/bin/env python
"""Stoker wet-wet dam-break convergence study.

Drives runs on N ∈ {100, 200, 400, 800, 1600} cells at both
reconstructionOrder ∈ {1, 2}.  Compares cell-centre depth + momentum
against the closed-form Stoker analytical solution; plots L1 error
vs cell size on a log-log panel.

Note (per the plan): Stoker has a shock at t=1s.  TVD limiters cap
global L1 convergence at ~1 regardless of formal scheme order.  The
2nd-order benefit shows as a **smaller multiplicative constant** —
the order-2 line sits ~2-4× below the order-1 line on the log-log
plot, both with slope ≈ 1.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sympy import Matrix

# ── locate roots, set up imports ───────────────────────────────────────
HERE = Path(__file__).resolve().parent
FOAM_ROOT = HERE.parent.parent
sys.path.insert(0, str(FOAM_ROOT / "tools"))

from compare_stoker import stoker

from zoomy_core.misc.misc import ZArray
from zoomy_core.model import boundary_conditions as BC
from zoomy_core.model import initial_conditions as IC
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.transformation.to_openfoam import (
    FoamNumericsPrinter,
    FoamSystemModelPrinter,
    FoamUpdateAuxPrinter,
)


# ── 1. Model: SWE 1D + bed, no friction ────────────────────────────────


class SWEBed1D(StructuredDerivativeModel):
    """SWE 1D with bed b in state, no friction.  Split flux + hydrostatic
    pressure for Audusse HR-WB compatibility."""

    dimension = 1
    variables = ["b", "h", "hu"]
    parameters = {"g": (9.81, "positive")}

    def flux(self):
        h, hu = self.Q.h, self.Q.hu
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[1, 0] = hu                 # continuity
        F[2, 0] = hu * hu / h        # advective momentum
        return ZArray(F)

    def hydrostatic_pressure(self):
        h = self.Q.h
        g = self.params.g
        P = Matrix.zeros(self.n_variables, self.dimension)
        P[2, 0] = 0.5 * g * h * h
        return ZArray(P)

    def nonconservative_matrix(self):
        h = self.Q.h
        g = self.params.g
        B = [[[0] * self.dimension for _ in range(self.n_variables)]
             for _ in range(self.n_variables)]
        B[2][0][0] = g * h
        return ZArray(B)

    def source(self):
        return ZArray.zeros(self.n_variables)

    def reconstruction_variables(self):
        b, h, hu = self.Q.b, self.Q.h, self.Q.hu
        return ZArray([b, h + b, hu / h])


def build_system_model():
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="inflow"),
        BC.Extrapolation(tag="outflow"),
    ])
    ic = IC.UserFunction(function=lambda x: np.zeros(3))
    model = SWEBed1D(boundary_conditions=bcs, initial_conditions=ic)
    return SystemModel.from_model(model)


def write_headers():
    sm = build_system_model()
    numerics = PositiveNonconservativeRusanov(model=sm)
    FoamSystemModelPrinter.write_code(
        sm, FOAM_ROOT / "Model.H", analytical_eigenvalues=True
    )
    FoamNumericsPrinter.write_code(numerics, FOAM_ROOT / "NumericsKernels.H")
    FoamUpdateAuxPrinter.write_code(sm, FOAM_ROOT / "UpdateAuxVariables.H")


# ── 2. Stoker setup ────────────────────────────────────────────────────

X_MIN, X_MAX = 0.0, 10.0
X0 = 5.0
H_L = 0.5
H_R = 0.01
T_END = 1.0
G = 9.81


def cellcent(n):
    edges = np.linspace(X_MIN, X_MAX, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _write_blockmesh(case_dir, n):
    (case_dir / "system" / "blockMeshDict").write_text(
        f"""FoamFile
{{ format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
vertices (
    (0 0 0) (10 0 0) (10 1 0) (0 1 0)
    (0 0 1) (10 0 1) (10 1 1) (0 1 1)
);
blocks ( hex (0 1 2 3 4 5 6 7) ({n} 1 1) simpleGrading (1 1 1) );
boundary (
    inflow       {{ type patch; faces ((0 4 7 3)); }}
    outflow      {{ type patch; faces ((1 2 6 5)); }}
    sides        {{ type empty; faces ((0 1 5 4) (3 7 6 2)); }}
    topAndBottom {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();
"""
    )


def _write_foam_field(path: Path, name: str, vals: np.ndarray):
    body = "\n".join(f"{v:.14e}" for v in vals)
    path.write_text(
        f"""FoamFile
{{ format ascii; class volScalarField; object {name}; }}
dimensions      [0 0 0 0 0 0 0];
internalField   nonuniform List<scalar>
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


def _write_initial(case_dir, n):
    xc = cellcent(n)
    b  = np.zeros(n)
    h  = np.where(xc < X0, H_L, H_R)
    hu = np.zeros(n)
    zero = case_dir / "0"
    zero.mkdir(exist_ok=True)
    _write_foam_field(zero / "Q0", "Q0", b)
    _write_foam_field(zero / "Q1", "Q1", h)
    _write_foam_field(zero / "Q2", "Q2", hu)


def _write_controldict(case_dir, order):
    (case_dir / "system" / "controlDict").write_text(
        f"""FoamFile
{{ format ascii; class dictionary; object controlDict; }}
application zoomyFoam;
startFrom startTime; startTime 0;
stopAt endTime; endTime {T_END};
deltaT 0.0001;
writeControl adjustableRunTime; writeInterval {T_END};
purgeWrite 0; writeFormat ascii; writePrecision 12; timeFormat general;
runTimeModifiable true; adjustTimeStep no; maxCo 0.4;
reconstructionOrder {order};
"""
    )


def _write_fvschemes(case_dir):
    (case_dir / "system" / "fvSchemes").write_text(
        """FoamFile
{ format ascii; class dictionary; object fvSchemes; }
ddtSchemes           { default Euler; }
gradSchemes          { default cellLimited Gauss linear 1; }
divSchemes           { default Gauss linear; }
laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }
"""
    )


def _write_fvsolution(case_dir):
    (case_dir / "system" / "fvSolution").write_text(
        """FoamFile
{ format ascii; class dictionary; object fvSolution; }
solvers { "Q.*" { solver diagonal; } }
"""
    )


def _read_internal(p: Path, n: int) -> np.ndarray:
    text = p.read_text()
    m = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(([^)]+)\)",
        text, re.DOTALL,
    )
    if m:
        return np.fromstring(m.group(2), sep="\n")
    m = re.search(r"internalField\s+uniform\s+([0-9eE.+\-]+)", text)
    if m:
        return np.full(n, float(m.group(1)))
    raise ValueError(f"can't parse {p}")


# ── 3. Driver ──────────────────────────────────────────────────────────

BASHRC = "/opt/openfoam13/etc/bashrc"


def _run(cmd, cwd):
    return subprocess.run(["bash", "-c", cmd], cwd=cwd, check=True,
                          capture_output=True)


def run_case(n, order):
    case_dir = HERE / f"run_N{n}_O{order}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    (case_dir / "system").mkdir()
    (case_dir / "constant").mkdir()
    _write_blockmesh(case_dir, n)
    _write_controldict(case_dir, order)
    _write_fvschemes(case_dir)
    _write_fvsolution(case_dir)
    _run(f"source {BASHRC} && blockMesh", case_dir)
    _write_initial(case_dir, n)
    _run(f"source {BASHRC} && unset FOAM_SIGFPE FOAM_SETNAN && zoomyFoam",
         case_dir)
    times = sorted(
        (float(d.name), d) for d in case_dir.iterdir()
        if d.is_dir() and re.fullmatch(r"\d+(?:\.\d+)?", d.name)
        and (d / "Q1").exists()
    )
    last = times[-1][1]
    h  = _read_internal(last / "Q1", n)
    hu = _read_internal(last / "Q2", n)
    return h, hu


# ── 4. Main: build, sweep, plot ────────────────────────────────────────


def main():
    print("[1/3] Build solver + emit headers…")
    write_headers()
    _run(f"source {BASHRC} && wmake", FOAM_ROOT)

    Ns = [100, 200, 400, 800, 1600]
    orders = [1, 2]
    results = {}

    print("[2/3] Sweep N × order:")
    for n in Ns:
        for o in orders:
            h, hu = run_case(n, o)
            xc = cellcent(n)
            h_an, u_an = stoker(xc, T_END, H_L, H_R, X0, G)
            hu_an = h_an * u_an
            l1_h  = float(np.mean(np.abs(h  - h_an)))
            l1_hu = float(np.mean(np.abs(hu - hu_an)))
            results[(n, o)] = (l1_h, l1_hu)
            print(f"  N={n:5d}  order={o}  L1 h={l1_h:.4e}  L1 hu={l1_hu:.4e}")

    # Rates (consecutive-N ratios)
    print("\n   N pair    rate(h, O1)  rate(h, O2)   rate(hu, O1)  rate(hu, O2)")
    for i in range(1, len(Ns)):
        a, b = Ns[i - 1], Ns[i]
        rh1  = np.log2(results[(a, 1)][0] / results[(b, 1)][0])
        rh2  = np.log2(results[(a, 2)][0] / results[(b, 2)][0])
        rhu1 = np.log2(results[(a, 1)][1] / results[(b, 1)][1])
        rhu2 = np.log2(results[(a, 2)][1] / results[(b, 2)][1])
        print(f"  {a:4d}→{b:4d}    {rh1:.3f}        {rh2:.3f}         "
              f"{rhu1:.3f}         {rhu2:.3f}")

    print("\n[3/3] Plot…")
    h_cells = [(X_MAX - X_MIN) / n for n in Ns]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, var, idx in [(axes[0], "h", 0), (axes[1], "hu", 1)]:
        for o, marker, label, col in [(1, "o", "order 1", "C0"),
                                       (2, "s", "order 2", "C3")]:
            errs = [results[(n, o)][idx] for n in Ns]
            ax.loglog(h_cells, errs, marker=marker, label=label,
                      color=col, lw=1.4, ms=7)
        # Reference slope-1 line anchored at the order-1 coarsest point.
        e0 = results[(Ns[0], 1)][idx]
        ax.loglog(
            [h_cells[0], h_cells[-1]],
            [e0, e0 * h_cells[-1] / h_cells[0]],
            "k--", lw=0.8, alpha=0.5, label="slope 1 reference",
        )
        ax.set_xlabel("cell size  h  [m]")
        ax.set_ylabel(f"L1 error in {var}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_title(f"Stoker wet-wet dam-break — L1 error in {var}")

    fig.suptitle(
        f"Convergence study  (h_L={H_L}, h_R={H_R}, x₀={X0}, t={T_END}, "
        f"PositiveNonconservativeRusanov + HR-WB)\n"
        "Global L1 rate is TVD-capped at ≈ 1 by the shock; "
        "2nd order shows as a downward shift, not steeper slope.",
        fontsize=10,
    )
    fig.tight_layout()
    out = HERE / "convergence_panel.png"
    fig.savefig(out, dpi=130)
    print(f"  → plot saved to {out}")


if __name__ == "__main__":
    main()
