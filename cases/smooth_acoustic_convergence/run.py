#!/usr/bin/env python
"""Smooth-IC linear-acoustic convergence test for the Foam backend.

Tiny Gaussian h-perturbation on still water; compare against linearised
d'Alembert split solution.  Pure-smooth flow → no TVD shock-cap → the
2nd-order scheme should reach L1 rate ≈ 1.7–2.0 (depending on how much
nonlinearity sneaks in at the chosen amplitude).

Setup
-----
Domain    x ∈ [0, 20] m,  flat bed b = 0
IC        h(x, 0) = h₀ + Δh · exp(-((x - x₀)/σ)²),   hu(x, 0) = 0
          h₀ = 1.0,  Δh = 0.01 (1 % perturbation),  σ = 1.0,  x₀ = 10
g         9.81 m/s²
t_end     0.5 s — wave travels c·t ≈ 1.6 m, well inside the domain

Linearised reference (d'Alembert split for ∂_t² η = c² ∂_x² η, u₀=0):
    c           = √(g·h₀)
    η(x, t)     = ½ [η₀(x − ct) + η₀(x + ct)]
    u(x, t)     = ½ (c / h₀) · [η₀(x − ct) − η₀(x + ct)]

The two pulses travel left/right at ±c with half amplitude each.

Convergence is measured against the linear analytical.  Δh = 0.01 keeps
nonlinear effects ~ 10⁻⁴ × signal, which is at or below the 2nd-order
truncation error at our finest mesh (N = 800).
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

# ── locate roots ───────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
FOAM_ROOT = HERE.parent.parent

from zoomy_core.misc.misc import ZArray
from zoomy_core.model import boundary_conditions as BC
from zoomy_core.model import initial_conditions as IC
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.fvm.riemann_solvers import (
    PositiveNonconservativeRusanov,
    PositiveNonconservativeHLL,
)
from zoomy_core.transformation.to_openfoam import (
    FoamNumericsPrinter,
    FoamSystemModelPrinter,
    FoamUpdateAuxPrinter,
)


# ── 1. Model — SWE 1D + flat bed, HR-WB compatible ─────────────────────


class SWEBed1D(StructuredDerivativeModel):
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


RIEMANN_CLASS = PositiveNonconservativeRusanov   # vs ...HLL


def write_headers():
    sm = build_system_model()
    numerics = RIEMANN_CLASS(model=sm)
    FoamSystemModelPrinter.write_code(
        sm, FOAM_ROOT / "Model.H", analytical_eigenvalues=True
    )
    FoamNumericsPrinter.write_code(numerics, FOAM_ROOT / "NumericsKernels.H")
    FoamUpdateAuxPrinter.write_code(sm, FOAM_ROOT / "UpdateAuxVariables.H")


# ── 2. Smooth-IC analytical ────────────────────────────────────────────

X_MIN, X_MAX = 0.0, 20.0
H0          = 1.0
DH          = 1.0e-4
SIGMA       = 1.0
X0          = 10.0
T_END       = 0.5
G           = 9.81


def gauss(x):
    return DH * np.exp(-((x - X0) / SIGMA) ** 2)


def dalembert(x, t):
    c = np.sqrt(G * H0)
    eta = 0.5 * (gauss(x - c * t) + gauss(x + c * t))
    u   = 0.5 * (c / H0) * (gauss(x - c * t) - gauss(x + c * t))
    h   = H0 + eta
    hu  = h * u
    return h, hu


def cellcent(n):
    edges = np.linspace(X_MIN, X_MAX, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


# ── 3. Case writers ────────────────────────────────────────────────────


def _write_blockmesh(case_dir, n):
    (case_dir / "system" / "blockMeshDict").write_text(
        f"""FoamFile
{{ format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
vertices (
    (0 0 0) ({X_MAX} 0 0) ({X_MAX} 1 0) (0 1 0)
    (0 0 1) ({X_MAX} 0 1) ({X_MAX} 1 1) (0 1 1)
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


def _write_initial(case_dir, n):
    xc = cellcent(n)
    b  = np.zeros(n)
    h  = H0 + gauss(xc)
    hu = np.zeros(n)
    z = case_dir / "0"
    z.mkdir(exist_ok=True)
    _write_foam_field(z / "Q0", "Q0", b)
    _write_foam_field(z / "Q1", "Q1", h)
    _write_foam_field(z / "Q2", "Q2", hu)


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
runTimeModifiable true; adjustTimeStep no; maxCo 0.3;
reconstructionOrder {order};
"""
    )


def _write_fvschemes(case_dir):
    # Unlimited Gauss linear — smooth IC has no shock, no need to limit.
    # (cellLimited 1 clips ~slope-2 to slope-1 even in smooth regions for
    # this Gaussian wave; see Phase 5.6 diagnosis notes.)
    (case_dir / "system" / "fvSchemes").write_text(
        """FoamFile
{ format ascii; class dictionary; object fvSchemes; }
ddtSchemes           { default Euler; }
gradSchemes          { default Gauss linear; }
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


def _read_internal(p, n):
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


# ── 4. Main ────────────────────────────────────────────────────────────


def main():
    print("[1/3] Build solver + emit headers…")
    write_headers()
    _run(f"source {BASHRC} && wmake", FOAM_ROOT)

    Ns = [50, 100, 200, 400, 800]
    orders = [1, 2]
    results = {}

    print("[2/3] Sweep N × order:")
    for n in Ns:
        for o in orders:
            h, hu = run_case(n, o)
            xc = cellcent(n)
            h_an, hu_an = dalembert(xc, T_END)
            # Subtract h₀ to compute error in the WAVE component
            l1_eta = float(np.mean(np.abs((h - H0) - (h_an - H0))))
            l1_hu  = float(np.mean(np.abs(hu - hu_an)))
            results[(n, o)] = (l1_eta, l1_hu)
            print(f"  N={n:5d}  order={o}  L1 η={l1_eta:.4e}  L1 hu={l1_hu:.4e}")

    print("\n  N pair    rate(η, O1)  rate(η, O2)   rate(hu, O1)  rate(hu, O2)")
    for i in range(1, len(Ns)):
        a, b = Ns[i - 1], Ns[i]
        ret1 = np.log2(results[(a, 1)][0] / results[(b, 1)][0])
        ret2 = np.log2(results[(a, 2)][0] / results[(b, 2)][0])
        rhu1 = np.log2(results[(a, 1)][1] / results[(b, 1)][1])
        rhu2 = np.log2(results[(a, 2)][1] / results[(b, 2)][1])
        print(f"  {a:4d}→{b:4d}    {ret1:.3f}        {ret2:.3f}         "
              f"{rhu1:.3f}         {rhu2:.3f}")

    print("\n[3/3] Plot…")
    h_cells = [(X_MAX - X_MIN) / n for n in Ns]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, var, idx in [(axes[0], "η = h − h₀", 0), (axes[1], "hu", 1)]:
        for o, marker, label, col in [(1, "o", "order 1", "C0"),
                                       (2, "s", "order 2", "C3")]:
            errs = [results[(n, o)][idx] for n in Ns]
            ax.loglog(h_cells, errs, marker=marker, label=label,
                      color=col, lw=1.4, ms=7)
        # Reference slope lines anchored on the order-2 coarsest point.
        e2 = results[(Ns[0], 2)][idx]
        h0 = h_cells[0]
        ax.loglog([h0, h_cells[-1]],
                  [e2, e2 * (h_cells[-1]/h0) ** 1],
                  "k:", lw=0.8, alpha=0.5, label="slope 1")
        ax.loglog([h0, h_cells[-1]],
                  [e2, e2 * (h_cells[-1]/h0) ** 2],
                  "k--", lw=0.8, alpha=0.7, label="slope 2")
        ax.set_xlabel("cell size h [m]")
        ax.set_ylabel(f"L1 error in {var}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_title(f"L1 error in {var}")

    fig.suptitle(
        f"Smooth-IC linear-acoustic convergence  "
        f"(h₀={H0}, Δh={DH}, σ={SIGMA}, t={T_END}; d'Alembert reference)\n"
        f"Smooth flow — no shock — 2nd-order rate should approach 2.",
        fontsize=10,
    )
    fig.tight_layout()
    out = HERE / "smooth_convergence_panel.png"
    fig.savefig(out, dpi=130)
    print(f"  → plot saved to {out}")


if __name__ == "__main__":
    main()
