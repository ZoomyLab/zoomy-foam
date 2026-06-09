#!/usr/bin/env python
"""Plain-SWE smooth convergence — isolates the 2nd-order machinery.

NO bed, NO NCP, NO HR: state [h, hu], pressure in the conservative
flux, plain Rusanov, primitive reconstruction [h, hu/h].  This is the
textbook MUSCL-Rusanov SWE — should hit 2nd-order rate on smooth data.

If THIS reaches rate ≈ 2, the rate-1.4 seen with the bed/NCP/HR
formulation points at that layering.  If this also caps at ~1.4, the
issue is in the core reconstruction / SSP-RK2 wiring.

Same smooth-IC d'Alembert setup as smooth_acoustic_convergence.
"""

from __future__ import annotations

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

HERE = Path(__file__).resolve().parent
FOAM_ROOT = HERE.parent.parent

from zoomy_core.misc.misc import ZArray
from zoomy_core.model import boundary_conditions as BC
from zoomy_core.model import initial_conditions as IC
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.fvm.riemann_solvers import Rusanov, HLLC
from zoomy_core.transformation.to_openfoam import (
    FoamNumericsPrinter,
    FoamSystemModelPrinter,
    FoamUpdateAuxPrinter,
)


# ── 1. Plain SWE 1D — pressure in flux, no bed ─────────────────────────


class SWE1D(StructuredDerivativeModel):
    dimension = 1
    variables = ["h", "hu"]
    parameters = {"g": (9.81, "positive")}

    def flux(self):
        h, hu = self.Q.h, self.Q.hu
        g = self.params.g
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[0, 0] = hu
        F[1, 0] = hu * hu / h + 0.5 * g * h * h   # pressure IN flux
        return ZArray(F)

    def source(self):
        return ZArray.zeros(self.n_variables)

    def reconstruction_variables(self):
        h, hu = self.Q.h, self.Q.hu
        return ZArray([h, hu / h])   # primitive [h, u]


def build_system_model():
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="inflow"),
        BC.Extrapolation(tag="outflow"),
    ])
    ic = IC.UserFunction(function=lambda x: np.zeros(2))
    model = SWE1D(boundary_conditions=bcs, initial_conditions=ic)
    return SystemModel.from_model(model)


# HLLC was tested and gives bit-identical results to Rusanov for the 1D
# symmetric acoustic wave (the HLLC contact wave is the transverse shear,
# irrelevant in 1D) — so the sub-2 rate is NOT a Rusanov-dissipation issue.
RIEMANN = Rusanov   # vs HLLC


def write_headers():
    sm = build_system_model()
    numerics = RIEMANN(model=sm)
    FoamSystemModelPrinter.write_code(
        sm, FOAM_ROOT / "Model.H", analytical_eigenvalues=True
    )
    FoamNumericsPrinter.write_code(numerics, FOAM_ROOT / "NumericsKernels.H")
    FoamUpdateAuxPrinter.write_code(sm, FOAM_ROOT / "UpdateAuxVariables.H")


# ── 2. Smooth-IC d'Alembert reference ──────────────────────────────────

X_MIN, X_MAX = 0.0, 20.0
H0, DH, SIGMA, X0, T_END, G = 1.0, 1.0e-4, 1.0, 10.0, 0.5, 9.81


def gauss(x):
    return DH * np.exp(-((x - X0) / SIGMA) ** 2)


def dalembert(x, t):
    c = np.sqrt(G * H0)
    eta = 0.5 * (gauss(x - c * t) + gauss(x + c * t))
    u = 0.5 * (c / H0) * (gauss(x - c * t) - gauss(x + c * t))
    h = H0 + eta
    return h, h * u


def cellcent(n):
    edges = np.linspace(X_MIN, X_MAX, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


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
    h  = H0 + gauss(xc)
    hu = np.zeros(n)
    z = case_dir / "0"
    z.mkdir(exist_ok=True)
    _write_foam_field(z / "Q0", "Q0", h)
    _write_foam_field(z / "Q1", "Q1", hu)


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
    return np.full(n, float(m.group(1)))


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
        and (d / "Q0").exists()
    )
    last = times[-1][1]
    h  = _read_internal(last / "Q0", n)
    hu = _read_internal(last / "Q1", n)
    return h, hu


def _restrict(fine, n_coarse):
    """Average a fine field (size k*n_coarse) down to n_coarse cells."""
    k = fine.size // n_coarse
    return fine.reshape(n_coarse, k).mean(axis=1)


def main():
    print("[1/3] Build solver + emit headers (plain SWE, Rusanov)…")
    write_headers()
    _run(f"source {BASHRC} && wmake", FOAM_ROOT)

    Ns = [50, 100, 200, 400, 800]
    N_REF = 3200
    results = {}

    print("[2/3] Sweep N × order (vs d'Alembert AND self-convergence):")
    # Fine reference at order 2 (same scheme — removes linear-vs-nonlinear bias).
    h_ref, hu_ref = run_case(N_REF, 2)

    for n in Ns:
        for o in (1, 2):
            h, hu = run_case(n, o)
            xc = cellcent(n)
            # d'Alembert (linear) reference
            h_an, hu_an = dalembert(xc, T_END)
            l1_eta_an = float(np.mean(np.abs((h - H0) - (h_an - H0))))
            # self-convergence vs fine numerical reference (restricted)
            h_ref_c  = _restrict(h_ref,  n)
            hu_ref_c = _restrict(hu_ref, n)
            l1_eta_sc = float(np.mean(np.abs(h  - h_ref_c)))
            l1_hu_sc  = float(np.mean(np.abs(hu - hu_ref_c)))
            results[(n, o)] = (l1_eta_an, l1_eta_sc, l1_hu_sc)
            print(f"  N={n:5d} O={o}  L1η(d'Alembert)={l1_eta_an:.3e}  "
                  f"L1η(self)={l1_eta_sc:.3e}  L1hu(self)={l1_hu_sc:.3e}")

    print("\n  rates (self-convergence, vs N=3200 reference):")
    print("  N pair     η  O1     η  O2     hu O1     hu O2")
    for i in range(1, len(Ns)):
        a, b = Ns[i - 1], Ns[i]
        re1 = np.log2(results[(a,1)][1] / results[(b,1)][1])
        re2 = np.log2(results[(a,2)][1] / results[(b,2)][1])
        rh1 = np.log2(results[(a,1)][2] / results[(b,1)][2])
        rh2 = np.log2(results[(a,2)][2] / results[(b,2)][2])
        print(f"  {a:4d}→{b:4d}   {re1:.3f}    {re2:.3f}    {rh1:.3f}    {rh2:.3f}")


if __name__ == "__main__":
    main()
