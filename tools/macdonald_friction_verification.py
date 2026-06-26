#!/usr/bin/env python3
"""Analytic SWE-with-source verification of the zoomyFoam IMEX-ARK time scheme.

Uses a MacDonald-type steady solution (the SWASHES friction family): for a
constant discharge q and a smooth depth profile h(x), the steady 1-D SWE
momentum balance fixes the bed b(x) that makes (h(x), q) an EXACT steady
solution under bed slope + Manning friction:

    b'(x) = (q^2/(g h^3) - 1) h'(x) - n^2 q|q| / h^(10/3)

zoomyFoam is initialised with that exact state, given discharge-in / depth-out
BCs (a subcritical steady state is otherwise not anchored — transmissive ends
drain the discharge), and evolved to steady state.  The Manning friction is the
stiff source the IMEX-ARK integrates implicitly, so this exercises the full
scheme on the real equations, not a 0-D ODE.  We report the L1 error of h(x)
against the analytic solution and its mesh-convergence order.

This regenerates the SWE+Manning+subcritical headers, builds zoomyFoam in the
OF13 apptainer, runs the convergence sweep, and writes
``tests/imex_unit/figures/macdonald_friction.png``.

Run (zoomy env, apptainer + OF13 sif available):
    python3 tools/macdonald_friction_verification.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.81
FOAM = Path(__file__).resolve().parent.parent
SIF = Path.home() / "of_build" / "zoomy_openfoam.sif"
FIG = FOAM / "tests" / "imex_unit" / "figures" / "macdonald_friction.png"
Q_IN, N_MANNING, L, T_END = 1.0, 0.04, 100.0, 150.0


def _apptainer(script: str) -> None:
    subprocess.run(["apptainer", "exec", str(SIF), "bash", "-lc",
                    "source /opt/openfoam13/etc/bashrc 2>/dev/null; " + script],
                   check=True)


def analytic(x, q, n):
    """Smooth depth bump + the bed b(x) from the steady momentum balance."""
    h = 1.0 + 0.2 * np.exp(-((x - 0.5 * L) / (0.1 * L)) ** 2)
    hp = np.gradient(h, x)
    Sf = n**2 * q * abs(q) / h ** (10.0 / 3.0)
    bp = (q**2 / (G * h**3) - 1.0) * hp - Sf
    b = np.concatenate([[0.0], np.cumsum(0.5 * (bp[1:] + bp[:-1]) * np.diff(x))])
    return b - b[-1], h


def _field(case, name, vals):
    body = "\n".join(f"{v:.10g}" for v in vals)
    (case / "0" / name).write_text(
        f"FoamFile {{ version 2.0; format ascii; class volScalarField; object {name}; }}\n"
        f"dimensions [0 0 0 0 0 0 0];\ninternalField nonuniform List<scalar>\n"
        f"{len(vals)}\n(\n{body}\n);\nboundaryField {{\n"
        "  left { type zeroGradient; } right { type zeroGradient; }\n"
        "  frontAndBack { type empty; } topAndBottom { type empty; }\n}\n")


def build_case(case, n_cells, scheme):
    if case.exists():
        shutil.rmtree(case)
    (case / "0").mkdir(parents=True); (case / "system").mkdir(); (case / "constant").mkdir()
    (case / "system" / "blockMeshDict").write_text(f"""FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1;
vertices ( (0 0 0)({L} 0 0)({L} 1 0)(0 1 0)(0 0 1)({L} 0 1)({L} 1 1)(0 1 1) );
blocks ( hex (0 1 2 3 4 5 6 7) ({n_cells} 1 1) simpleGrading (1 1 1) );
edges (); boundary
(
  left  {{ type patch; faces ( (0 4 7 3) ); }}
  right {{ type patch; faces ( (1 2 6 5) ); }}
  frontAndBack {{ type empty; faces ( (0 1 5 4) (3 7 6 2) ); }}
  topAndBottom {{ type empty; faces ( (0 3 2 1) (4 5 6 7) ); }}
);
mergePatchPairs ();
""")
    (case / "system" / "fvSchemes").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n"
        "ddtSchemes { default none; } gradSchemes { default Gauss linear; }\n"
        "divSchemes { default none; } laplacianSchemes { default none; }\n"
        "interpolationSchemes { default linear; } snGradSchemes { default corrected; }\n")
    (case / "system" / "fvSolution").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers {}\n")
    (case / "system" / "controlDict").write_text(f"""FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application zoomyFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {T_END}; deltaT 0.01;
writeControl adjustableRunTime; writeInterval {T_END}; purgeWrite 0;
maxCo 0.4; reconstructionOrder 1; timeScheme {scheme}; imexTableau ars232;
imexMaxIter 20; imexTol 1e-12;
modelParameters {{ g {G}; n {N_MANNING}; }}
""")
    x_nodes = np.linspace(0, L, n_cells + 1)
    xc = 0.5 * (x_nodes[1:] + x_nodes[:-1])
    b, h = analytic(xc, Q_IN, N_MANNING)
    _field(case, "Q0", b); _field(case, "Q1", h); _field(case, "Q2", np.full_like(xc, Q_IN))
    return xc, h


def _read(path):
    t = path.read_text()
    m = re.search(r"internalField\s+nonuniform[^(]*\(\s*(.*?)\s*\)", t, re.S)
    if m:
        return np.array([float(v) for v in m.group(1).split()])
    return float(re.search(r"internalField\s+uniform\s+([-\d.eE+]+)", t).group(1))


def run(case):
    _apptainer(f"cd {case}; blockMesh >/dev/null 2>&1 && zoomyFoam > run.log 2>&1; echo done")
    last = sorted([d for d in case.iterdir() if re.fullmatch(r"[0-9.]+", d.name) and d.name != "0"],
                  key=lambda d: float(d.name))[-1]
    return _read(last / "Q1")


def solve_at(work, n_cells, scheme, xc_ref=None):
    case = work / f"mac_{scheme}_{n_cells}"
    xc, h_an = build_case(case, n_cells, scheme)
    h = run(case)
    if np.isscalar(h):
        h = np.full_like(xc, h)
    return xc, h_an, h


def main():
    # Regenerate SWE(level0)+Manning+subcritical headers and build once.
    # Use the SAME interpreter running this harness (the zoomy env with zoomy_core).
    subprocess.run([sys.executable, str(FOAM / "create_model.py"), "--level", "0",
                    "--closure", "manning", "--bcs", "subcritical",
                    "--q-in", str(Q_IN), "--h-out", "1.0"], check=True)
    _apptainer(f"cd {FOAM}; wclean >/dev/null 2>&1; wmake 2>&1 | tail -1")

    work = Path(tempfile.mkdtemp(prefix="macdonald_"))
    grids = [50, 100, 200, 400]
    dxs, l1s = [], []
    print(f"MacDonald SWE+Manning: q={Q_IN}, n={N_MANNING}, L={L}")
    for n in grids:
        xc, h_an, h = solve_at(work, n, "imex")
        l1 = float(np.mean(np.abs(h - h_an)))
        dxs.append(L / n); l1s.append(l1)
        print(f"  imex n={n:4d}: L1(h)={l1:.3e}")
    rate = float(np.polyfit(np.log(dxs), np.log(l1s), 1)[0])
    print(f"  observed L1 order = {rate:.2f}")

    xc, h_an, h_imex = solve_at(work, 200, "imex")
    _, _, h_exp = solve_at(work, 200, "explicit")

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    axL.plot(xc, h_an, "-", color="0.4", lw=3, label="analytic MacDonald h(x)")
    axL.plot(xc, h_imex, "o", color="navy", ms=3, label="zoomyFoam IMEX-ARK")
    axL.plot(xc, h_exp, "+", color="crimson", ms=4, label="zoomyFoam explicit")
    axL.set_xlabel("x [m]"); axL.set_ylabel("depth h [m]")
    axL.set_title(f"(a) steady profile, n=200   L1(h)={np.mean(abs(h_imex-h_an)):.2e}")
    axL.legend(fontsize=8)
    axR.loglog(dxs, l1s, "o-", color="navy", label="IMEX-ARK L1(h)")
    axR.loglog(dxs, [l1s[0] * (d / dxs[0]) for d in dxs], ":", color="0.5",
               label=r"$\propto \Delta x$ (order 1)")
    axR.set_xlabel(r"$\Delta x$ [m]"); axR.set_ylabel("L1 error vs analytic")
    axR.set_title(f"(b) mesh convergence — order {rate:.2f}"); axR.legend(fontsize=8)
    fig.suptitle("zoomyFoam IMEX-ARK vs MacDonald steady SWE + Manning friction "
                 "(analytic source-term verification)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
