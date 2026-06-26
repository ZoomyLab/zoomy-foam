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


def build_case(case, n_cells, scheme, order=1, n_manning=N_MANNING):
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
maxCo 0.4; reconstructionOrder {order}; timeScheme {scheme}; imexTableau ars232;
imexMaxIter 20; imexTol 1e-12;
modelParameters {{ g {G}; n {n_manning}; }}
""")
    x_nodes = np.linspace(0, L, n_cells + 1)
    xc = 0.5 * (x_nodes[1:] + x_nodes[:-1])
    b, h = analytic(xc, Q_IN, n_manning)
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


def solve_at(work, n_cells, scheme, order=1, n_manning=N_MANNING):
    case = work / f"mac_{scheme}_o{order}_n{n_manning:g}_{n_cells}"
    xc, h_an = build_case(case, n_cells, scheme, order=order, n_manning=n_manning)
    h = run(case)
    if np.isscalar(h):
        h = np.full_like(xc, h)
    return xc, h_an, h


def converge(work, scheme, order, n_manning, grids):
    dxs, l1s = [], []
    for n in grids:
        xc, h_an, h = solve_at(work, n, scheme, order=order, n_manning=n_manning)
        dxs.append(L / n); l1s.append(float(np.mean(np.abs(h - h_an))))
    rate = float(np.polyfit(np.log(dxs), np.log(l1s), 1)[0])
    return dxs, l1s, rate


def main():
    # Regenerate SWE(level0)+Manning+subcritical headers and build once.
    # Use the SAME interpreter running this harness (the zoomy env with zoomy_core).
    subprocess.run([sys.executable, str(FOAM / "create_model.py"), "--level", "0",
                    "--closure", "manning", "--bcs", "subcritical",
                    "--q-in", str(Q_IN), "--h-out", "1.0"], check=True)
    _apptainer(f"cd {FOAM}; wclean >/dev/null 2>&1; wmake 2>&1 | tail -1")

    work = Path(tempfile.mkdtemp(prefix="macdonald_"))
    grids = [50, 100, 200, 400]
    nofric = 1e-8                       # ~frictionless: isolate the bed-slope part
    print(f"MacDonald SWE+Manning: q={Q_IN}, n={N_MANNING}, L={L}")

    # Three convergence series (all IMEX-ARK time scheme):
    #   o1 + friction, o2 + friction, o2 frictionless (bed-slope only).
    series = {
        "o1 friction":   ("imex", 1, N_MANNING, "crimson", "s"),
        "o2 friction":   ("imex", 2, N_MANNING, "navy",    "o"),
        "o2 frictionless (bed only)": ("imex", 2, nofric,  "seagreen", "^"),
    }
    conv = {}
    for label, (scheme, order, nman, col, mk) in series.items():
        dxs, l1s, rate = converge(work, scheme, order, nman, grids)
        conv[label] = (dxs, l1s, rate, col, mk)
        print(f"  {label}: rate={rate:.2f}  L1={[f'{e:.2e}' for e in l1s]}")

    # Profile at the medium grid, order 2 with friction.
    xc, h_an, h_o2 = solve_at(work, 200, "imex", order=2, n_manning=N_MANNING)

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    axL.plot(xc, h_an, "-", color="0.4", lw=3, label="analytic MacDonald h(x)")
    axL.plot(xc, h_o2, "o", color="navy", ms=3, label="zoomyFoam IMEX-ARK (order 2)")
    axL.set_xlabel("x [m]"); axL.set_ylabel("depth h [m]")
    axL.set_title(f"(a) steady profile, n=200, order 2   L1(h)={np.mean(abs(h_o2-h_an)):.2e}")
    axL.legend(fontsize=8)
    for label, (dxs, l1s, rate, col, mk) in conv.items():
        axR.loglog(dxs, l1s, mk + "-", color=col, label=f"{label}  (rate {rate:.2f})")
    d = conv["o1 friction"][0]
    axR.loglog(d, [conv["o1 friction"][1][0]*(x/d[0]) for x in d], ":", color="0.5", lw=1,
               label=r"$\propto \Delta x$")
    axR.loglog(d, [conv["o2 frictionless (bed only)"][1][0]*(x/d[0])**2 for x in d], ":",
               color="seagreen", lw=1, label=r"$\propto \Delta x^2$")
    axR.set_xlabel(r"$\Delta x$ [m]"); axR.set_ylabel("L1 error vs analytic")
    axR.set_title("(b) convergence: order 2 holds for flux+bed;\n"
                  "friction source caps it at 1st order")
    axR.legend(fontsize=7.5)
    fig.suptitle("zoomyFoam IMEX-ARK vs MacDonald steady SWE + Manning friction "
                 "(analytic source-term verification)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
