#!/usr/bin/env python3
"""Subcritical flow over a bump — VAM (Chorin) vs Bernoulli analytic.

Self-contained deliverable for task 0031: regenerates the Chorin VAM(1,2)
subcritical headers, builds chorinFoam in the OF13 apptainer, runs the standard
subcritical bump (q=4.42 m²/s, downstream h=2 m, bump b=0.2-0.05(x-10)² on
[8,12]), and renders figures/vam_bump.gif (free-surface evolution) +
figures/vam_bump_final.png (steady η and discharge) against the analytic
Bernoulli energy solution  H = h + b + q²/(2 g h²) = const.

⚠ The Chorin driver is order-1 and not yet well-balanced: the steady result is a
faithful QUALITATIVE reproduction (correct subcritical dip, stable — the
subcritical case avoids the REQ-17 transcritical blow-up) but carries ~3% h
error / ~4% discharge deficit vs Bernoulli. A WB source + order-2 are the path
to quantitative agreement.

Run (zoomy env; apptainer + OF13 sif):  python3 vam_bump_verification.py
"""
from __future__ import annotations
import re, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

G, L, Q, HOUT = 9.81, 25.0, 4.42, 2.0
FOAM = Path(__file__).resolve().parent.parent.parent
SIF = Path.home() / "of_build" / "zoomy_openfoam.sif"
FIG = Path(__file__).resolve().parent / "figures"


def bed(x):
    b = np.zeros_like(x); m = (x >= 8) & (x <= 12)
    b[m] = 0.2 - 0.05 * (x[m] - 10) ** 2; return b


def analytic_h(x):
    b = bed(x); c = Q**2 / (2 * G); E0 = HOUT + c / HOUT**2
    h = np.full_like(x, HOUT)
    for _ in range(200):
        h = np.maximum(h - (h + b + c/h**2 - E0) / (1 - 2*c/h**3), 1e-3)
    return h, b


def _ap(script):
    subprocess.run(["apptainer", "exec", str(SIF), "bash", "-lc",
                    "source /opt/openfoam13/etc/bashrc 2>/dev/null; " + script], check=True)


def field(case, name, vals):
    body = (f"uniform {vals}" if np.isscalar(vals) else
            "nonuniform List<scalar>\n%d\n(\n%s\n)" % (len(vals), "\n".join(f"{v:.10g}" for v in vals)))
    (case/"0"/name).write_text(
        f"FoamFile {{ version 2.0; format ascii; class volScalarField; object {name}; }}\n"
        f"dimensions [0 0 0 0 0 0 0]; internalField {body};\n"
        "boundaryField { left { type zeroGradient; } right { type zeroGradient; } fb { type empty; } }\n")


def build_case(case, n, tend, dtw):
    if case.exists(): shutil.rmtree(case)
    (case/"0").mkdir(parents=True); (case/"system").mkdir(); (case/"constant").mkdir()
    (case/"system"/"blockMeshDict").write_text(f"""FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
convertToMeters 1; vertices ( (0 0 0)({L} 0 0)({L} 1 0)(0 1 0)(0 0 1)({L} 0 1)({L} 1 1)(0 1 1) );
blocks ( hex (0 1 2 3 4 5 6 7) ({n} 1 1) simpleGrading (1 1 1) ); edges ();
boundary ( left {{ type patch; faces ((0 4 7 3)); }} right {{ type patch; faces ((1 2 6 5)); }}
  fb {{ type empty; faces ((0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7)); }} ); mergePatchPairs ();
""")
    (case/"system"/"fvSchemes").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n"
        "ddtSchemes { default none; } gradSchemes { default Gauss linear; }\n"
        "divSchemes { default none; } laplacianSchemes { default none; }\n"
        "interpolationSchemes { default linear; } snGradSchemes { default corrected; }\n")
    (case/"system"/"fvSolution").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers {}\n")
    (case/"system"/"controlDict").write_text(f"""FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application chorinFoam; startFrom startTime; startTime 0; stopAt endTime; endTime {tend};
deltaT 0.001; writeControl adjustableRunTime; writeInterval {dtw}; maxCo 0.4; purgeWrite 0;
""")
    xn = np.linspace(0, L, n+1); xc = 0.5*(xn[1:]+xn[:-1])
    field(case, "Q0", bed(xc)); field(case, "Q1", HOUT); field(case, "Q2", Q)
    for nm in ("Q3", "Q4", "Q5", "Q6", "Q7"): field(case, nm, 0.0)
    return xc


def read_times(case):
    out = {}
    for d in case.iterdir():
        if re.fullmatch(r"[0-9.]+", d.name) and (d/"Q1").exists():
            def rd(nm):
                t = (d/nm).read_text(); m = re.search(r"nonuniform[^(]*\(\s*(.*?)\s*\)", t, re.S)
                if m: return np.array([float(v) for v in m.group(1).split()])
                return float(re.search(r"uniform\s+([-\d.eE+]+)", t).group(1))
            out[float(d.name)] = {"h": rd("Q1"), "b": rd("Q0"), "q": rd("Q2")}
    return dict(sorted(out.items()))


def main():
    n, tend, dtw = 80, 30.0, 0.5
    subprocess.run([sys.executable, str(FOAM/"create_model.py"), "--scheme", "chorin",
                    "--level", "1", "--dim", "2", "--bcs", "subcritical",
                    "--q-in", str(Q), "--h-out", str(HOUT)], check=True)
    _ap(f"cd {FOAM}/chorin_app; wclean >/dev/null 2>&1; wmake 2>&1 | tail -1")
    work = Path(tempfile.mkdtemp(prefix="bump_"))
    case = work / "bump"
    xc = build_case(case, n, tend, dtw)
    _ap(f"cd {case}; blockMesh >/dev/null 2>&1 && chorinFoam > run.log 2>&1; echo done")
    ts = read_times(case); times = sorted(ts)
    ha, b = analytic_h(xc); eta_a = ha + b
    geth = lambda t: (np.full_like(xc, ts[t]["h"]) if np.isscalar(ts[t]["h"]) else ts[t]["h"])
    L1 = np.mean(np.abs(geth(times[-1]) - ha))
    print(f"steady L1(h vs Bernoulli) = {L1:.3e}  (n={n}, t={times[-1]:.0f})")

    FIG.mkdir(exist_ok=True)
    # GIF
    fig, ax = plt.subplots(figsize=(8, 4.2))
    def draw(i):
        ax.clear(); t = times[i]; h = geth(t)
        ax.fill_between(xc, 0, b, color="0.6", label="bed b(x)")
        ax.plot(xc, eta_a, "--", color="crimson", lw=2, label="Bernoulli analytic (steady)")
        ax.plot(xc, h+b, "-", color="navy", lw=2, label="VAM-Chorin η=h+b")
        ax.set_xlim(0, L); ax.set_ylim(0, 2.5); ax.set_xlabel("x [m]"); ax.set_ylabel("elevation [m]")
        ax.set_title(f"Subcritical flow over a bump — VAM (Chorin), foam\n"
                     f"t={t:.1f}s   L1(h vs Bernoulli)={np.mean(np.abs(h-ha)):.2e}   q_in={Q}")
        ax.legend(loc="upper right", fontsize=8)
    FuncAnimation(fig, draw, frames=len(times)).save(FIG/"vam_bump.gif", writer=PillowWriter(fps=8))
    # static steady comparison
    h = geth(times[-1]); q = ts[times[-1]]["q"]
    fig2, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.fill_between(xc, 0, b, color="0.6", label="bed"); a1.plot(xc, eta_a, "--", color="crimson", lw=2, label="Bernoulli η")
    a1.plot(xc, h+b, "o-", color="navy", ms=3, label="VAM-Chorin η"); a1.set_ylim(0, 2.5)
    a1.legend(fontsize=8); a1.set_ylabel("η=h+b [m]"); a1.set_title(f"VAM-Chorin steady bump  L1(h)={L1:.2e}")
    a2.plot(xc, q, "o-", color="seagreen", ms=3, label="VAM-Chorin q"); a2.axhline(Q, ls="--", color="0.5", label=f"q_in={Q}")
    a2.legend(fontsize=8); a2.set_xlabel("x [m]"); a2.set_ylabel("discharge q [m²/s]")
    fig2.tight_layout(); fig2.savefig(FIG/"vam_bump_final.png", dpi=120)
    shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {FIG}/vam_bump.gif + vam_bump_final.png")


if __name__ == "__main__":
    main()
