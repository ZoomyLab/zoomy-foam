#!/usr/bin/env python
"""Reproduce the SWE preCICE self-coupling and plot coupled vs monolithic vs Stoker.

Splits a 1D Stoker wet-wet dam-break domain at its centre into two preCICE-coupled
zoomyFoam participants, also runs a monolithic single-domain reference, and overlays
all three (joined coupling, monolithic reference, Stoker analytic) of h(x) and u(x)
at t_end.

OpenFOAM execution:
  - Native:    OF13 + preCICE on PATH  ->  runs directly.
  - Container: set ZOOMY_OF_SIF=/path/to/zoomy_openfoam.sif  ->  every OF command and
               both participants run via `apptainer exec` (host net -> preCICE sockets,
               bound FS -> shared exchange-directory).

Usage:
    ZOOMY_OF_SIF=~/of_build/zoomy_openfoam.sif \
        python cases/self_coupling/couple_demo.py --t-end 3 --n 400 --order 2

The defaults run a few seconds of physical time on a [0,50] domain so the dam-break
waves propagate well away from the central coupling interface (error has room to grow
while the Stoker similarity solution stays valid — no boundary interaction).
"""
from __future__ import annotations

import argparse, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FOAM_ROOT = HERE.parent.parent                     # library/zoomy_foam
REPO = FOAM_ROOT.parents[1]                         # …/Zoomy
sys.path.insert(0, str(FOAM_ROOT / "tools"))
sys.path.insert(0, str(HERE))
import run as R                                     # noqa: E402  (case writers + config)
import model as coupling_model                      # noqa: E402
from compare_stoker import stoker                   # noqa: E402

BASHRC = "/opt/openfoam13/etc/bashrc"
SIF = os.environ.get("ZOOMY_OF_SIF", "")            # empty -> native OF on PATH


def _wrap(cmd, cwd):
    """Build the argv that runs `cmd` (cwd applied) under OF env, native or container."""
    inner = f"cd {cwd} && source {BASHRC} && {cmd}"
    if SIF:
        return ["apptainer", "exec", "--bind", f"{REPO}:{REPO}", SIF, "bash", "-c", inner]
    return ["bash", "-c", inner]


def cexec(cmd, cwd, timeout=1200):
    return subprocess.run(_wrap(cmd, cwd), check=True, capture_output=True, text=True, timeout=timeout)


def cpopen(cmd, cwd):
    return subprocess.Popen(_wrap(f"unset FOAM_SIGFPE FOAM_SETNAN && {cmd}", cwd))


def run_coupled(x_min, x_max, t_end, n, order, scheme, h_L, h_R, deltaT):
    """Set geometry/scenario on run.py's module globals, then orchestrate."""
    R.X_MIN, R.X_MAX, R.X_MID = x_min, x_max, 0.5 * (x_min + x_max)
    R.SCENARIOS["dam_break"].update(t_end=t_end, h_L=h_L, h_R=h_R)
    # run.py writes deltaT 0.0001 hard; patch its controlDict writer for this demo.
    _orig_cd = R._write_controldict
    def _cd(case_dir, te, od, precice=None):
        _orig_cd(case_dir, te, od, precice)
        p = case_dir / "system" / "controlDict"
        p.write_text(p.read_text().replace("deltaT 0.0001;", f"deltaT {deltaT};"))
    R._write_controldict = _cd

    nh = n // 2
    work = HERE / f"demo_dam_break_{scheme}"
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    cmax = np.sqrt(R.G * h_L)
    window = 0.9 * 0.4 * (x_max - x_min) / n / cmax

    ref = work / "reference"
    R._make_case(ref, x_min, x_max, n, "dam_break", t_end, order,
                 patches={"outer": "(0 4 7 3) (1 2 6 5)"}, precice=None)
    R._write_initial(ref, x_min, x_max, n, "dam_break", ["outer"])
    da = work / "domainA"
    R._make_case(da, x_min, R.X_MID, nh, "dam_break", t_end, order,
                 patches={"outer": "(0 4 7 3)", "coupled": "(1 2 6 5)"},
                 precice=dict(participant="domainA", mesh="MeshA",
                              write=" ".join(R.DATA_A), read=" ".join(R.DATA_B)))
    R._write_initial(da, x_min, R.X_MID, nh, "dam_break", ["outer", "coupled"])
    db = work / "domainB"
    R._make_case(db, R.X_MID, x_max, nh, "dam_break", t_end, order,
                 patches={"coupled": "(0 4 7 3)", "outer": "(1 2 6 5)"},
                 precice=dict(participant="domainB", mesh="MeshB",
                              write=" ".join(R.DATA_B), read=" ".join(R.DATA_A)))
    R._write_initial(db, R.X_MID, x_max, nh, "dam_break", ["coupled", "outer"])
    R._write_precice_config(work / "precice-config.xml", scheme, t_end, window)

    print(f"  meshing + reference (deltaT={deltaT}, window={window:.4g})…", flush=True)
    for c in (ref, da, db):
        cexec(f"blockMesh -case {c.name}", work)
    try:
        cexec("precice-config-validate precice-config.xml", work)
    except Exception:
        print("  (precice-config-validate unavailable — non-fatal)", flush=True)
    cexec("unset FOAM_SIGFPE FOAM_SETNAN && zoomyFoam -case reference > reference/log.zoomyFoam 2>&1", work)

    print("  launching domainA + domainB (this is the few-second coupled run)…", flush=True)
    pa = cpopen("zoomyFoam -case domainA > domainA/log.zoomyFoam 2>&1", work)
    pb = cpopen("zoomyFoam -case domainB > domainB/log.zoomyFoam 2>&1", work)
    ra = pa.wait(timeout=1800); rb = pb.wait(timeout=1800)
    if ra or rb:
        raise RuntimeError(f"coupled run failed (A={ra}, B={rb}); see {work}/*/log.zoomyFoam")

    def fld(case, q, m): return R._read_internal(R._last_time(case) / q, m)
    xc = R.cellcent(x_min, x_max, n)
    h_ref, hu_ref = fld(ref, "Q1", n), fld(ref, "Q2", n)
    h_j = np.concatenate([fld(da, "Q1", nh), fld(db, "Q1", nh)])
    hu_j = np.concatenate([fld(da, "Q2", nh), fld(db, "Q2", nh)])
    h_an, u_an = stoker(xc, t_end, h_L, h_R, R.X_MID, R.G)
    return dict(xc=xc, h_ref=h_ref, hu_ref=hu_ref, h_j=h_j, hu_j=hu_j,
                h_an=h_an, u_an=u_an, X_MID=R.X_MID, t_end=t_end, n=n,
                order=order, scheme=scheme, work=work)


def plot(res, out):
    xc, X_MID = res["xc"], res["X_MID"]
    def u(h, hu): return np.divide(hu, h, out=np.zeros_like(hu), where=h > 1e-9)
    fig, (axH, axU) = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, num_ref, num_j, ana, ylab, ttl in [
        (axH, res["h_ref"], res["h_j"], res["h_an"], "h [m]", "Water depth h"),
        (axU, u(res["h_ref"], res["hu_ref"]), u(res["h_j"], res["hu_j"]), res["u_an"], "u [m/s]", "Velocity u"),
    ]:
        ax.plot(xc, ana, "-", color="0.55", lw=3, label="Stoker analytic")
        ax.plot(xc, num_ref, "k-", lw=1.4, label="monolithic reference")
        ax.plot(xc, num_j, "C3o", ms=3, mfc="none", label="joined coupling (A|B)")
        ax.axvline(X_MID, color="C0", ls=":", lw=1.3, label="coupling interface")
        ax.set(xlabel="x [m]", ylabel=ylab, title=ttl)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    linf = np.max(np.abs(res["h_j"] - res["h_ref"]))
    l1s = np.mean(np.abs(res["h_j"] - res["h_an"]))
    fig.suptitle(f"Stoker wet-wet dam-break, preCICE self-coupling — t={res['t_end']}s, "
                 f"N={res['n']}, O{res['order']}, {res['scheme']}\n"
                 f"joined vs monolithic: Linf(h)={linf:.2e}   joined vs Stoker: L1(h)={l1s:.2e}",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"  Linf(joined-monolithic) h = {linf:.3e};  L1(joined-Stoker) h = {l1s:.3e}")
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-min", type=float, default=0.0)
    ap.add_argument("--x-max", type=float, default=50.0)
    ap.add_argument("--t-end", type=float, default=3.0)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--scheme", default="serial-explicit", choices=R.SCHEMES)
    ap.add_argument("--h-l", type=float, default=0.5)
    ap.add_argument("--h-r", type=float, default=0.01)
    ap.add_argument("--dt", type=float, default=2e-4)
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    if not args.no_build:
        print("[build] emit Model.H from model.py + wmake", flush=True)
        coupling_model.write_headers()
        cexec("wmake", FOAM_ROOT)
    res = run_coupled(args.x_min, args.x_max, args.t_end, args.n, args.order,
                      args.scheme, args.h_l, args.h_r, args.dt)
    plot(res, HERE / "couple_demo.png")


if __name__ == "__main__":
    main()
