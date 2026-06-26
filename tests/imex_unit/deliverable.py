#!/usr/bin/env python3
"""§6.1 deliverable for the foam additive IMEX-ARK time integrator.

Regenerates ``figures/imex_ark.png`` headlessly from the *actual compiled C++
kernel* (imex_kernel.H): builds the unit-test binary in the OF13 apptainer,
runs its CSV-sweep mode, and plots two panels:

  (left)  per-stage implicit solve — the cell-local Newton tracks exact
          backward-Euler and stays bounded for every K·dt, whereas explicit
          forward-Euler diverges (|1-K·dt|>1) past K·dt = 2.
  (right) IMEX-ARK temporal convergence — error vs dt for ARS232 / ARS343 on
          a non-stiff ODE; slopes ~2 / ~3 confirm a GENUINE additive
          Runge–Kutta (a Lie–Trotter operator split would be only 1st order).

Run:  python3 deliverable.py
"""
from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SIF = Path.home() / "of_build" / "zoomy_openfoam.sif"
CSV = HERE / "imex_sweep.csv"
CONV = HERE / "ark_convergence.csv"
FIG = HERE / "figures" / "imex_ark.png"


def regenerate_csv() -> None:
    """Build + run the kernel sweep inside the apptainer (writes imex_sweep.csv)."""
    script = (
        "source /opt/openfoam13/etc/bashrc 2>/dev/null; "
        f"cd {HERE}; wmake 2>/dev/null; "
        f"$FOAM_USER_APPBIN/test_imex_kernel {CSV}"
    )
    subprocess.run(["apptainer", "exec", str(SIF), "bash", "-lc", script],
                   check=True)


def _read(path):
    rows = list(csv.DictReader(open(path)))
    cols = {k: [float(r[k]) for r in rows] for k in rows[0]}
    return cols


def main() -> int:
    if not CSV.exists() or not CONV.exists() or os.environ.get("REGEN"):
        regenerate_csv()

    s = _read(CSV)
    c = _read(CONV)

    FIG.parent.mkdir(exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # Left: per-stage implicit solve stiff-stability.
    axL.axhspan(-1.0, 1.0, color="0.92", zorder=0, label="bounded |q| ≤ q*")
    axL.plot(s["Kdt"], s["forward_euler"], "--", color="crimson", lw=1.8,
             label="explicit forward-Euler (1 step)")
    axL.plot(s["Kdt"], s["backward_euler"], "-", color="0.4", lw=3.0,
             label="exact backward-Euler")
    axL.plot(s["Kdt"], s["imex"], "o", color="navy", ms=3.5,
             label="per-stage cell-Newton (C++)")
    axL.axvline(2.0, color="crimson", ls=":", lw=1.0)
    axL.text(2.05, -1.9, "explicit unstable →", color="crimson", fontsize=8)
    axL.set_xlabel(r"$K\,\Delta t$  (source stiffness × step)")
    axL.set_ylabel(r"$q^{n+1}/q^{*}$  after one implicit stage")
    axL.set_title("(a) per-stage implicit solve: stiff-stable")
    axL.set_ylim(-3.0, 1.5)
    axL.legend(fontsize=8, loc="lower left")

    # Right: IMEX-ARK temporal convergence (log-log), with reference slopes.
    dt = c["dt"]
    axR.loglog(dt, c["err_ars232"], "o-", color="navy", label="ARS232 (order 2)")
    axR.loglog(dt, c["err_ars343"], "s-", color="seagreen", label="ARS343 (order 3)")
    d0 = dt[0]
    axR.loglog(dt, [c["err_ars232"][0]*(x/d0)**2 for x in dt], ":",
               color="navy", lw=1.0, label=r"$\propto \Delta t^2$")
    axR.loglog(dt, [c["err_ars343"][0]*(x/d0)**3 for x in dt], ":",
               color="seagreen", lw=1.0, label=r"$\propto \Delta t^3$")
    axR.set_xlabel(r"$\Delta t$")
    axR.set_ylabel("error vs exact at $T=1$")
    axR.set_title("(b) IMEX-ARK temporal order\n(coupled additive RK, not a split)")
    axR.legend(fontsize=8, loc="lower right")

    fig.suptitle("foam additive IMEX-ARK  (zoomyFoam, C++ kernel)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print(f"wrote {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
