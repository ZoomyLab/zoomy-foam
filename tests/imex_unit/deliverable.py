#!/usr/bin/env python3
"""§6.1 deliverable for the foam IMEX implicit-source kernel.

Regenerates ``figures/imex_stiff_stability.png`` headlessly from the *actual
compiled C++ kernel* (imex_kernel.H): builds the unit-test binary in the OF13
apptainer, runs its CSV-sweep mode over a stiff linear source S = -K q, and
plots the IMEX cell-Newton result against exact backward-Euler and one
explicit forward-Euler step as a function of K·dt.

Headline: the implicit step tracks backward-Euler and stays bounded for every
K·dt, whereas explicit forward-Euler diverges (|1-K·dt|>1) past K·dt = 2 — the
stability the IMEX split buys for a stiff source.

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
FIG = HERE / "figures" / "imex_stiff_stability.png"


def regenerate_csv() -> None:
    """Build + run the kernel sweep inside the apptainer (writes imex_sweep.csv)."""
    script = (
        "source /opt/openfoam13/etc/bashrc 2>/dev/null; "
        f"cd {HERE}; wmake 2>/dev/null; "
        f"$FOAM_USER_APPBIN/test_imex_kernel {CSV}"
    )
    subprocess.run(["apptainer", "exec", str(SIF), "bash", "-lc", script],
                   check=True)


def main() -> int:
    if not CSV.exists() or os.environ.get("REGEN"):
        regenerate_csv()

    kdt, imex, be, fe = [], [], [], []
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            kdt.append(float(r["Kdt"]))
            imex.append(float(r["imex"]))
            be.append(float(r["backward_euler"]))
            fe.append(float(r["forward_euler"]))

    FIG.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhspan(-1.0, 1.0, color="0.92", zorder=0, label="bounded |q| ≤ q*")
    ax.plot(kdt, fe, "--", color="crimson", lw=1.8,
            label="explicit forward-Euler (1 step)")
    ax.plot(kdt, be, "-", color="0.4", lw=3.0, label="exact backward-Euler")
    ax.plot(kdt, imex, "o", color="navy", ms=3.5,
            label="IMEX cell-Newton (C++ kernel)")
    ax.axvline(2.0, color="crimson", ls=":", lw=1.0)
    ax.text(2.05, -1.9, "explicit unstable →", color="crimson", fontsize=8)
    ax.set_xlabel(r"$K\,\Delta t$  (source stiffness × step)")
    ax.set_ylabel(r"$q^{n+1}/q^{*}$  after one source step")
    ax.set_title("foam IMEX implicit source: stiff-stability\n"
                 r"$S=-Kq$,  $q^{n+1}=q^*+\Delta t\,S(q^{n+1})$")
    ax.set_ylim(-3.0, 1.5)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print(f"wrote {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
