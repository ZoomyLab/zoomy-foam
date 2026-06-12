#!/usr/bin/env python
"""Friction SME(0)↔SME(1) gif — case-specific composer on column_plots tools.

Top: free surface of the joined L0|L1 pair over BOTH monolithic references
(SME(0) and SME(1)).  Bottom: the friction-excited first moment q_1(x) —
joined L1 half vs mono SME(1); SME(0) cannot represent it.

Usage: build_gif_friction.py [results_dir] [out.gif]
"""
import sys
from pathlib import Path
import numpy as np

from zoomy_core.model.models import SME
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Coupled, FromModel)
from zoomy_core.postprocessing import style
style.use("screen")
from zoomy_core.postprocessing.column_plots import (
    read_zoomyfoam, read_of_states, mark_stations, animate)

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "sme0_sme1_friction.gif"
X_MID, N = 25.0, 200
NH = N // 2
X1 = np.linspace(0, 25, NH + 1); X1 = 0.5 * (X1[:-1] + X1[1:])
X2 = X1 + 25.0
XM = np.linspace(0, 50, N + 1); XM = 0.5 * (XM[:-1] + XM[1:])

bcs = BoundaryConditions([FromModel(tag="outer", definition="wall"),
                          Coupled(tag="coupled", mesh_name="interface")])
sm0 = SME(level=0, boundary_conditions=bcs).system_model
sm1 = SME(level=1, boundary_conditions=bcs).system_model

cf1 = read_zoomyfoam(SRC / "part1",   sm0, NH, X1, label="SME(0) participant")
cf2 = read_zoomyfoam(SRC / "part2",   sm1, NH, X2, label="SME(1) participant")
cm0 = read_zoomyfoam(SRC / "mono",    sm0, N,  XM, label="mono SME(0)")
cm1 = read_zoomyfoam(SRC / "mono_l1", sm1, N,  XM, label="mono SME(1)")

# q_1 is raw state (Q3 of SME(1)) — not part of the canonical column fields
Tm1, Qm1 = read_of_states(SRC / "mono_l1", 4, N)
T2,  Q2  = read_of_states(SRC / "part2",   4, NH)


def draw(fig, t):
    axH, axQ = fig.subplots(2, 1, sharex=True, height_ratios=[2, 1])
    for cf, gray in ((cm0, "0.65"), (cm1, "0.3")):
        i = cf.at(t)
        axH.plot(cf.x, cf.fields["h"][i, :, 0], color=gray, ls="--",
                 marker="", label=cf.label)
    for k, cf in enumerate((cf1, cf2)):
        i = cf.at(t)
        axH.plot(cf.x, cf.fields["h"][i, :, 0], color=style.CYCLE[k],
                 marker=style.MARKERS[k], markevery=style.MARKEVERY,
                 label=cf.label)
    mark_stations(axH, [X_MID], [style.COLORS["interface"]])
    axH.set_ylim(0.05, 0.55)
    axH.set_ylabel("h [m]")
    axH.set_title("SME(0) ↔ SME(1), bottom friction λ_s=0.5 ν=1e-3   "
                  f"t = {t:4.2f} s")

    im1 = int(np.argmin(np.abs(Tm1 - t)))
    i2 = int(np.argmin(np.abs(T2 - t)))
    axQ.plot(XM, Qm1[im1, 3], color="0.3", ls="--", marker="")
    axQ.plot(X2, Q2[i2, 3], color=style.CYCLE[1], marker=style.MARKERS[1],
             markevery=style.MARKEVERY)
    mark_stations(axQ, [X_MID], [style.COLORS["interface"]])
    axQ.set_xlim(0, 50)
    axQ.set_xlabel("x [m]")
    axQ.set_ylabel(r"$q_1$")
    axQ.set_title(r"first moment $q_1$ (friction-excited; SME(0) cannot "
                  "represent it)")
    style.figure_legend(fig, extra=[
        ("coupling interface", style.line("interface", ls="--"))],
        ncol=3, reserve=0.16)


times = [t for t in cm0.t
         if min(abs(cf1.t - t).min(), abs(cf2.t - t).min(),
                abs(Tm1 - t).min()) < 0.02]
animate(draw, times, OUT, figsize=(10, 6.5), fps=10)
print(f"GIF: {len(times)} frames -> {OUT}")
