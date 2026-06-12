#!/usr/bin/env python
"""sme0_sme1 gif — thin config over column_plots.fig_reduced_coupling.

SME(0)↔SME(1) inter-level coupling: free surface of the joined pair over
the monolithic SME(0) reference and the Stoker analytic, profiles at mid
part1 / interface / mid part2 (the L0 profile is depth-uniform by
construction; L1 carries the linear moment).

Usage: build_gif.py [results_dir] [out.gif]   (results_dir holds mono/part1/part2)
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
    read_zoomyfoam, fig_reduced_coupling, animate)
from zoomy_core.postprocessing.analytic import stoker

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "sme0_sme1.gif"
X_MID, N = 25.0, 200
NH = N // 2
H_L, H_R, G = 0.5, 0.1, 9.81
X1 = np.linspace(0, 25, NH + 1); X1 = 0.5 * (X1[:-1] + X1[1:])
X2 = X1 + 25.0
XM = np.linspace(0, 50, N + 1); XM = 0.5 * (XM[:-1] + XM[1:])

bcs = BoundaryConditions([FromModel(tag="outer", definition="wall"),
                          Coupled(tag="coupled", mesh_name="interface")])
sm0 = SME(level=0, boundary_conditions=bcs).system_model
sm1 = SME(level=1, boundary_conditions=bcs).system_model

cf1 = read_zoomyfoam(SRC / "part1", sm0, NH, X1, K=40, label="SME(0) participant")
cf2 = read_zoomyfoam(SRC / "part2", sm1, NH, X2, K=40, label="SME(1) participant")
cfm = read_zoomyfoam(SRC / "mono",  sm0, N,  XM, K=40, label="monolithic SME(0)")

panels = [("mid part1", 12.5, [cf1, cfm], "tab:blue"),
          ("interface", 25.0, [cf1, cf2, cfm], "tab:purple"),
          ("mid part2", 37.5, [cf2, cfm], "tab:green")]


def draw(fig, t):
    axs = fig_reduced_coupling(
        fig, t, [(cf1, None), (cf2, None)], [(cfm, None)],
        panels=panels, interface_x=X_MID, ylim=(0.05, 0.55),
        ulim=(-0.25, 1.0),
        title=f"SME(0) ↔ SME(1) inter-level coupling   t = {t:4.2f} s")
    h_an, _ = stoker(XM, max(t, 1e-9), H_L, H_R, X_MID, G)
    axs["water"].plot(XM, h_an, color="0.75", lw=3, marker="",
                      zorder=0, label="Stoker analytic")


times = [t for t in cfm.t
         if min(abs(cf1.t - t).min(), abs(cf2.t - t).min()) < 0.02]
animate(draw, times, OUT, figsize=(10, 6.5), fps=10)
print(f"GIF: {len(times)} frames -> {OUT}")
