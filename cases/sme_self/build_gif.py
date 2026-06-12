#!/usr/bin/env python
"""sme_self gif — thin config over column_plots.fig_reduced_coupling.

Coupled pair (part1+part2) vs monolithic; velocity profiles at mid part1 /
interface (both sides overlaid) / mid part2.  The u(zeta) lift comes from
the model's own interpolate_to_3d (no re-implementation).

Usage: build_gif.py LEVEL [SUFFIX]   (reads part1[_SUFFIX]/part2[_SUFFIX]/mono[_SUFFIX])
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

HERE = Path(__file__).resolve().parent
LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SUF = ("_" + sys.argv[2]) if len(sys.argv) > 2 else ""
NH, N = 100, 200
X1 = np.linspace(0, 25, NH + 1); X1 = 0.5 * (X1[:-1] + X1[1:])
X2 = X1 + 25.0
XM = np.linspace(0, 50, N + 1); XM = 0.5 * (XM[:-1] + XM[1:])

bcs = BoundaryConditions([FromModel(tag="outer", definition="wall"),
                          Coupled(tag="coupled", mesh_name="interface")])
sm = SME(level=LEVEL, boundary_conditions=bcs).system_model

cf1 = read_zoomyfoam(HERE / f"part1{SUF}", sm, NH, X1, K=40, label="part1 (coupled)")
cf2 = read_zoomyfoam(HERE / f"part2{SUF}", sm, NH, X2, K=40, label="part2 (coupled)")
cfm = read_zoomyfoam(HERE / f"mono{SUF}",  sm, N,  XM, K=40, label="monolithic")

panels = [("mid part1", 12.5, [cf1, cfm], "tab:blue"),
          ("interface", 25.0, [cf1, cf2, cfm], "tab:purple"),
          ("mid part2", 37.5, [cf2, cfm], "tab:green")]

# write grid is denser than needed for the gif: frame every 0.02 s
times = [t for t in cfm.t if abs(t / 0.02 - round(t / 0.02)) < 1e-9]

out = HERE / f"sme_self_L{LEVEL}.gif"
animate(lambda fig, t: fig_reduced_coupling(
            fig, t, [(cf1, None), (cf2, None)], [(cfm, None)],
            panels=panels, interface_x=25.0,
            ylim=(0.05, 0.55), ulim=(-0.25, 1.0),
            title=f"SME({LEVEL}) | SME({LEVEL})   t = {t:5.2f} s"),
        times, out, figsize=(10, 6.5), fps=8)
print(f"GIF: {len(times)} frames -> {out}")
