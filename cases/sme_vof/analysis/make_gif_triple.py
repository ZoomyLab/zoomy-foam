#!/usr/bin/env python
"""SME | VOF | SME triple-coupling gif — composer on column_plots tools.
Usage: make_gif_triple.py RUNDIR LEVEL OUT"""
import sys
import numpy as np
from zoomy_core.model.models import SME
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Coupled, FromModel)
from zoomy_core.postprocessing import style
style.use("screen")
from zoomy_core.postprocessing.column_plots import (
    read_zoomyfoam, read_vof_raw, plot_water_columns, plot_water_vof,
    plot_profiles, mark_stations, frame_color, animate)

RUN, LEVEL, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
bcs = BoundaryConditions([FromModel(tag="outer", definition="wall"),
                          Coupled(tag="coupled", mesh_name="interface")])
sm = SME(level=LEVEL, boundary_conditions=bcs).system_model
x1 = np.linspace(-0.6, 0, 121); x1 = 0.5 * (x1[:-1] + x1[1:])
x2 = np.linspace(1.5, 2.1, 121); x2 = 0.5 * (x2[:-1] + x2[1:])

sme1 = read_zoomyfoam(RUN + "/swe_case",  sm, 120, x1, K=40, label="SME")
sme2 = read_zoomyfoam(RUN + "/swe2_case", sm, 120, x2, K=40, label="SME")
raw, vmid = read_vof_raw(RUN + "/vof_case", nx=120, ny=40, lx=1.5, ly=0.4,
                         stations=[0.75], label="VOF")

panels = [("mid SME",  -0.30, [sme1], "tab:blue"),
          ("intf L",    0.00, [sme1, vmid], "tab:purple"),
          ("mid VOF",   0.75, [vmid], "tab:green"),
          ("intf R",    1.50, [sme2, vmid], "tab:orange"),
          ("mid SME2",  1.80, [sme2], "tab:red")]


def draw(fig, t):
    gs = fig.add_gridspec(2, len(panels), height_ratios=[1.2, 1.0],
                          hspace=0.45, wspace=0.45)
    ax = fig.add_subplot(gs[0, :])
    plot_water_vof(ax, raw, t, color=style.COLORS["water"])
    plot_water_columns(ax, sme1, t, style="fill", color=style.COLORS["water"])
    plot_water_columns(ax, sme2, t, style="fill", color=style.COLORS["water"])
    for xi in (0.0, 1.5):
        ax.axvline(xi, color=style.COLORS["interface"], lw=1.6)
    mark_stations(ax, [s for _, s, _, _ in panels],
                  [c for _, _, _, c in panels],
                  labels=[n for n, _, _, _ in panels])
    ax.set_xlim(-0.6, 2.1); ax.set_ylim(0, 0.22)
    ax.set_title(f"SME({LEVEL}) | VOF | SME({LEVEL})   t = {t:5.2f} s")
    for k, (name, xq, src, color) in enumerate(panels):
        a = fig.add_subplot(gs[1, k])
        plot_profiles(a, src, xq, t,
                      colors=[style.COLORS["reduced"],
                              style.COLORS["resolved"]][:len(src)])
        a.set_title(name)
        a.set_xlim(-0.6, 0.6)
        frame_color(a, color)
    style.figure_legend(fig, extra=[
        ("interface", style.line("interface")),
        ("water", style.line("water", lw=5)),
    ])


animate(draw, list(sme1.t), OUT, figsize=(13, 6.5), fps=8)
print("->", OUT)
