#!/usr/bin/env python
"""sme_vof gif — thin config over the layer-2 composed figure.
Usage: make_gif.py RUNDIR LEVEL OUT"""
import sys
import numpy as np
from zoomy_core.model.models import SME
from zoomy_core.model.boundary_conditions import (
    BoundaryConditions, Coupled, FromModel)
from zoomy_core.postprocessing import style
style.use("screen")   # gif canvas: scaled-up fonts
from zoomy_core.postprocessing.column_plots import (
    read_columns, read_zoomyfoam, read_vof_raw, fig_coupling, animate)

RUN, LEVEL, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
bcs = BoundaryConditions([FromModel(tag="outer", definition="wall"),
                          Coupled(tag="coupled", mesh_name="interface")])
sm = SME(level=LEVEL, boundary_conditions=bcs).system_model
x = np.linspace(-0.6, 0, 121); x = 0.5 * (x[:-1] + x[1:])

sme = read_zoomyfoam(RUN + "/swe_case", sm, 120, x, K=40, label="SME")
vif = read_columns(RUN + "/vof_case", label="VOF")
raw, vmid = read_vof_raw(RUN + "/vof_case", nx=120, ny=40, lx=1.5, ly=0.4,
                         stations=[0.75], label="VOF")

panels = [("mid SME",   -0.30, [sme],        "tab:blue"),
          ("interface",  0.00, [sme, vif],   "tab:purple"),
          ("mid VOF",    0.75, [vmid],       "tab:green")]

animate(lambda fig, t: fig_coupling(
            fig, t, sme, raw, panels, interface_x=0.0,
            xlim=(-0.6, 1.5), ylim=(0, 0.22), ulim=(-0.6, 0.6),
            title=f"SME({LEVEL}) | VOF   t={t:5.2f}s"),
        list(sme.t), OUT, figsize=(10, 6), fps=8)
print("->", OUT)
