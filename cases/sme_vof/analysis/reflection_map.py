#!/usr/bin/env python
"""Quantify the interface reflection on the SME side of a twoway run.

After the dam-break front crosses the interface (t ~ 0.35) the SME domain
sits on the star plateau until physical signals return from the VOF wall
(t ~ 2).  In that window any left-moving disturbance emanating from x=0 is
coupling-interface reflection.  The map is column_plots.plot_xt with the
'plateau' transform (in-house high-pass detrending, see its docstring);
reported is max |D| in the interior probe region.

Usage: reflection_map.py RUN1 [RUN2 ...]  ->  reflection_maps.png (cwd)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from zoomy_core.postprocessing import style
style.use()
from zoomy_core.postprocessing.column_plots import (
    ColumnField, read_of_states, plot_xt)

SWE_N = 120
T_WIN = (0.45, 1.90)          # plateau window (front passed, no physical return)
X_PROBE = (-0.55, -0.05)      # interior, away from both boundaries
XC = np.linspace(-0.6, 0.0, SWE_N + 1)
XC = 0.5 * (XC[:-1] + XC[1:])

runs = sys.argv[1:]
fig, axs = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 3.6),
                        squeeze=False)
print(f"{'run':38s}  max|D| (m)   [plateau t in {T_WIN}, x in {X_PROBE}]")
for a, run in zip(axs[0], runs):
    T, Q = read_of_states(Path(run) / "swe_case", 2, SWE_N)
    m = (T >= T_WIN[0]) & (T <= T_WIN[1])
    cf = ColumnField(T[m], XC, np.array([0.5]),
                     {"h": Q[m, 1, :, None]}, Path(run).name)
    # same detrending as plot_xt(transform='plateau') — for the printed metric
    D = Q[m, 1, :] - Q[m, 1, :].mean(axis=1, keepdims=True)
    D = D - D.mean(axis=0, keepdims=True)
    xm = (XC >= X_PROBE[0]) & (XC <= X_PROBE[1])
    amp = np.abs(D[:, xm]).max()
    print(f"{Path(run).name:38s}  {amp:.3e}")
    im = plot_xt(a, cf, "h", transform="plateau", cmap=style.CMAP_DIVERGING,
                 vmax=3e-3)
    a.set_title(f"{cf.label}\nmax|D|={amp:.2e} m")
    plt.colorbar(im, ax=a, label="h - plateau (m)")
fig.tight_layout()
out = Path.cwd() / "reflection_maps.png"
fig.savefig(out)
print(f"-> {out}")
