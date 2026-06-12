#!/usr/bin/env python
"""Two-way SME<->VOF dam-break GIF with a velocity-profile row.

Top: joint free surface (SME h lifted | VOF alpha).  Bottom: u(z) profiles at
4 stations — mid SME, SME side of interface, VOF side of interface, mid VOF.
SME profiles from the modal ansatz u(zeta) = (q0 + sum_j q_j P~_j(zeta))/h
(shifted Legendre P~1 = 2z-1, P~2 = 6z^2-6z+1 — exactly the emitted
interpolate_to_3d).  VOF profiles from the U field column (water cells).

Usage: build_twoway_gif_profiles.py RUNDIR OUTNAME LEVEL
"""
import re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

DIR = Path("/mnt/userdrive/Users/home/adam-obbpb5az1dhsjzf/git/Zoomy/thesis/notebooks/coupling/vof_spike")
RUN = Path(sys.argv[1])
OUTNAME = sys.argv[2]
LEVEL = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SWE, VOF = RUN / "swe_case", RUN / "vof_case"

SWE_XMIN, SWE_XMAX, SWE_N = -0.6, 0.0, 120
VOF_LX, VOF_LY, VOF_NX, VOF_NY = 1.5, 0.4, 120, 40
dy = VOF_LY / VOF_NY
dxv = VOF_LX / VOF_NX
ycc = (np.arange(VOF_NY) + 0.5) * dy
swe_xc = np.linspace(SWE_XMIN, SWE_XMAX, SWE_N + 1)
swe_xc = 0.5 * (swe_xc[:-1] + swe_xc[1:])
ZETA = np.linspace(0, 1, 41)


def basis(j, z):
    if j == 0: return np.ones_like(z)
    if j == 1: return 2*z - 1
    if j == 2: return 6*z**2 - 6*z + 1
    raise ValueError(j)


def u_profile_sme(qcell, z):
    h = qcell[1]
    if h <= 1e-12: return np.zeros_like(z)
    u = np.zeros_like(z)
    for j in range(LEVEL + 1):
        u += qcell[2 + j] * basis(j, z)
    return u / h


def read_scalar(f, n):
    t = f.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<\w+>\s*\d+\s*\((.*?)\)\s*;", t, re.S)
    if m:
        return np.fromstring(m.group(1).replace("\n", " "), sep=" ")
    m = re.search(r"internalField\s+uniform\s+([-\d.eE]+)", t)
    return np.full(n, float(m.group(1)))


def read_vector_x(f, n):
    t = f.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\d+\s*\((.*?)\)\s*;", t, re.S)
    if m:
        vals = re.findall(r"\(([^)]*)\)", m.group(1))
        return np.array([float(v.split()[0]) for v in vals])
    m = re.search(r"internalField\s+uniform\s+\(([^)]*)\)", t)
    return np.full(n, float(m.group(1).split()[0]))


def times(case, field):
    return {round(float(d.name), 4): d for d in case.iterdir()
            if re.fullmatch(r"\d+(\.\d+)?", d.name) and (d / field).exists()}


swe_items = sorted(times(SWE, "Q1").items())
vof_items = sorted(times(VOF, "alpha.water").items())
n = min(len(swe_items), len(vof_items))
pairs = list(zip(swe_items[:n], vof_items[:n]))
print(f"SWE frames {len(swe_items)}, VOF frames {len(vof_items)}, paired {n}")

NQ = 3 + LEVEL
# bottom row: 3 panels — mid SME | interface (both sides overlaid) | mid VOF
ix_s_mid, ix_s_if = SWE_N // 2, SWE_N - 1
ix_v_if, ix_v_mid = 0, VOF_NX // 2
PANELS = [
    (f"mid SME (x={swe_xc[ix_s_mid]:.2f})",  [("sme", ix_s_mid, "tab:blue")]),
    ("interface (SME blue | VOF green)",      [("sme", ix_s_if,  "tab:blue"),
                                               ("vof", ix_v_if,  "tab:green")]),
    (f"mid VOF (x={(ix_v_mid+0.5)*dxv:.2f})", [("vof", ix_v_mid, "tab:green")]),
]

frames = []
for (t, swe_dir), (tv, vof_dir) in pairs:
    Qs = np.array([read_scalar(swe_dir / f"Q{q}", SWE_N)[:SWE_N] for q in range(NQ)])
    h = Qs[1]
    A_swe = (ycc[:, None] < h[None, :]).astype(float)
    A_vof = read_scalar(vof_dir / "alpha.water", VOF_NX * VOF_NY)[:VOF_NX * VOF_NY].reshape(VOF_NY, VOF_NX)
    Ux = read_vector_x(vof_dir / "U", VOF_NX * VOF_NY).reshape(VOF_NY, VOF_NX)

    fig = plt.figure(figsize=(11, 6.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.45, wspace=0.38)
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(A_swe, origin="lower", extent=[SWE_XMIN, 0, 0, VOF_LY], aspect="auto",
              cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(A_vof, origin="lower", extent=[0, VOF_LX, 0, VOF_LY], aspect="auto",
              cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.axvline(0, color="red", lw=2.5)
    for lab, members in PANELS:
        for side, idx, col in members:
            xs = swe_xc[idx] if side == "sme" else (idx + 0.5) * dxv
            ax.axvline(xs, color=col, lw=0.7, alpha=0.5)
    ax.set_xlim(SWE_XMIN, VOF_LX); ax.set_ylim(0, 0.22)
    ax.set_title(f"TWO-WAY SME({LEVEL}) (zoomyFoam) ─ coupling ─ VOF (incompressibleVoF)   t={t:5.2f}s",
                 fontsize=10.5)
    ax.set_xlabel("x   (SME: x<0  |  VOF: x>0)"); ax.set_ylabel("y")

    for k, (lab, members) in enumerate(PANELS):
        a = fig.add_subplot(gs[1, k])
        for side, idx, col in members:
            if side == "sme":
                qc = Qs[:, idx]
                a.plot(u_profile_sme(qc, ZETA), ZETA, color=col, lw=1.6)
            else:
                wet = A_vof[:, idx] > 0.5
                hcol = (A_vof[:, idx] * dy).sum()
                if hcol > 1e-9:
                    a.plot(Ux[wet, idx], np.clip(ycc[wet]/hcol, 0, 1),
                           color=col, lw=1.6, marker=".", ms=3)
        a.axvline(0, color="gray", lw=0.5, ls=":")
        a.set_title(lab, fontsize=8)
        a.set_xlim(-0.6, 0.6); a.set_ylim(0, 1)
        a.tick_params(labelsize=7)
        if k == 0: a.set_ylabel("ζ = z/h")
        a.set_xlabel("u (m/s)", fontsize=8)

    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)

out = DIR / OUTNAME
imageio.mimsave(out, frames, fps=8, loop=0)
print(f"GIF: {len(frames)} frames -> {out}")
