"""Compare the zoomyFoam 1D dam-break run against the Stoker analytical.

Stoker (1957) wet-wet shallow-water Riemann solution: h_L > h_R > 0,
u_L = u_R = 0, flat bed, gravity g.  Closed-form rarefaction (left) +
contact + shock (right).

Reads ``cases/swe_dambreak_1d/<endTime>/Q0`` (depth) for the numerical
side and writes a PNG comparison.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


# ── Stoker analytical ────────────────────────────────────────────────────


def stoker(x: np.ndarray, t: float, h_L: float, h_R: float, x0: float,
           g: float = 9.81):
    """Stoker wet-wet dam-break solution.

    Returns (h, u) at points x, time t > 0.  Assumes u_L = u_R = 0.
    """
    c_L = np.sqrt(g * h_L)
    c_R = np.sqrt(g * h_R)

    # Middle state h_M from the shock+rarefaction matching condition.
    # Shock branch: u_M(h_M) = (h_M - h_R) * sqrt(g/2 * (h_M+h_R)/(h_M*h_R))
    # Rarefaction (left) branch: u_M(h_M) = 2*(c_L - sqrt(g*h_M))
    def _residual(h_M):
        c_M = np.sqrt(g * h_M)
        u_raref = 2.0 * (c_L - c_M)
        u_shock = (h_M - h_R) * np.sqrt(0.5 * g * (h_M + h_R) / (h_M * h_R))
        return u_raref - u_shock

    # h_M lies in (h_R, h_L).
    h_M = brentq(_residual, h_R + 1e-12, h_L - 1e-12, xtol=1e-14)
    c_M = np.sqrt(g * h_M)
    u_M = 2.0 * (c_L - c_M)

    # Wave speeds (in lab frame; u_L = 0 so absolute = relative-to-L)
    s_HL = -c_L                      # left rarefaction head
    s_TL = u_M - c_M                 # left rarefaction tail
    s_S = u_M * h_M / (h_M - h_R)    # shock

    # Allocate
    h = np.empty_like(x, dtype=float)
    u = np.empty_like(x, dtype=float)
    xi = (x - x0) / max(t, 1e-12)

    # Undisturbed upstream: ξ < s_HL
    upstream = xi < s_HL
    h[upstream] = h_L
    u[upstream] = 0.0

    # Rarefaction fan: s_HL ≤ ξ < s_TL
    fan = (xi >= s_HL) & (xi < s_TL)
    u[fan] = 2.0 / 3.0 * (c_L + xi[fan])
    c_fan = (2.0 * c_L - xi[fan]) / 3.0
    h[fan] = c_fan ** 2 / g

    # Middle constant state: s_TL ≤ ξ < s_S
    middle = (xi >= s_TL) & (xi < s_S)
    h[middle] = h_M
    u[middle] = u_M

    # Undisturbed downstream: ξ ≥ s_S
    downstream = xi >= s_S
    h[downstream] = h_R
    u[downstream] = 0.0

    return h, u


# ── Foam I/O ─────────────────────────────────────────────────────────────


def read_internal_field(path: Path) -> np.ndarray:
    """Parse the ``internalField nonuniform List<scalar> N (...)`` block."""
    text = path.read_text()
    m = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(([^)]+)\)",
        text,
        re.DOTALL,
    )
    if not m:
        m = re.search(r"internalField\s+uniform\s+([0-9eE.+\-]+)", text)
        if m:
            return None  # uniform — let caller fill in
        raise ValueError(f"could not parse internalField from {path}")
    n = int(m.group(1))
    vals = np.fromstring(m.group(2), sep="\n")
    assert vals.size == n, f"{path}: expected {n}, got {vals.size}"
    return vals


def cell_centres(x_min: float, x_max: float, n: int) -> np.ndarray:
    edges = np.linspace(x_min, x_max, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case",
        default=str(Path(__file__).resolve().parent.parent
                    / "cases" / "swe_dambreak_1d"),
    )
    ap.add_argument("--time", default="1", help="time directory to read")
    ap.add_argument("--h-L", type=float, default=0.5)
    ap.add_argument("--h-R", type=float, default=0.01)
    ap.add_argument("--x0", type=float, default=5.0)
    ap.add_argument("--x-min", type=float, default=0.0)
    ap.add_argument("--x-max", type=float, default=10.0)
    ap.add_argument("--g", type=float, default=9.81)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent
                    / "stoker_comparison.png"),
    )
    args = ap.parse_args()

    case = Path(args.case)
    t = float(args.time)

    # Numerical fields.  For swe_dambreak_1d (b-in-state):
    #   Q0=b, Q1=h, Q2=hu  →  depth is Q1.
    # For swe_dambreak_b_as_aux_1d (b-as-aux):
    #   Q0=h, Q1=hu       →  depth is Q0.
    # Detect by presence of Q2.
    q2_path = case / args.time / "Q2"
    if q2_path.exists():
        h_num = read_internal_field(case / args.time / "Q1")
        hu_num = read_internal_field(case / args.time / "Q2")
        label = "b in state (Q0=b, Q1=h, Q2=hu)"
    else:
        h_num = read_internal_field(case / args.time / "Q0")
        hu_num = read_internal_field(case / args.time / "Q1")
        label = "b as aux (Q0=h, Q1=hu)"
    u_num = np.where(h_num > 1e-12, hu_num / h_num, 0.0)
    n_cells = h_num.size
    xc = cell_centres(args.x_min, args.x_max, n_cells)

    # Analytical Stoker.
    x_an = np.linspace(args.x_min, args.x_max, 2000)
    h_an, u_an = stoker(x_an, t, args.h_L, args.h_R, args.x0, args.g)

    # Plot.
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax = axes[0]
    ax.plot(x_an, h_an, "k-", lw=1.5, label="Stoker analytical")
    ax.plot(xc, h_num, "C0o", ms=3, alpha=0.7, label="zoomyFoam")
    ax.set_ylabel("depth $h$ [m]")
    ax.set_title(
        f"Stoker wet-wet dam-break  "
        f"(h_L={args.h_L}, h_R={args.h_R}, x0={args.x0}, g={args.g}, "
        f"t={t})  —  {label}"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(x_an, u_an, "k-", lw=1.5, label="Stoker analytical")
    ax.plot(xc, u_num, "C0o", ms=3, alpha=0.7, label="zoomyFoam")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("velocity $u$ [m/s]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    # Summary L1 error vs analytical, interpolated onto cell centres.
    h_an_at_cells = np.interp(xc, x_an, h_an)
    u_an_at_cells = np.interp(xc, x_an, u_an)
    print(f"L1 err  h: {np.mean(np.abs(h_num - h_an_at_cells)):.4e}")
    print(f"L1 err  u: {np.mean(np.abs(u_num - u_an_at_cells)):.4e}")


if __name__ == "__main__":
    main()
