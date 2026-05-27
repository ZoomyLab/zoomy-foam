#!/usr/bin/env python
"""Single-file config + driver for the *lake-at-rest with bump* test.

This is the "Python notebook" pattern for a zoomy_foam case:

  1. Build the model + numerics symbolically.
  2. Emit Model.H + NumericsKernels.H + UpdateAuxVariables.H directly
     into the zoomy_foam solver tree (so a subsequent ``wmake`` picks
     them up).
  3. Generate the OpenFOAM case files (blockMesh + IC + boundaries).
  4. Drive ``wmake → blockMesh → zoomyFoam``.
  5. Plot the water surface ``η = h + b`` over time — well-balanced
     schemes preserve the initial flat surface to machine epsilon;
     non-WB schemes show spurious currents.

Model:  SWE 1D + bed b in state + Manning friction (n_m=0 here for the
        pure WB test).
Numerics: ``PositiveNonconservativeRusanov`` — Audusse-style
          hydrostatic reconstruction + NCP path-integral.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from sympy import Matrix, Abs

# ── locate zoomy_foam root + add tests/cases to sys.path ───────────────
HERE = Path(__file__).resolve().parent           # cases/lake_at_rest_bump_1d/
FOAM_ROOT = HERE.parent.parent                   # library/zoomy_foam/
sys.path.insert(0, str(FOAM_ROOT / "tests"))

from zoomy_core.misc.misc import ZArray
from zoomy_core.model import boundary_conditions as BC
from zoomy_core.model import initial_conditions as IC
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.transformation.to_openfoam import (
    FoamNumericsPrinter,
    FoamSystemModelPrinter,
    FoamUpdateAuxPrinter,
)


# ── 1. Model: SWE 1D + bed b in state, Manning optional ────────────────


class SWEBedFriction1D(StructuredDerivativeModel):
    """1D shallow water + bed + Manning friction (n_m=0 disables it).

    Splits the momentum flux into advective (in ``flux``) and hydrostatic
    pressure (in ``hydrostatic_pressure``) — required so that
    PositiveRusanov's Audusse-style well-balancing can extract P at the
    reconstructed states to compute the bed-step correction.
    """

    dimension = 1
    variables = ["b", "h", "hu"]
    parameters = {"g": (9.81, "positive"), "n_m": (0.0, "non-negative")}

    def flux(self):
        h, hu = self.Q.h, self.Q.hu
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[1, 0] = hu                  # continuity
        F[2, 0] = hu * hu / h         # advective momentum (no pressure)
        return ZArray(F)

    def hydrostatic_pressure(self):
        h = self.Q.h
        g = self.params.g
        P = Matrix.zeros(self.n_variables, self.dimension)
        P[2, 0] = 0.5 * g * h * h     # 1/2 g h^2 — WB-aware via HR
        return ZArray(P)

    def nonconservative_matrix(self):
        h = self.Q.h
        g = self.params.g
        B = [[[0] * self.dimension for _ in range(self.n_variables)]
             for _ in range(self.n_variables)]
        B[2][0][0] = g * h             # g·h · ∂_x b
        return ZArray(B)

    def source(self):
        h, hu = self.Q.h, self.Q.hu
        g, n_m = self.params.g, self.params.n_m
        S = Matrix.zeros(self.n_variables, 1)
        S[2, 0] = -g * n_m * n_m * hu * Abs(hu) / h ** (7 / 3)
        return ZArray(S)

    def reconstruction_variables(self):
        """Primitives that the limiter sees — chosen so lake-at-rest
        (h+b = const, u = 0) maps to W = [b, const, 0] which is
        preserved by any standard cellLimited scheme."""
        b, h, hu = self.Q.b, self.Q.h, self.Q.hu
        return ZArray([b, h + b, hu / h])


def build_system_model():
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="wall"),
        BC.Extrapolation(tag="inflow"),
        BC.Extrapolation(tag="outflow"),
    ])
    ic = IC.UserFunction(function=lambda x: np.zeros(3))
    model = SWEBedFriction1D(
        boundary_conditions=bcs, initial_conditions=ic,
    )
    return SystemModel.from_model(model)


# ── 2. Emit headers into the zoomy_foam tree ───────────────────────────


def write_headers():
    sm = build_system_model()
    numerics = PositiveNonconservativeRusanov(model=sm)

    FoamSystemModelPrinter.write_code(
        sm, FOAM_ROOT / "Model.H", analytical_eigenvalues=True
    )
    FoamNumericsPrinter.write_code(numerics, FOAM_ROOT / "NumericsKernels.H")
    FoamUpdateAuxPrinter.write_code(sm, FOAM_ROOT / "UpdateAuxVariables.H")
    print(f"  → headers written to {FOAM_ROOT}")


# ── 3. Mesh + IC: lake at rest with Gaussian bump ──────────────────────

# Lake parameters
ETA0    = 0.5        # constant water surface elevation
B_PEAK  = 0.2        # bump amplitude (peak < eta0 → stays wet)
B_CENT  = 12.5
B_SIGMA = 1.0
H_MIN   = 1e-3       # numerical floor (we stay well above)

X_MIN, X_MAX = 0.0, 25.0
N_CELLS = 200


def bed(x):
    return B_PEAK * np.exp(-((x - B_CENT) / B_SIGMA) ** 2)


def cell_centres():
    edges = np.linspace(X_MIN, X_MAX, N_CELLS + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _write_foam_field(path: Path, name: str, values: np.ndarray):
    """Write a volScalarField with non-uniform internalField."""
    n = values.size
    vals = "\n".join(f"{v:.14e}" for v in values)
    path.write_text(f"""FoamFile
{{
    format      ascii;
    class       volScalarField;
    object      {name};
}}

dimensions      [0 0 0 0 0 0 0];
internalField   nonuniform List<scalar>
{n}
(
{vals}
)
;

boundaryField
{{
    inflow        {{ type zeroGradient; }}
    outflow       {{ type zeroGradient; }}
    sides         {{ type empty; }}
    topAndBottom  {{ type empty; }}
}}
""")


def write_initial_fields():
    xc = cell_centres()
    b_field = bed(xc)
    h_field = np.maximum(ETA0 - b_field, H_MIN)
    hu_field = np.zeros_like(xc)

    zero_dir = HERE / "0"
    zero_dir.mkdir(exist_ok=True)
    _write_foam_field(zero_dir / "Q0", "Q0", b_field)
    _write_foam_field(zero_dir / "Q1", "Q1", h_field)
    _write_foam_field(zero_dir / "Q2", "Q2", hu_field)
    print(f"  → IC fields written (max bump = {b_field.max():.3f}, "
          f"η = {ETA0})")


# ── 4. Drive: wmake + blockMesh + zoomyFoam ────────────────────────────


def _run(cmd, cwd, log=None):
    """Run cmd in cwd, optionally tee stdout to log file."""
    print(f"  $ ({cwd}) {' '.join(cmd)}")
    if log:
        with open(log, "w") as f:
            res = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    else:
        res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise SystemExit(f"command failed (rc={res.returncode}); see {log}")


def driver():
    # Clean prior run.
    for d in HERE.glob("[0-9]*"):
        if d.is_dir():
            import shutil
            shutil.rmtree(d)
    for d in (HERE / "constant").glob("polyMesh"):
        import shutil
        shutil.rmtree(d)
    for f in HERE.glob("log.*"):
        f.unlink()

    print("[1/4] Build solver (wmake)…")
    bashrc = "/opt/openfoam13/etc/bashrc"
    _run(["bash", "-c", f"source {bashrc} && wmake"], FOAM_ROOT,
         log=HERE / "log.wmake")

    print("[2/4] Mesh (blockMesh)…")
    _run(["bash", "-c", f"source {bashrc} && blockMesh"], HERE,
         log=HERE / "log.blockMesh")

    print("[3/4] Write IC fields directly from Python…")
    write_initial_fields()

    print("[4/4] Run (zoomyFoam)…")
    env_unset = "unset FOAM_SIGFPE FOAM_SETNAN"
    _run(["bash", "-c",
          f"source {bashrc} && {env_unset} && zoomyFoam"], HERE,
         log=HERE / "log.zoomyFoam")
    print("  → done.")


# ── 5. Plot water surface over time ────────────────────────────────────


def _read_internal(path: Path) -> np.ndarray:
    text = path.read_text()
    m = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(([^)]+)\)",
        text, re.DOTALL,
    )
    if m:
        return np.fromstring(m.group(2), sep="\n")
    m = re.search(r"internalField\s+uniform\s+([0-9eE.+\-]+)", text)
    if m:
        return np.full(N_CELLS, float(m.group(1)))
    raise ValueError(f"could not parse {path}")


def plot_surface():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xc = cell_centres()
    b_init = bed(xc)

    times = sorted(
        (float(d.name), d) for d in HERE.iterdir()
        if d.is_dir() and re.fullmatch(r"\d+(?:\.\d+)?", d.name)
        and (d / "Q1").exists()
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Water surface η = h + b vs the analytical flat ETA0.
    ax = axes[0]
    ax.fill_between(xc, 0, b_init, color="#b08054", alpha=0.5, label="bed b")
    ax.axhline(ETA0, color="k", ls="--", lw=1.0,
               label=f"analytical η = {ETA0}")
    cmap = plt.get_cmap("viridis")
    for k, (t, d) in enumerate(times):
        b = _read_internal(d / "Q0")
        h = _read_internal(d / "Q1")
        eta = h + b
        ax.plot(xc, eta, "-", color=cmap(k / max(1, len(times) - 1)),
                lw=1.2, label=f"η at t={t:.2f}")
    ax.set_ylabel("elevation")
    ax.set_title(
        "Lake at rest with Gaussian bump  —  "
        "PositiveNonconservativeRusanov (HR + NCP)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Velocity max over time — should be 0 for a true WB scheme.
    ax = axes[1]
    for k, (t, d) in enumerate(times):
        h = _read_internal(d / "Q1")
        hu = _read_internal(d / "Q2")
        u = np.where(h > 1e-12, hu / h, 0.0)
        ax.plot(xc, u, "-", color=cmap(k / max(1, len(times) - 1)),
                lw=1.0, label=f"u at t={t:.2f}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("velocity u [m/s]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = HERE / "lake_at_rest_bump.png"
    fig.savefig(out, dpi=130)
    print(f"  → plot saved to {out}")
    return out


# ── main ───────────────────────────────────────────────────────────────


def main():
    print("=== lake-at-rest with bump · single-file config ===")
    print("[0/5] Emit C++ headers from Model + Numerics…")
    write_headers()
    driver()
    print("[5/5] Plot…")
    out = plot_surface()
    # Final max-|u| over the last time step — should be tiny if WB holds.
    last = sorted(
        (float(d.name), d) for d in HERE.iterdir()
        if d.is_dir() and re.fullmatch(r"\d+(?:\.\d+)?", d.name)
        and (d / "Q1").exists()
    )[-1][1]
    h = _read_internal(last / "Q1")
    hu = _read_internal(last / "Q2")
    u = np.where(h > 1e-12, hu / h, 0.0)
    print(f"  max |u| at final time = {np.max(np.abs(u)):.3e}")
    print(f"  (well-balanced scheme should give ~ machine epsilon)")


if __name__ == "__main__":
    main()
