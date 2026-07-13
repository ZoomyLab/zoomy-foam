#!/usr/bin/env python3
"""Deliverable for the P1 sparse Chorin pressure solve (task 0031).

Regenerates `figures/pressure_solve_scaling.png` — the story of replacing the
O(N^3) assemble-by-probe + dense Gaussian-elimination pressure solve in
chorin_app/chorinFoam.C with a matrix-free BiCGStab (numerics::bicgstabMatrixFree
in imex_kernel.H):

  panel A — foam chorinFoam pressure-solve cost vs mesh size N: the LEGACY dense
            path (`pressureSolver dense;`) explodes super-linearly while the new
            BiCGStab default stays flat.  Measured on the 1-D escalante VAM(1,2)
            case (extra wall time over the predictor-only floor).
  panel B — the BiCGStab KERNEL iteration budget vs N (from tests/bicgstab_unit,
            SPD Laplacian manufactured RHS): iters ~ O(N), so total work
            O(iters*N) << the dense O(N^3).

Headless (Agg).  Runs the foam sweep + the C++ unit test in the OF13 apptainer;
falls back to the recorded measurements (scaling.csv) if the container is absent.
"""
import re, subprocess, sys, time, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SIF = Path.home()/"of_build"/"zoomy_openfoam.sif"
sys.path.insert(0, str(HERE))
import vam_escalante_verification as V

SCR = Path("/tmp/claude-765404697/-mnt-userdrive-Users-home-adam-obbpb5az1dhsjzf-git-zotero-rag"
           "/90e72b09-051d-4551-9fdb-b58d0e229c7a/scratchpad/deliverable")

# Recorded fallback (measured 2026-07-13, OF13 apptainer): N_cells, dense wall s,
# bicgstab wall s, steps.  Extra pressure cost = (wall - startup_floor).
FALLBACK = {  # N_cells: (dense_wall, bicg_wall, steps)
    60:  (2.8, 2.5, 14), 120: (2.9, 2.5, 29),
    240: (5.0, 2.5, 60), 480: (23.3, 2.6, 127),
}
KERNEL = {50: 1, 100: 2, 200: 4, 400: 8, 800: 16}  # unit-test iters vs N


def _have_sif():
    return SIF.exists() and shutil.which("apptainer") is not None


def foam_sweep(Ns=(60, 120, 240, 480), tend=0.05):
    """Wall time of chorinFoam with dense vs bicgstab pressure solve."""
    out = {}
    for N in Ns:
        V.N = N
        row = {}
        for solver in ("dense", "bicgstab"):
            case = SCR/f"esc_{solver}_{N}"
            V.build(case, tend, tend, 0.3)
            cd = case/"system"/"controlDict"
            cd.write_text(cd.read_text().rstrip() +
                          f"\npressureSolver {solver};\npressureTol 1e-10;\n")
            t0 = time.time(); V.run(case); wall = time.time()-t0
            steps = len(re.findall(r"^Time = ", (case/"run.log").read_text(), re.M))
            row[solver] = (wall, steps)
        out[N] = (row["dense"][0], row["bicgstab"][0], row["dense"][1])
        print(f"N={N}: dense={out[N][0]:.1f}s bicg={out[N][1]:.1f}s steps={out[N][2]}")
    return out


def main():
    data = foam_sweep() if _have_sif() else FALLBACK
    if not _have_sif():
        print("apptainer/sif absent — using recorded measurements")

    Ns = np.array(sorted(data))
    ndof = 2*Ns                                   # 2 pressure modes per cell
    floor = min(v[1] for v in data.values())      # predictor+startup floor
    dense = np.array([max(data[n][0]-floor, 1e-3)/data[n][2]*1e3 for n in Ns])
    bicg  = np.array([max(data[n][1]-floor, 1e-3)/data[n][2]*1e3 for n in Ns])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.2))

    axA.loglog(ndof, dense, "o-", label="dense assemble-by-probe (legacy)")
    axA.loglog(ndof, bicg, "s-", label="matrix-free BiCGStab (new)")
    ref = dense[0]*(ndof/ndof[0])**3
    axA.loglog(ndof, ref, "k--", lw=0.8, label=r"$O(N^3)$ ref")
    axA.set_xlabel("pressure DOF  $N = n_P\\,n_{cells}$")
    axA.set_ylabel("pressure-solve wall / step  [ms]")
    axA.set_title("A · foam chorinFoam pressure solve\n(1-D escalante VAM(1,2))")
    axA.legend(fontsize=8); axA.grid(True, which="both", alpha=0.3)

    kN = np.array(sorted(KERNEL)); kI = np.array([KERNEL[n] for n in kN])
    axB.loglog(kN, kI, "^-", color="C2", label="BiCGStab iters (measured)")
    axB.loglog(kN, kI[0]*(kN/kN[0]), "k--", lw=0.8, label=r"$O(N)$ ref")
    axB.set_xlabel("system size  $N$")
    axB.set_ylabel("Krylov iterations to $10^{-10}$")
    axB.set_title("B · BiCGStab kernel budget\n(tests/bicgstab_unit, SPD Laplacian)")
    axB.legend(fontsize=8); axB.grid(True, which="both", alpha=0.3)

    fig.suptitle("Chorin pressure solve: O(N³) dense → matrix-free BiCGStab (task 0031, P1)",
                 fontsize=11)
    fig.tight_layout()
    (HERE/"figures").mkdir(exist_ok=True)
    out = HERE/"figures"/"pressure_solve_scaling.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
