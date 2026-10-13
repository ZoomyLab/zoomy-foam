"""Foam driver of the symbolic solvers: prints the C++ solver, builds it with wmake in the OpenFOAM
apptainer, runs the case.  Sits at the end of Model -> SystemModel -> NumericalSystemModel -> C++
template -> foam runtime (`symsolver/rt.H`).
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import param
from scipy.spatial import cKDTree

from zoomy_core.mesh.geometry_xp import build_topology, compute_geometry
from zoomy_core.misc.options import Option, OptionParam
from zoomy_core.symsolver.backend_cxx import RUNTIME_HEADER, CxxBackend, CxxChorinBackend
from zoomy_core.symsolver.settings import LinearSolver
from zoomy_foam._pipeline import (FOAM_ROOT, _BINCACHE, _apptainer,
                                  _mesh_geometry,
                                  _write_grid_system)
from zoomy_foam.symkernels import write_kernels

APP = FOAM_ROOT / "symsolver"
# gradients, limiters (rt.H gradient / minmod / ...), the pressure preconditioner
SCHEMES = """FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes { default none; }
divSchemes { default none; }
gradSchemes
{
    default leastSquares;
    limited(minmod) cellLimited leastSquares 1;
    limited(barth_jespersen) cellLimited leastSquares 1;
    limited(venkatakrishnan) cellLimited<Venkatakrishnan> leastSquares 1;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
# preCICE the binary links: a prefix whose version matches the Python peers
# (conda-forge stops at 3.4.0, the container ships 3.4.1; preCICE refuses to
# couple different versions).  Unset or missing: the container's own.
PRECICE = Path(os.environ.get("ZOOMY_PRECICE_PREFIX",
                              Path.home() / "of_build" / "precice-3.4.0"))
# hypre for linear_solver=LinearSolver.hypre: the static CPU libHYPRE.a of
# containers/zoomy_amrex/build_hypre_cuda.sh (variant cpu, built in the AMReX
# image).  The OF13 image has the same Ubuntu 24.04, gcc 13.3 and OpenMPI 4.1.6,
# so symFoam links it unchanged (not the image's own hypre 2.28).
HYPRE_VERSION = "3.2.0"
HYPRE_CACHE = Path(os.environ.get("ZOOMY_HYPRE_CACHE", Path.home() / ".cache" / "zoomy"))
HYPRE_SCRIPT = FOAM_ROOT.parent.parent / "containers" / "zoomy_amrex" / "build_hypre_cuda.sh"


class Decomposition(Option):
    """decomposePar's method for an MPI run."""
    #: graph partition
    scotch = "scotch"
    #: nprocs slabs along x
    simple = "simple"


class FoamDriver(param.Parameterized):
    """Foam runtime of a C++ solver backend (`zoomy_core.symsolver.backend_cxx`): build tree,
    case, launch, read-back.

    References:
        Weller et al. 1998, https://doi.org/10.1063/1.168744
    """
    nprocs = param.Integer(default=1, bounds=(1, None),
                           doc="MPI ranks: decomposePar, mpirun -parallel.")
    decomposition = OptionParam(
        Decomposition, default=Decomposition.scotch,
        doc="decomposePar method: 'scotch' (graph partition), 'simple' (nprocs "
            "slabs along x).")

    def _control(self):
        """Extra controlDict entries of this solver (none by default)."""
        return ""

    def _hypre(self):
        """True if the solver links hypre (False by default)."""
        return False

    def build(self, nsm, dimension, topo=None):
        """Print the solver, write the build tree and wmake it (cached by a hash of the sources).

        Args:
            nsm: the NumericalSystemModel (or the tuple of systems of a split template)
            dimension: spatial dimension of the case
            topo: mesh topology whose boundary-closure classes the kernels carry; None (precompile):
                none
        Returns:
            Path of the `symFoam` binary.
        """
        self.source = self._print(nsm, dimension)
        files = {"Solver.H": self.source,
                 "symFoam.C": (APP / "symFoam.C").read_text(),
                 "rt.H": (APP / "rt.H").read_text(),
                 "UserFunctions.H": (FOAM_ROOT / "UserFunctions.H").read_text(),
                 RUNTIME_HEADER.name: RUNTIME_HEADER.read_text()}
        hypre = hypre_prefix() if self._hypre() else None
        h = hashlib.sha256(_options(hypre, "BUILD").encode())   # the build flags, then the sources
        for k in sorted(files):
            h.update(k.encode() + files[k].encode())
        _BINCACHE.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=_BINCACHE) as d:   # per build: runs may be concurrent
            tmp = Path(d) / "Kernels.H"
            write_kernels(self._systems(nsm), self.source, tmp, topo)
            files["Kernels.H"] = tmp.read_text()
        h.update(files["Kernels.H"].encode())
        build = _BINCACHE / f"sym_{h.hexdigest()[:16]}"
        binary = build / "symFoam"
        if binary.exists():
            return binary
        (build / "Make").mkdir(parents=True, exist_ok=True)
        for k, v in files.items():
            (build / k).write_text(v)
        (build / "Make" / "files").write_text(f"symFoam.C\n\nEXE = {binary}\n")
        (build / "Make" / "options").write_text(_options(hypre, build))
        r = _apptainer(f"cd {build}; wmake > wmake.log 2>&1; rc=$?; "
                       f"tail -40 wmake.log; [ $rc -eq 0 ]",
                       binds=[build] + ([hypre] if hypre else []),
                       capture_output=True, text=True)
        if r.returncode != 0 or not binary.exists():
            raise RuntimeError(f"symFoam wmake failed ({build}):\n{r.stdout}\n{r.stderr}")
        return binary

    def run(self, mesh, model, Q0, workdir):
        """Write the case (mesh, fields, stencils), launch `symFoam` (`mpirun -parallel` for
        `nprocs > 1`) and read the result back.

        Args:
            mesh: zoomy_core mesh
            Q0: initial state `(n_state, n_cells)` in the mesh order
            workdir: case directory (recreated)
        Returns:
            `(time, Q)` with Q in the mesh order; the written snapshots are kept in
                `self.snapshots`.
        """
        nsm = self._prepare(mesh, model)
        if self.couple and (self.restart_from is not None or self.dt_checkpoint is not None):
            raise NotImplementedError("restart / checkpoints with preCICE coupling")
        restart = (None if self.restart_from is None
                   else restart_state(self.restart_from, self.nprocs))
        from zoomy_core.mesh import ensure_lsq_mesh
        mesh = ensure_lsq_mesh(mesh, nsm)        # its LSQ stencil: the derivatives'
        structured = _structured(mesh)
        cyclic = _cyclic(nsm)
        if structured:
            lo, hi, n, order, dim = _mesh_geometry(mesh)
            names = _side_names(mesh, dim)
        else:                        # gmshToFoam on the mesh extruded by one layer
            dim, names = 2, list(mesh.boundary_conditions_sorted_names)
        topo = build_topology(mesh, nsm=nsm,
                              systems=[sm for _, sm in self._systems(nsm)[1:]])
        binary = self.build(nsm, dim, topo)
        case = Path(workdir)
        staged = None
        if restart is not None and case.resolve() in (restart[0], *restart[0].parents):
            staged = Path(tempfile.mkdtemp(dir=case.parent))   # out of the case recreated here
            restart = _copy_state(restart, staged, [None] * self.nprocs)
        if case.exists():
            shutil.rmtree(case)
        for d in ("0", "system", "constant"):
            (case / d).mkdir(parents=True)
        ckpt = None if self.checkpoint is None else Path(self.checkpoint).resolve()
        start = (f"startTime {float(self.time_start)!r}; zoomyStart \"0\";\n" if restart is None
                 else f"startTime {restart[2]!r}; zoomyStart \"{restart[1]}\"; "
                      "zoomyRestart true;\n")
        (case / "system" / "controlDict").write_text(
            "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
            f"application symFoam; startFrom startTime; stopAt endTime;\n{start}"
            f"endTime {1e30 if self.time_end is None else self.time_end!r}; deltaT 1; writeControl timeStep; "
            f"writeInterval 1; writeFormat ascii; writePrecision 17; spaceDimension {dim};\n"
            + ("" if ckpt is None else f"zoomyCheckpoint \"{ckpt}\";\n")
            + self._control() + _precice_dict(self.couple))
        if structured:
            _write_grid_system(case, lo, hi, n, dim, names, cyclic)
            mesher = "blockMesh > log.blockMesh 2>&1"
        else:
            _write_grid_system(case, [0, 1], [1, 1], [1, 1], 2, names)   # fvSolution
            _write_extruded_msh(mesh, case / "mesh.msh", cyclic)
            _apptainer(f"cd {case}; gmshToFoam mesh.msh > log.gmshToFoam 2>&1 && foamPostProcess "
                       "-func writeCellCentres -time 0 > log.cellCentres 2>&1",
                       binds=[case.parent], check=True)
            bnd = case / "constant" / "polyMesh" / "boundary"
            text = re.sub(r"(frontAndBack\s*\{\s*type\s+)\w+", r"\1empty", bnd.read_text())
            for nm, partner in cyclic.items():     # the seam pairs: translational cyclics
                text = re.sub(rf"(\n\s*{nm}\s*\{{\s*type\s+)\w+;",
                              rf"\1cyclic; neighbourPatch {partner}; transformType "
                              rf"translational; separation {_separation(mesh, nm, partner)};",
                              text)
            bnd.write_text(text)
            C = _read_vectors(case / "0" / "C")
            xc = np.asarray(mesh.cell_centers)[:2, :int(mesh.n_inner_cells)].T
            dist, order = cKDTree(xc).query(C[:, :2])
            assert dist.max() < 1e-8 * np.ptp(xc), "gmshToFoam cells do not match the mesh"
            for f in (case / "0").glob("C*"):     # C, Ccx, ...: not ours to decompose
                f.unlink()
            mesher = "true"
        (case / "system" / "fvSchemes").write_text(SCHEMES)
        Q0 = np.asarray(Q0)[:, order]
        patches = {nm: "cyclic" if nm in cyclic else "zeroGradient" for nm in names}
        if dim < 3:
            patches["frontAndBack"] = "empty"
        if dim == 1:
            patches["topAndBottom"] = "empty"
        for i, q in enumerate(Q0):
            _write_field(case, f"Q{i}", q, patches)
        run = f"{binary} > log.symFoam 2>&1"
        if self.nprocs > 1:
            (case / "system" / "decomposeParDict").write_text(
                "FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }\n"
                f"numberOfSubdomains {self.nprocs};\nmethod {self.decomposition};\n"
                f"simpleCoeffs {{ n ({self.nprocs} 1 1); }}\n")
            mesher += " && decomposePar > log.decomposePar 2>&1"
            run = (f"mpirun -np {self.nprocs} --bind-to none {binary} -parallel "
                   "> log.symFoam 2>&1")
        _apptainer(f"cd {case}; {mesher}", binds=[case.parent], check=True)
        parts = ([case] if self.nprocs == 1 else
                 [case / f"processor{k}" for k in range(self.nprocs)])
        cells = [None] if self.nprocs == 1 else [
            _read_labels(d / "constant" / "polyMesh" / "cellProcAddressing") for d in parts]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        if restart is not None:
            _copy_state(restart, case, parts, cells)
        if staged is not None:
            shutil.rmtree(staged)
        if ckpt is not None:            # the decomposition, checked by a restart from there
            for k, a in enumerate(cells):
                if a is not None:
                    (ckpt / f"processor{k}").mkdir(parents=True, exist_ok=True)
                    shutil.copy(parts[k] / "constant" / "polyMesh" / "cellProcAddressing",
                                ckpt / f"processor{k}" / "cellProcAddressing")
        _write_stencils(mesh, topo, order, inv, parts, cells)
        if self._hypre():
            _write_colours(topo, int(mesh.n_inner_cells), order, parts, cells)
        r = _apptainer(f"cd {case}; {run}",
                       binds=[case.parent, binary.parent] + ([] if ckpt is None else [ckpt]),
                       capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"symFoam failed, see {case}/log.symFoam:\n"
                               + (case / "log.symFoam").read_text()[-3000:])

        def read(name):
            """Read `out/<name>` of every part into the mesh cell order."""
            time, Q = None, np.empty((Q0.shape[0], len(order)))
            for d, c in zip(parts, cells):
                time, q = _read_dat(d / "out" / name)
                Q[:, slice(None) if c is None else c] = q
            return time, Q[:, inv]
        snaps = sorted((parts[0] / "out").glob("[0-9]*.dat"), key=lambda f: int(f.stem))
        self.snapshots = [read(f.name) for f in snaps]
        return read("final.dat")


class SymbolicHyperbolicSolverFoam(FoamDriver, CxxBackend):
    """The explicit hyperbolic template on foam: `time, Q = solver.run(mesh, nsm, Q0, workdir)`."""


class SymbolicChorinSolverFoam(FoamDriver, CxxChorinBackend):
    """The Chorin projection template on foam, run with `(pred, press, corr)` in place of `nsm`.

    `pred` is the predictor NSM on the full state, `press` / `corr` the pressure and corrector
    systems of the model's split.

    References:
        Chorin 1968, https://doi.org/10.1090/S0025-5718-1968-0242392-2
    """
    linear_solver = OptionParam(
        LinearSolver, default=LinearSolver.hypre,
        members=[LinearSolver.hypre, LinearSolver.schur_gamg, LinearSolver.gmres], doc=(
            "Linear solves (linsolve): 'hypre' = hypre FlexGMRES + one BoomerAMG V-cycle on the "
            "operator assembled by colouring, left-scaled by its inverse cell blocks; "
            "'schur-gamg' = flexible GMRES right-preconditioned by OpenFOAM's GAMG on the "
            "elliptic rows' Schur complement; 'gmres' = flexible GMRES, no preconditioner."))

    def _control(self):
        """Pass the linear solver choice to the controlDict."""
        return f"linearSolver {self.linear_solver};\n"

    def _hypre(self):
        """True for `linear_solver=LinearSolver.hypre`."""
        return self.linear_solver == "hypre"


def _options(hypre, build):
    """`Make/options` of the build tree `build`: preCICE; with the hypre prefix `hypre` its
    static library and OpenFOAM's MPI flags."""
    inc, lib = "", "-lprecice"
    if (PRECICE / "lib" / "libprecice.so").exists():
        inc = f"-I{PRECICE}/include "
        lib = f"-L{PRECICE}/lib -Wl,-rpath,{PRECICE}/lib -lprecice"
    head = ""
    if hypre is not None:   # static: nothing to bind at run time
        head = "-include $(GENERAL_RULES)/mplib$(WM_MPLIB)\n"
        inc += f"-DZS_HYPRE $(PFLAGS) $(PINC) -I{hypre}/include "
        lib += f" {hypre}/lib/libHYPRE.a $(PLIBS) -lm"
    return (f"{head}EXE_INC = -std=c++17 -DZS_PRECICE {inc}-I{build} -I{FOAM_ROOT} "
            "-I/usr/include/eigen3 \\\n"
            "    -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\n"
            f"EXE_LIBS = -lfiniteVolume -lmeshTools {lib}\n")


def _precice_dict(c):
    """The `precice` controlDict entry of a coupled run (empty without coupling)."""
    if not c:
        return ""
    return (f"precice {{ participant {c['participant']}; config \"{c['config']}\"; "
            f"mesh {c['mesh']}; write ({' '.join(c['write'])}); "
            f"read ({' '.join(c['read'])}); tag {c['tag']}; }}\n")


def _cyclic(nsm):
    """`{tag: partner}` of the model's Periodic BCs as OpenFOAM cyclic pairs; each side must name
    the other.
    """
    from zoomy_core.model.boundary_conditions import Periodic
    bcs = getattr(nsm.sm, "_bc_source", None)
    pairs = {} if bcs is None else {
        b.tag: b.periodic_to_physical_tag
        for b in bcs.boundary_conditions_list if type(b) is Periodic}
    for a, b in pairs.items():
        if pairs.get(b) != a:
            raise ValueError(f"Periodic {a!r} -> {b!r} needs Periodic {b!r} -> {a!r} "
                             "(OpenFOAM's cyclic patches come in pairs)")
    return pairs


def _write_stencils(mesh, topo, order, inv, parts, cells):
    """Write `constant/zoomyStencil` of every part: zoomy_core's LSQ tables (derivative aux fit,
    boundary closure) per cell, in the global cell numbering of the decomposition.
    """
    nc = int(mesh.n_inner_cells)
    dim = int(mesh.dimension)
    geom = compute_geometry(topo, np.asarray(mesh.vertex_coordinates)[:dim])
    scale = np.asarray(topo.lsq_scale_factors)
    base = np.asarray(geom["lsq_gradQ"])[:nc] * scale             # (nc, R, M)
    nbr = np.asarray(topo.lsq_neighbors)[:nc]
    bdy = np.asarray(topo.lsq_boundary_face_neighbors)[:nc]
    bfc = np.zeros((int(mesh.n_boundary_faces), 3))
    bfc[:, :dim] = np.asarray(geom["face_centers"])[np.asarray(topo.boundary_faces), :dim]
    R, M = base.shape[1:]
    cl = topo.lsq_plan.get("closure")
    where, active, table = {}, np.zeros((0, 0), dtype=bool), None
    if cl is not None and len(cl["cells"]):
        where = {int(c): i for i, c in enumerate(cl["cells"])}
        active = np.asarray(cl["active"])                         # (n_class, n_cc)
        table = np.asarray(geom["lsq_closure"]) * scale           # (n_class, n_cc, R, M)
    # global numbering of the parts' cells (OpenFOAM's globalIndex: part by part)
    owner = np.zeros(nc, dtype=int)
    local = np.arange(nc)
    sizes = []
    for p, a in enumerate(cells):
        if a is not None:
            owner[a], local[a] = p, np.arange(len(a))
        sizes.append(nc if a is None else len(a))
    gid = np.concatenate([[0], np.cumsum(sizes)])[owner] + local     # per OF cell

    def row(values):
        """Format an array as one line of repr floats."""
        return " ".join(repr(float(v)) for v in np.ravel(values))
    for d, a in zip(parts, cells):
        lines = [f"{nc if a is None else len(a)} {nbr.shape[1]} {bdy.shape[1]} {M}",
                 " ".join(str(int(p)) for mi in topo.lsq_monomial_multi_index
                          for p in (tuple(mi) + (0, 0, 0))[:3])]
        for k in (range(nc) if a is None else a):     # OF cells of this part
            c = order[k]
            b = bdy[c][bdy[c] >= 0]
            near = np.where(nbr[c] < nc, nbr[c], c)   # (padding is the cell itself)
            parts_c = [str(len(b)), row(bfc[b]), " ".join(str(int(gid[inv[j]])) for j in near),
                       row(base[c])]
            act = np.flatnonzero(active[:, where[c]]) if c in where else []
            parts_c.append(str(len(act)))
            parts_c += [f"{int(j)} {row(table[j, where[c]])}" for j in act]
            lines.append(" ".join(parts_c))
        (d / "constant" / "zoomyStencil").write_text("\n".join(lines) + "\n")


def _write_colours(topo, nc, order, parts, cells):
    """Write `constant/zoomyColours` of every part (count and number of colours, then one colour
    per OF cell): a greedy distance-2 colouring of the global LSQ pattern in the mesh's cell order,
    so the cells of a stencil (a matrix row) differ and rt.H's hypre_solve recovers the operator by
    one probe per colour and row (9 colours on quads with the vertex stencil).
    """
    import scipy.sparse as sps
    nbr = np.asarray(topo.lsq_neighbors)[:nc]
    cell = np.arange(nc)[:, None]
    stencil = np.concatenate([cell, np.where(nbr < nc, nbr, cell)], axis=1)
    P = sps.csr_matrix((np.ones(stencil.size, np.int8),
                        (np.repeat(np.arange(nc), stencil.shape[1]), stencil.ravel())),
                       shape=(nc, nc))
    C = (P.T @ P).tocsr()                          # two cells in one row
    ptr, ind = C.indptr, C.indices
    colour = np.full(nc, -1, np.int64)
    used = np.zeros(int(np.diff(ptr).max()) + 1, bool)   # greedy: <= degree + 1 colours
    for c in range(nc):
        nb = colour[ind[ptr[c]:ptr[c + 1]]]
        nb = nb[nb >= 0]
        used[nb] = True
        colour[c] = int(np.argmin(used))
        used[nb] = False
    ncol = int(colour.max()) + 1
    for d, a in zip(parts, cells):
        k = np.arange(nc) if a is None else a     # OF cells of this part
        (d / "constant" / "zoomyColours").write_text(
            f"{len(k)} {ncol}\n" + " ".join(map(str, colour[order[k]])) + "\n")


def hypre_prefix():
    """The CPU hypre install symFoam links (`~/.cache/zoomy/hypre-cpu-3.2.0`, or
    `ZOOMY_HYPRE_DIR`), built first by `containers/zoomy_amrex/build_hypre_cuda.sh cpu` when
    missing.

    References:
        Falgout, Yang 2002, https://doi.org/10.1007/3-540-47789-6_66
    """
    if os.environ.get("ZOOMY_HYPRE_DIR"):
        return Path(os.environ["ZOOMY_HYPRE_DIR"])
    prefix = HYPRE_CACHE / f"hypre-cpu-{HYPRE_VERSION}"
    if (prefix / "lib" / "libHYPRE.a").exists():
        return prefix
    if not HYPRE_SCRIPT.exists():
        raise RuntimeError(f"no hypre in {prefix}: build it with build_hypre_cuda.sh cpu "
                           "(Zoomy containers/zoomy_amrex), or "
                           "linear_solver=LinearSolver.schur_gamg")
    HYPRE_CACHE.mkdir(parents=True, exist_ok=True)
    with open(HYPRE_CACHE / ".hypre-cpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (prefix / "lib" / "libHYPRE.a").exists():
            r = subprocess.run(["bash", str(HYPRE_SCRIPT), "cpu"], capture_output=True, text=True,
                               env=dict(os.environ, HYPRE_VERSION=HYPRE_VERSION,
                                        ZOOMY_HYPRE_CACHE=str(HYPRE_CACHE)))
            if r.returncode != 0:
                raise RuntimeError(f"hypre build failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return prefix


def _seam(mesh, tag):
    """Boundary-face centres of `tag`, in the mesh's boundary order."""
    names = list(mesh.boundary_conditions_sorted_names)
    nb = int(mesh.n_boundary_faces)
    nums = np.asarray(mesh.boundary_face_function_numbers, dtype=int)[:nb]
    faces = np.asarray(mesh.boundary_face_face_indices, dtype=int)[:nb]
    fc = np.asarray(mesh.face_centers)
    fc = (fc if fc.shape[0] == int(mesh.n_faces) else fc.T)[:, :int(mesh.dimension)]
    return fc[faces[nums == names.index(tag)]]


def _separation(mesh, tag, partner):
    """OpenFOAM's cyclic `separation` vector of `tag` (from the partner's faces to its own)."""
    s = np.zeros(3)
    s[:int(mesh.dimension)] = _seam(mesh, tag).mean(axis=0) - _seam(mesh, partner).mean(axis=0)
    return f"({float(s[0])!r} {float(s[1])!r} {float(s[2])!r})"


def _structured(mesh):
    """True for a uniform 1-D, quad or hexahedral grid (blockMesh); anything else goes through
    gmshToFoam.
    """
    if mesh.type == "line":
        return True
    dim = {"quad": 2, "hexahedron": 3}.get(mesh.type)
    if dim is None:
        return False
    xc = np.round(np.asarray(mesh.cell_centers)[:dim, :int(mesh.n_inner_cells)], 9)
    return np.prod([len(np.unique(x)) for x in xc]) == int(mesh.n_inner_cells)


def _write_extruded_msh(mesh, path, cyclic=None):
    """Write a 2-D mesh as one layer of prisms or hexahedra (gmsh 2.2) with the boundary tags on the
    side faces and `frontAndBack` at z = 0, 1; cyclic faces are sorted along the seam.
    """
    import meshio
    X = np.asarray(mesh.vertex_coordinates)[:2].T
    nv, nc = len(X), int(mesh.n_inner_cells)
    cv = np.asarray(mesh.cell_vertices)[:, :nc].T.copy()
    e = np.roll(cv, -1, axis=1)                              # counter-clockwise cells
    area = (X[cv, 0] * X[e, 1] - X[e, 0] * X[cv, 1]).sum(axis=1)
    cv[area < 0] = cv[area < 0, ::-1]
    names = list(mesh.boundary_conditions_sorted_names)
    fc = np.asarray(mesh.face_centers)
    fc = (fc if fc.shape[0] == int(mesh.n_faces) else fc.T)[:, :2]
    edges = {i: [] for i in range(len(names) + 1)}
    for k, f in enumerate(np.asarray(mesh.boundary_face_face_indices)[:int(mesh.n_boundary_faces)]):
        c = int(np.asarray(mesh.face_cells)[0, f])   # the owner (a periodic mesh remaps boundary_face_cells)
        a = cv[c]
        b = np.roll(a, -1)
        j = int(np.argmin(np.linalg.norm(0.5 * (X[a] + X[b]) - fc[f], axis=1)))
        edges[int(mesh.boundary_face_function_numbers[k])].append((a[j], b[j]))
    kind = {3: ("wedge", "triangle"), 4: ("hexahedron", "quad")}[cv.shape[1]]
    pts = np.vstack([np.c_[X, np.zeros(nv)], np.c_[X, np.ones(nv)]])
    blocks, phys = [(kind[0], np.c_[cv, cv + nv])], [np.full(nc, len(names) + 1)]
    for t, ab in edges.items():
        if ab:
            ab = np.array(ab)
            if t < len(names) and names[t] in (cyclic or {}):
                mid = 0.5 * (X[ab[:, 0]] + X[ab[:, 1]])
                along = int(np.ptp(mid[:, 1]) > np.ptp(mid[:, 0]))
                ab = ab[np.argsort(mid[:, along], kind="stable")]
            blocks.append(("quad", np.c_[ab[:, 1], ab[:, 0], ab[:, 0] + nv, ab[:, 1] + nv]))
            phys.append(np.full(len(ab), t + 1))
    blocks += [(kind[1], cv[:, ::-1]), (kind[1], cv + nv)]
    phys += [np.full(nc, len(names) + 2)] * 2
    field = {nm: [i + 1, 2] for i, nm in enumerate(names)}
    field.update(internal=[len(names) + 1, 3], frontAndBack=[len(names) + 2, 2])
    meshio.write(str(path), meshio.Mesh(pts, blocks, cell_data={
        "gmsh:physical": phys, "gmsh:geometrical": phys}, field_data=field),
        file_format="gmsh22", binary=False)


def _read_vectors(path):
    """Internal field of an ascii volVectorField as an array."""
    body = Path(path).read_text().split("internalField", 1)[1].split("boundaryField", 1)[0]
    return np.array([v.split() for v in re.findall(r"\(([^()]*)\)", body)], dtype=float)


def _side_names(mesh, dim):
    """Boundary tag of each side (x-lo, x-hi, y-lo, ...), read off the boundary faces by their
    outward normal.
    """
    names = list(mesh.boundary_conditions_sorted_names)
    nums = np.asarray(mesh.boundary_face_function_numbers, dtype=int)
    faces = np.asarray(mesh.boundary_face_face_indices, dtype=int)
    normals = np.asarray(mesh.face_normals)[:dim]
    side = {}
    for i, f in enumerate(faces[:int(mesh.n_boundary_faces)]):
        ax = int(np.argmax(np.abs(normals[:, f])))
        side.setdefault((ax, normals[ax, f] > 0), names[nums[i]])
    return [side[(ax, hi)] for ax in range(dim) for hi in (False, True)]


def _write_field(case, name, vals, patches):
    """Write the scalar field `0/<name>` with the given patch types."""
    body = "\n".join(repr(float(v)) for v in vals)
    bf = "\n".join(f"  {p} {{ type {t}; }}" for p, t in patches.items())
    (case / "0" / name).write_text(
        f"FoamFile {{ version 2.0; format ascii; class volScalarField; object {name}; }}\n"
        f"dimensions [0 0 0 0 0 0 0];\ninternalField nonuniform List<scalar>\n"
        f"{len(vals)}\n(\n{body}\n);\nboundaryField {{\n{bf}\n}}\n")


def _state_time(d):
    """The time of the restart state in the time directory `d`; None: none."""
    try:
        m = re.search(r"\btime\s+([^;\s]+)\s*;", (d / "uniform" / "zoomyState").read_text())
    except OSError:
        return None
    return None if m is None else float(m.group(1))


def restart_state(path, nprocs=1):
    """The restart state `(base, name, time)` at `path`: a time directory of `io.save_state`
    (`<base>/[processor<k>/]<name>`), or a case / checkpoint directory (its latest).
    """
    from zoomy_core.symsolver.restart import missing
    p = Path(path).resolve()
    part0 = (lambda b: b) if nprocs == 1 else (lambda b: b / "processor0")
    if not p.exists():
        raise missing(p)
    if _state_time(part0(p.parent) / p.name) is not None:
        base, name = p.parent, p.name
    else:
        times = ([(_state_time(d), d.name) for d in part0(p).iterdir()]
                 if part0(p).is_dir() else [])
        times = [(t, n) for t, n in times if t is not None]
        if not times:
            other = "serial" if nprocs > 1 else "parallel (processor*)"
            raise FileNotFoundError(
                f"restart_from {str(path)!r}: no foam restart state for {nprocs} rank(s) (a time "
                f"directory with uniform/zoomyState, or a case / checkpoint directory holding one; "
                f"a {other} state needs nprocs {'1' if nprocs > 1 else '> 1'})")
        base, name = p, max(times)[1]
    if nprocs > 1 and ((base / f"processor{nprocs}").exists()
                       or not (base / f"processor{nprocs - 1}" / name).is_dir()):
        n = len(list(base.glob("processor*")))
        raise ValueError(f"restart_from {str(path)!r}: written by {n} ranks, this run has nprocs "
                         f"{nprocs} (foam restarts on the same decomposition)")
    return base, name, _state_time(part0(base) / name)


def _copy_state(restart, case, parts, cells=None):
    """Copy the restart time directories into the `parts` of `case`; `cells`: their
    cellProcAddressing, None: stage them with theirs in `case` and return the new state.
    """
    base, name, time = restart
    src = [base] if len(parts) == 1 else [base / f"processor{k}" for k in range(len(parts))]
    for d, part, a in zip(src, parts, cells or [None] * len(parts)):
        addr = [f for f in (d / "constant" / "polyMesh" / "cellProcAddressing",
                            d / "cellProcAddressing") if f.exists()]
        if cells is None:                     # staged, with the decomposition
            to = case / d.relative_to(base)
            shutil.copytree(d / name, to / name)
            if addr:
                shutil.copy(addr[0], to / "cellProcAddressing")
        elif a is not None and addr and not np.array_equal(_read_labels(addr[0]), a):
            raise ValueError(f"restart_from {str(base)!r}: another decomposition than this run's "
                             f"({d.name}); restart with the same mesh, nprocs and decomposition")
        else:
            shutil.copytree(d / name, part / name)
    return case, name, time


def _read_dat(path):
    """Read an `out/*.dat` file: time on the first line, then one row per state variable."""
    lines = Path(path).read_text().split("\n")
    time = float(lines[0])
    return time, np.array([[float(x) for x in ln.split()] for ln in lines[1:] if ln.strip()])


def _read_labels(path):
    """Read an OpenFOAM ascii labelList (decomposePar's cellProcAddressing)."""
    body = Path(path).read_text().split("}", 1)[1]
    return np.array(body[body.index("(") + 1:body.index(")")].split(), dtype=int)
