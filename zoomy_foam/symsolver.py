"""Driver: print the symbolic solvers for foam (C++), build, run.

    solver = SymbolicHyperbolicSolverFoam(time_end=0.1)
    time, Q = solver.run(mesh, nsm, Q0, workdir)        # Q in the mesh's order

The printed C++ (``Solver.H``), the kernels (``Kernels.H``) and the app
(``symsolver/symFoam.C`` + ``rt.H`` + zoomy_core's ``rt_common.H``) are
compiled in the OpenFOAM-13 apptainer, cached by a hash of the sources.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import param
from scipy.spatial import cKDTree

from zoomy_core.mesh.geometry_xp import build_topology, compute_geometry
from zoomy_core.symsolver.backend_cxx import RUNTIME_HEADER, CxxBackend, CxxChorinBackend
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


class FoamDriver(param.Parameterized):
    """The foam runtime of a C++ solver backend (:mod:`zoomy_core.symsolver.backend_cxx`):
    the build tree (wmake), the case (blockMesh / gmshToFoam, fields, stencils),
    the launch (``mpirun -parallel``) and the read-back."""
    nprocs = param.Integer(default=1, bounds=(1, None),
                           doc="MPI ranks: decomposePar, mpirun -parallel.")
    decomposition = param.Selector(
        default="scotch", objects=["scotch", "simple"],
        doc="decomposePar method: 'scotch' (graph partition), 'simple' (nprocs "
            "slabs along x).")

    def _control(self):
        """Run-time settings of this solver in the controlDict."""
        return ""

    def build(self, nsm, dimension, topo):
        """Print, write the build tree, wmake (cached).  Returns the binary.
        ``topo``: the mesh's topology (the kernels carry its boundary-closure
        classes)."""
        self.source = self._print(nsm, dimension)
        files = {"Solver.H": self.source,
                 "symFoam.C": (APP / "symFoam.C").read_text(),
                 "rt.H": (APP / "rt.H").read_text(),
                 RUNTIME_HEADER.name: RUNTIME_HEADER.read_text()}
        h = hashlib.sha256(str(PRECICE).encode())
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
        inc, lib = "", "-lprecice"
        if (PRECICE / "lib" / "libprecice.so").exists():
            inc = f"-I{PRECICE}/include "
            lib = f"-L{PRECICE}/lib -Wl,-rpath,{PRECICE}/lib -lprecice"
        (build / "Make" / "options").write_text(
            f"EXE_INC = -std=c++17 -DZS_PRECICE {inc}-I{build} -I{FOAM_ROOT} "
            "-I/usr/include/eigen3 \\\n"
            "    -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\n"
            f"EXE_LIBS = -lfiniteVolume -lmeshTools {lib}\n")
        r = _apptainer(f"cd {build}; wmake > wmake.log 2>&1; rc=$?; "
                       f"tail -40 wmake.log; [ $rc -eq 0 ]",
                       binds=[build], capture_output=True, text=True)
        if r.returncode != 0 or not binary.exists():
            raise RuntimeError(f"symFoam wmake failed ({build}):\n{r.stdout}\n{r.stderr}")
        return binary

    def run(self, mesh, model, Q0, workdir):
        """March ``Q0`` (mesh order); returns ``(time, Q)`` and keeps the
        snapshots in ``self.snapshots`` [(time, Q)]."""
        nsm = self._prepare(mesh, model)
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
        if case.exists():
            shutil.rmtree(case)
        for d in ("0", "system", "constant"):
            (case / d).mkdir(parents=True)
        (case / "system" / "controlDict").write_text(
            "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
            f"application symFoam; startFrom startTime; startTime 0; stopAt endTime;\n"
            f"endTime {1e30 if self.time_end is None else self.time_end!r}; deltaT 1; writeControl timeStep; "
            f"writeInterval 1; writeFormat ascii; writePrecision 17; spaceDimension {dim};\n"
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
        _write_stencils(mesh, topo, order, inv, parts, cells)
        r = _apptainer(f"cd {case}; {run}", binds=[case.parent, binary.parent],
                       capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"symFoam failed, see {case}/log.symFoam:\n"
                               + (case / "log.symFoam").read_text()[-3000:])

        def read(name):
            time, Q = None, np.empty((Q0.shape[0], len(order)))
            for d, c in zip(parts, cells):
                time, q = _read_dat(d / "out" / name)
                Q[:, slice(None) if c is None else c] = q
            return time, Q[:, inv]
        snaps = sorted((parts[0] / "out").glob("[0-9]*.dat"), key=lambda f: int(f.stem))
        self.snapshots = [read(f.name) for f in snaps]
        return read("final.dat")


class SymbolicHyperbolicSolverFoam(FoamDriver, CxxBackend):
    """The explicit hyperbolic template on foam.

        time, Q = SymbolicHyperbolicSolverFoam(time_end=0.1).run(
            mesh, nsm, Q0, workdir)       # Q in the mesh's order"""


class SymbolicChorinSolverFoam(FoamDriver, CxxChorinBackend):
    """The Chorin projection template on foam.

        time, Q = SymbolicChorinSolverFoam(time_end=0.5, dt=2**-7).run(
            mesh, (pred, press, corr), Q0, workdir)

    ``pred``: predictor NSM on the full state; ``press`` / ``corr``: the
    pressure and corrector systems of the model's split (zoomy's ``dt``)."""
    pressure_preconditioner = param.Selector(
        default="gamg", objects=["gamg", "none"],
        doc="'gamg': GMRES right preconditioned per pressure row by OpenFOAM's "
            "GAMG on a Laplacian scaled to the operator's mean diagonal.")

    def _control(self):
        return f"pressurePreconditioner {self.pressure_preconditioner};\n"


def _precice_dict(c):
    if not c:
        return ""
    return (f"precice {{ participant {c['participant']}; config \"{c['config']}\"; "
            f"mesh {c['mesh']}; write ({' '.join(c['write'])}); "
            f"read ({' '.join(c['read'])}); tag {c['tag']}; }}\n")


def _cyclic(nsm):
    """{tag: partner} of the model's Periodic BCs: OpenFOAM cyclic pairs
    (a periodic seam is a pair, so each side must name the other)."""
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
    """``constant/zoomyStencil`` of every part: zoomy_core's LSQ fit of the
    derivative aux, per cell (:func:`geometry_xp.compute_geometry`, the one
    JAX evaluates).  Header: ``n_cells  max_nbr  max_bdy  n_mon`` and the
    monomials' powers.  Per cell: the centres of its ``n_b`` sampled
    boundary faces (zoomy_core's slots; the runtime matches them to its own
    boundary faces), the ``max_nbr`` stencil cells (global numbers of the
    decomposition; padding: the cell itself), the base table
    ``lsq_gradQ * lsq_scale_factors`` ``(R, n_mon)`` and, per active closure
    class, ``class table`` (``lsq_closure * lsq_scale_factors``, same layout,
    boundary slots last).  Across a periodic seam the stencil is the wrapped
    ring (offsets to the nearest image: in the tables)."""
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


def _seam(mesh, tag):
    """Boundary-face centres of ``tag``, in the mesh's boundary order."""
    names = list(mesh.boundary_conditions_sorted_names)
    nb = int(mesh.n_boundary_faces)
    nums = np.asarray(mesh.boundary_face_function_numbers, dtype=int)[:nb]
    faces = np.asarray(mesh.boundary_face_face_indices, dtype=int)[:nb]
    fc = np.asarray(mesh.face_centers)
    fc = (fc if fc.shape[0] == int(mesh.n_faces) else fc.T)[:, :int(mesh.dimension)]
    return fc[faces[nums == names.index(tag)]]


def _separation(mesh, tag, partner):
    """OpenFOAM's cyclic ``separation`` of ``tag``: the vector from the
    partner's faces to its own (as blockMesh writes it)."""
    s = np.zeros(3)
    s[:int(mesh.dimension)] = _seam(mesh, tag).mean(axis=0) - _seam(mesh, partner).mean(axis=0)
    return f"({float(s[0])!r} {float(s[1])!r} {float(s[2])!r})"


def _structured(mesh):
    """A uniform 1-D, quad or hexahedral grid: blockMesh; anything else:
    gmshToFoam."""
    if mesh.type == "line":
        return True
    dim = {"quad": 2, "hexahedron": 3}.get(mesh.type)
    if dim is None:
        return False
    xc = np.round(np.asarray(mesh.cell_centers)[:dim, :int(mesh.n_inner_cells)], 9)
    return np.prod([len(np.unique(x)) for x in xc]) == int(mesh.n_inner_cells)


def _write_extruded_msh(mesh, path, cyclic=None):
    """A 2-D mesh as one layer of prisms / hexahedra (z in [0, 1]), gmsh 2.2:
    the boundary tags on the side faces, ``frontAndBack`` at z = 0 and 1.
    ``cyclic`` tags list their faces sorted along the seam, so the faces of a
    pair correspond one by one (OpenFOAM's cyclic ordering)."""
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
    """The internal field of an ascii volVectorField."""
    body = Path(path).read_text().split("internalField", 1)[1].split("boundaryField", 1)[0]
    return np.array([v.split() for v in re.findall(r"\(([^()]*)\)", body)], dtype=float)


def _side_names(mesh, dim):
    """Boundary tag of each side (x-lo, x-hi, y-lo, y-hi), read off the
    mesh's boundary faces by their outward normal."""
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
    body = "\n".join(repr(float(v)) for v in vals)
    bf = "\n".join(f"  {p} {{ type {t}; }}" for p, t in patches.items())
    (case / "0" / name).write_text(
        f"FoamFile {{ version 2.0; format ascii; class volScalarField; object {name}; }}\n"
        f"dimensions [0 0 0 0 0 0 0];\ninternalField nonuniform List<scalar>\n"
        f"{len(vals)}\n(\n{body}\n);\nboundaryField {{\n{bf}\n}}\n")


def _read_dat(path):
    lines = Path(path).read_text().split("\n")
    time = float(lines[0])
    return time, np.array([[float(x) for x in ln.split()] for ln in lines[1:] if ln.strip()])


def _read_labels(path):
    """An OpenFOAM ascii labelList (decomposePar's cellProcAddressing)."""
    body = Path(path).read_text().split("}", 1)[1]
    return np.array(body[body.index("(") + 1:body.index(")")].split(), dtype=int)
