"""Driver: print the symbolic solvers for foam (C++), build, run.

    solver = SymbolicHyperbolicSolverFoam(time_end=0.1)
    time, Q = solver.run(mesh, nsm, Q0, workdir)        # Q in the mesh's order

The printed C++ (``Solver.H``), the kernels (``Kernels.H``) and the app
(``symsolver/symFoam.C`` + ``rt.H``) are compiled in the OpenFOAM-13
apptainer, cached by a hash of the sources.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import param
import sympy as sp

from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.symsolver import CxxPrinter
from zoomy_core.solver.constants import ELLIPTIC_RTOL, elliptic_constants
from zoomy_core.symsolver import chorin as CH
from zoomy_core.symsolver import hyperbolic as T
from zoomy_foam._pipeline import (FOAM_ROOT, _BINCACHE, _apptainer,
                                  _mesh_geometry,
                                  _write_grid_system)
from zoomy_foam.symkernels import write_kernels

APP = FOAM_ROOT / "symsolver"
# preCICE the binary links: a prefix whose version matches the Python peers
# (conda-forge stops at 3.4.0, the container ships 3.4.1; preCICE refuses to
# couple different versions).  Unset or missing: the container's own.
PRECICE = Path(os.environ.get("ZOOMY_PRECICE_PREFIX",
                              Path.home() / "of_build" / "precice-3.4.0"))


class SymbolicHyperbolicSolverFoam(param.Parameterized):
    time_end = param.Number(default=None, allow_None=True)
    cfl = param.Number(default=0.9, bounds=(0, None))
    dt = param.Number(default=None, allow_None=True,
                      doc="Constant time step; None = adaptive CFL step.")
    dt_write = param.Number(default=None, allow_None=True,
                            doc="Output interval; None = final state only.")
    couple = param.Dict(default=None, allow_None=True,
                        doc="preCICE: dict(participant=, config=, mesh=, "
                            "write=[..], read=[..], tag=, side=).")
    subcycling = param.Boolean(default=False,
                               doc="preCICE: adaptive steps inside a window.")
    imex = param.Boolean(default=False,
                         doc="IMEX (Lie): implicit source slot by Newton.")
    implicit_tol = param.Number(default=1e-8, bounds=(0, None))
    implicit_maxiter = param.Integer(default=6, bounds=(1, None))
    gmres_tol = param.Number(default=1e-7, bounds=(0, None))
    gmres_restart = param.Integer(default=20, bounds=(1, None))
    gmres_maxiter = param.Integer(default=30, bounds=(1, None))

    def _systems(self, nsm):
        return [("nsm", nsm)]

    def _linsolve(self, nsm):
        return None

    def _print(self, nsm, dimension):
        run, march = self._print_run(nsm, dimension)
        run += ("\n\ninline auto march = [](auto Q, auto Qaux, auto p, auto time_end, "
                f"auto mesh) {{\n    return {march};\n}};")
        Q, Qaux, p, mesh = sp.symbols("Q Qaux p mesh")
        aux = CxxPrinter().procedure(
            "initial_aux", (Q, Qaux, p, mesh),
            lambda Q, Qaux, p, mesh: (T.full_aux(nsm, Q, Qaux, p, sp.S.Zero,
                                                 sp.S.One, mesh),))
        return (run + "\n\n" + aux + "\n\n"
                + self._configure(nsm, self._linsolve(nsm)) + "\n")

    def _print_run(self, nsm, dimension):
        """The printed march ``run`` and the call ``march`` makes of it."""
        Q, Qaux, p, time, time_end, mesh = sp.symbols(
            "Q Qaux p time time_end mesh")
        kw = dict(cfl=self.cfl, dimension=dimension, mood=False,
                  dt_write=self.dt_write, imex=self.imex,
                  dt_constant=None if self.dt is None else sp.Float(self.dt))
        if self.couple:
            run = CxxPrinter().procedure(
                "run", (Q, Qaux, p, time, mesh),
                lambda *a: T.hyperbolic_precice(
                    nsm, *a, side=self.couple["side"],
                    subcycling=self.subcycling, **kw))
            march = "run(Q, Qaux, p, 0.0, mesh, 0L)"
        else:
            run = CxxPrinter().procedure(
                "run", (Q, Qaux, p, time, time_end, mesh),
                lambda *a: T.hyperbolic(nsm, *a, **kw))
            march = "run(Q, Qaux, p, 0.0, time_end, mesh, 0L)"
        return run, march

    def _configure(self, nsm, linsolve=None):
        """Backend settings: limiter keyword, implicit-solve tolerances."""
        ls = linsolve or dict(tol=self.gmres_tol, restart=self.gmres_restart,
                              maxiter=self.gmres_maxiter, accept=-1.0)
        return ("inline void configure()\n{\n"
                f'    settings().limiter = "{nsm.reconstruction.limiter}";\n'
                f"    settings().lsq_degree = {int(nsm.resolved_lsq_degree())};\n"
                f"    implicit().tol = {self.implicit_tol!r};\n"
                f"    implicit().maxiter = {self.implicit_maxiter};\n"
                f"    implicit().ls_tol = {float(ls['tol'])!r};\n"
                f"    implicit().ls_restart = {int(ls['restart'])};\n"
                f"    implicit().ls_maxiter = {int(ls['maxiter'])};\n"
                f"    implicit().ls_accept = {float(ls['accept'])!r};\n}}")

    def build(self, nsm, dimension):
        """Print, write the build tree, wmake (cached).  Returns the binary."""
        self.source = self._print(nsm, dimension)
        files = {"Solver.H": self.source,
                 "symFoam.C": (APP / "symFoam.C").read_text(),
                 "rt.H": (APP / "rt.H").read_text()}
        h = hashlib.sha256(str(PRECICE).encode())
        for k in sorted(files):
            h.update(k.encode() + files[k].encode())
        tmp = _BINCACHE / "sym_kernels.H"
        _BINCACHE.mkdir(exist_ok=True)
        write_kernels(self._systems(nsm), self.source, tmp)
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
            f"EXE_INC = -std=c++17 {inc}-I{build} -I{FOAM_ROOT} -I/usr/include/eigen3 \\\n"
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
        nsm = (model if isinstance(model, NumericalSystemModel)
               else NumericalSystemModel.from_system_model(model))
        lo, hi, n, order, dim = _mesh_geometry(mesh)
        binary = self.build(nsm, dim)
        case = Path(workdir)
        if case.exists():
            shutil.rmtree(case)
        for d in ("0", "system", "constant"):
            (case / d).mkdir(parents=True)
        names = _side_names(mesh, dim)
        _write_grid_system(case, lo, hi, n, dim, names)
        (case / "system" / "controlDict").write_text(
            "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
            f"application symFoam; startFrom startTime; startTime 0; stopAt endTime;\n"
            f"endTime {1e30 if self.time_end is None else self.time_end!r}; deltaT 1; writeControl timeStep; "
            f"writeInterval 1; spaceDimension {dim};\n" + _precice_dict(self.couple))
        Q0 = np.asarray(Q0)[:, order]
        patches = {nm: "zeroGradient" for nm in names}
        patches["frontAndBack"] = "empty"
        if dim == 1:
            patches["topAndBottom"] = "empty"
        for i, q in enumerate(Q0):
            _write_field(case, f"Q{i}", q, patches)
        r = _apptainer(f"cd {case}; blockMesh > log.blockMesh 2>&1 && "
                       f"{binary} > log.symFoam 2>&1",
                       binds=[case.parent, binary.parent], capture_output=True,
                       text=True)
        if r.returncode != 0:
            raise RuntimeError(f"symFoam failed, see {case}/log.symFoam:\n"
                               + (case / "log.symFoam").read_text()[-3000:])
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        read = lambda f: _read_dat(f, inv)                       # noqa: E731
        snaps = sorted((case / "out").glob("[0-9]*.dat"), key=lambda f: int(f.stem))
        self.snapshots = [read(f) for f in snaps]
        return read(case / "out" / "final.dat")


def _precice_dict(c):
    if not c:
        return ""
    return (f"precice {{ participant {c['participant']}; config \"{c['config']}\"; "
            f"mesh {c['mesh']}; write ({' '.join(c['write'])}); "
            f"read ({' '.join(c['read'])}); tag {c['tag']}; }}\n")


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


def _read_dat(path, inv):
    lines = Path(path).read_text().split("\n")
    time = float(lines[0])
    Q = np.array([[float(x) for x in ln.split()] for ln in lines[1:] if ln.strip()])
    return time, Q[:, inv]


class SymbolicChorinSolverFoam(SymbolicHyperbolicSolverFoam):
    """The Chorin projection template on foam.

        time, Q = SymbolicChorinSolverFoam(time_end=0.5, dt=2**-7).run(
            mesh, (pred, press, corr), Q0, workdir)

    ``pred``: predictor NSM on the full state; ``press`` / ``corr``: the
    pressure and corrector systems of the model's split (zoomy's ``dt``)."""
    time_order = param.Integer(default=1, bounds=(1, 2))
    pressure_tol = param.Number(default=ELLIPTIC_RTOL, bounds=(0, None))
    pressure_maxit = param.Integer(default=100, bounds=(1, None))
    pressure_restart = param.Integer(default=None, allow_None=True, bounds=(1, None))

    def _systems(self, nsm):
        return [("pred", nsm), ("press", self.press), ("corr", self.corr)]

    def _linsolve(self, nsm):
        n = len(self.press.equation_to_state_index) * self._nc
        ell = elliptic_constants(n_unknowns=n, tol=self.pressure_tol,
                                 restart=self.pressure_restart).values
        tol = float(ell["c_elliptic_tol"])
        return dict(tol=tol, restart=int(ell["c_elliptic_restart"]),
                    maxiter=self.pressure_maxit, accept=max(1e-6, 10.0 * tol))

    def _print_run(self, nsm, dimension):
        Q, Qaux, p, time, time_end, mesh = sp.symbols(
            "Q Qaux p time time_end mesh")
        run = CxxPrinter().procedure(
            "run", (Q, Qaux, p, time, time_end, mesh),
            lambda *a: CH.chorin(
                nsm, self.press, self.corr, *a, cfl=self.cfl,
                dimension=dimension, imex=self.imex, time_order=self.time_order,
                dt_write=self.dt_write,
                dt_constant=None if self.dt is None else sp.Float(self.dt)))
        return run, "run(Q, Qaux, p, 0.0, time_end, mesh, 0L)"

    def run(self, mesh, systems, Q0, workdir):
        pred, press, corr = systems
        named = []
        for name, sm in (("pred", pred), ("press", press), ("corr", corr)):
            nsm = (sm if isinstance(sm, NumericalSystemModel)
                   else NumericalSystemModel.from_system_model(sm))
            nsm._symbolic_name = name
            named.append(nsm)
        pred, self.press, self.corr = named
        self._nc = int(mesh.n_inner_cells)
        return super().run(mesh, pred, Q0, workdir)
