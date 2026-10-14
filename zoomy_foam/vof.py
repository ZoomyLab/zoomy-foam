"""The resolved participant of an SME | VOF coupling: OpenFOAM 13 `incompressibleVoF` in a
rectangular tank, with the `swePreciceCoupling` function object (this package's
`swePreciceCoupling/`) imposing the reduced arm's column on the coupled patches.

The function object compiles the reduced model's kernels in (its `Model.H` /
`NumericsKernels.H`, emitted here from the model's SystemModel), so `coupling_library(sm)`
builds it per model, cached by the hash of its sources, the emitted headers and the image.

    tank = Tank(h0=0.04)
    write_config(run / "precice-config.xml", arms=[Arm("Swe", "SweMesh", "VofInletMesh")],
                 t_end=2.5, window=2e-3, exchange=run)
    case = write_case(run / "vof_case", tank, arms=..., config=..., t_end=2.5, window=2e-3)
    lib = coupling_library(sm)
    proc = start(case, lib)                    # blockMesh, setFields, foamRun

The column contract: six fields `[b, h, u, v, w, p]` (`<field>_S` written by the reduced
arm, `<field>_V` by this participant), the reduced arm's vertices `(x, y, sigma)` at the
cell-centre heights of `nz` = the tank's cells over its height (`ny`), the model's
`project_nz`.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from zoomy_foam._pipeline import FOAM_ROOT, OF_BASHRC, SIF, _BINCACHE, _apptainer, _apptainer_cmd

FIELDS = ("b", "h", "u", "v", "w", "p")
#: the function object's settings of the production SME | VOF path at Fr* = 0.752 (the
#: star-state imposition with the characteristic mass row, no debt ledger;
#: thesis/cases/applications/sme_vof_overshoot/geometry.py): `executeAtStart no` keeps the
#: coupling cycle off t = 0, where it would close a window this participant never ran
FO_STARSTATE = {
    "executeAtStart": "no", "relax": "1.0", "imposeMode": "starstate",
    "qstarMode": "characteristic", "maxCo": "1.0", "maxAlphaCo": "1.0",
    "debtRepayWindows": "1e18", "ledgerLog": "yes", "writeColumns": "yes",
    "pSub": "star", "fitTarget": "charged", "writePhi": "yes", "writeState": "yes",
    "writeAlpha": "no", "writePressure": "no", "imposeVertical": "yes",
    "alphaDirSwitch": "no", "consistentWrite": "no"}

_HDR = "FoamFile {{ format ascii; class {c}; object {o}; }}"
_CONSTANT = {
    "g": ("uniformDimensionedVectorField", "dimensions [0 1 -2 0 0 0 0];\nvalue (0 -9.81 0);"),
    "momentumTransport": ("dictionary", "simulationType laminar;"),
    "phaseProperties": ("dictionary", "phases (water air);\nsigma 0;"),
    "physicalProperties.air": ("dictionary", "viscosityModel constant;\nnu  1.48e-5;\nrho 1;"),
    "physicalProperties.water": ("dictionary", "viscosityModel constant;\nnu  1e-6;\nrho 1000;"),
}
_SCHEMES = """ddtSchemes        { default Euler; }
gradSchemes       { default Gauss linear; }
divSchemes
{
    div(rhoPhi,U)   Gauss linearUpwind grad(U);
    div(phi,alpha)  Gauss interfaceCompression vanLeer 1;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes  { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes     { default corrected; }"""
_SOLUTION = """solvers
{
    "alpha.water.*"
    {
        nCorrectors 2; nSubCycles 1; MULESCorr yes;
        MULES { nIter 10; tolerance 1e-2; }
        solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0;
    }
    "pcorr.*"  { solver PCG; preconditioner DIC; tolerance 1e-5; relTol 0; }
    p_rgh      { solver PCG; preconditioner DIC; tolerance 1e-7; relTol 0.05; }
    p_rghFinal { solver PCG; preconditioner DIC; tolerance 1e-7; relTol 0; }
    "U.*"      { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0; }
}
PIMPLE { momentumPredictor no; nCorrectors 3; nNonOrthogonalCorrectors 0; }"""


@dataclass
class Arm:
    """A reduced participant coupled to the tank: its preCICE name, its mesh, the tank's mesh
    of the shared patch, and which tank side it sits on (`left`: the inlet at `x = 0`,
    `right`: the outlet at `x = lx`)."""
    participant: str
    mesh: str
    vof_mesh: str
    side: str = "left"


@dataclass
class Tank:
    """The VOF tank `[0, lx] x [0, ly]` (thin in z), `nx x ny` cells, still water `h0`; the sides
    an `Arm` does not take are walls."""
    lx: float = 1.5
    ly: float = 0.4
    nx: int = 120
    ny: int = 40
    lz: float = 0.02
    h0: float = 0.04
    fo: dict = field(default_factory=lambda: dict(FO_STARSTATE))


def _w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip("\n") + "\n")


def write_config(path, *, arms, t_end, window, exchange):
    """The preCICE configuration of the tank (participant `Vof`) and its reduced `arms`:
    one `parallel-explicit` scheme for one arm, a `multi` scheme with the tank as controller
    (max-iterations 1: the explicit-equivalent single pass) for several; m2n sockets through
    the directory `exchange` (both containers must see it at that path)."""
    path = Path(path)
    ds, dv = [f"{f}_S" for f in FIELDS], [f"{f}_V" for f in FIELDS]
    data = "\n  ".join(f'<data:scalar name="{d}"/>' for d in ds + dv)
    use = "\n    ".join(f'<use-data name="{d}"/>' for d in ds + dv)
    meshes = "\n".join(f'  <mesh name="{m}" dimensions="3">\n    {use}\n  </mesh>'
                       for a in arms for m in (a.mesh, a.vof_mesh))
    nn = ('<mapping:nearest-neighbor direction="read" from="{f}" to="{t}" '
          'constraint="consistent"/>')
    parts = []
    for a in arms:
        parts.append(f"""  <participant name="{a.participant}">
      <provide-mesh name="{a.mesh}"/>
      <receive-mesh name="{a.vof_mesh}" from="Vof"/>
      """ + "\n      ".join([f'<write-data name="{d}" mesh="{a.mesh}"/>' for d in ds]
                            + [f'<read-data name="{d}" mesh="{a.mesh}"/>' for d in dv]
                            + [nn.format(f=a.vof_mesh, t=a.mesh)]) + "\n  </participant>")
    vof = []
    for a in arms:
        vof += [f'<provide-mesh name="{a.vof_mesh}"/>',
                f'<receive-mesh name="{a.mesh}" from="{a.participant}"/>']
        vof += [f'<write-data name="{d}" mesh="{a.vof_mesh}"/>' for d in dv]
        vof += [f'<read-data name="{d}" mesh="{a.vof_mesh}"/>' for d in ds]
        vof += [nn.format(f=a.mesh, t=a.vof_mesh)]
    parts.append('  <participant name="Vof">\n      ' + "\n      ".join(vof)
                 + "\n  </participant>")
    m2n = "\n  ".join(f'<m2n:sockets acceptor="{a.participant}" connector="Vof" '
                      f'exchange-directory="{exchange}"/>' for a in arms)
    ex = "\n    ".join(
        s for a in arms for s in
        [f'<exchange data="{d}" mesh="{a.mesh}" from="{a.participant}" to="Vof"/>' for d in ds]
        + [f'<exchange data="{d}" mesh="{a.vof_mesh}" from="Vof" to="{a.participant}"/>'
           for d in dv])
    if len(arms) == 1:
        scheme = f"""<coupling-scheme:parallel-explicit>
    <participants first="{arms[0].participant}" second="Vof"/>
    <max-time value="{t_end}"/>
    <time-window-size value="{window}"/>
    {ex}
  </coupling-scheme:parallel-explicit>"""
    else:
        ctrl = "\n    ".join([f'<participant name="{a.participant}"/>' for a in arms]
                             + ['<participant name="Vof" control="yes"/>'])
        scheme = f"""<coupling-scheme:multi>
    {ctrl}
    <max-time value="{t_end}"/>
    <time-window-size value="{window}"/>
    <max-iterations value="1"/>
    {ex}
  </coupling-scheme:multi>"""
    _w(path, f"""<?xml version="1.0" encoding="UTF-8" ?>
<precice-configuration>
  {data}

{meshes}

{chr(10).join(parts)}

  {m2n}

  {scheme}
</precice-configuration>""")
    return path


def write_case(dest, tank, *, arms, config, t_end, window, write_interval=0.05):
    """The tank case in `dest`: mesh (`blockMeshDict`), still water `h0` (`setFieldsDict`),
    0/ fields with the coupled patches `fixedValue` (the function object imposes them),
    `controlDict` with the function object on `config`. Returns `dest`."""
    C = Path(dest)
    if C.exists():
        shutil.rmtree(C)
    for name, (cls, body) in _CONSTANT.items():
        _w(C / "constant" / name, _HDR.format(c=cls, o=name) + "\n" + body)
    _w(C / "system/fvSchemes", _HDR.format(c="dictionary", o="fvSchemes") + "\n" + _SCHEMES)
    _w(C / "system/fvSolution", _HDR.format(c="dictionary", o="fvSolution") + "\n" + _SOLUTION)
    sides = {a.side: a for a in arms}
    if set(sides) - {"left", "right"} or len(sides) != len(arms):
        raise ValueError("one arm per tank side, left or right")
    left = "inlet" if "left" in sides else "wall_left"
    right = "outlet" if "right" in sides else "wall_right"
    kind = lambda s, n: "patch" if s in sides else "wall"  # noqa: E731
    lx, ly, lz = tank.lx, tank.ly, tank.lz
    _w(C / "system/blockMeshDict", f"""{_HDR.format(c="dictionary", o="blockMeshDict")}
convertToMeters 1;
vertices (
  (0 0 0) ({lx} 0 0) ({lx} {ly} 0) (0 {ly} 0)
  (0 0 {lz}) ({lx} 0 {lz}) ({lx} {ly} {lz}) (0 {ly} {lz})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({tank.nx} {tank.ny} 1) simpleGrading (1 1 1) );
boundary (
  {left}        {{ type {kind("left", left)}; faces ((0 4 7 3)); }}
  {right} {{ type {kind("right", right)};  faces ((1 2 6 5)); }}
  bottom       {{ type wall;  faces ((0 1 5 4)); }}
  atmosphere   {{ type patch; faces ((3 7 6 2)); }}
  frontAndBack {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();""")
    _w(C / "system/setFieldsDict", f"""{_HDR.format(c="dictionary", o="setFieldsDict")}
defaultFieldValues ( volScalarFieldValue alpha.water 0 );
regions ( boxToCell {{ box (-1 -1 -1) ({lx + 1} {tank.h0} {lz + 1});
  fieldValues ( volScalarFieldValue alpha.water 1 ); }} );""")
    cU, ca, cp = ("{ type fixedValue; value uniform (0 0 0); }",
                  "{ type fixedValue; value uniform 0; }",
                  "{ type fixedFluxPressure; value uniform 0; }")
    wU, wa, wp = ("{ type noSlip; }", "{ type zeroGradient; }",
                  "{ type fixedFluxPressure; value uniform 0; }")

    def side(s, coupled, wall):
        return coupled if s in sides else wall

    for name, cls, dims, internal, coupled, wall, bottom, atm, fb in (
            ("U", "volVectorField", "[0 1 -1 0 0 0 0]", "(0 0 0)", cU, wU, "{ type noSlip; }",
             "{ type pressureInletOutletVelocity; value uniform (0 0 0); }",
             "{ type empty; }"),
            ("alpha.water", "volScalarField", "[]", "0", ca, wa, "{ type zeroGradient; }",
             "{ type inletOutlet; inletValue uniform 0; value uniform 0; }", "{ type empty; }"),
            ("p_rgh", "volScalarField", "[1 -1 -2 0 0 0 0]", "0", cp, wp,
             "{ type fixedFluxPressure; value uniform 0; }",
             "{ type totalPressure; p0 uniform 0; value uniform 0; }", "{ type empty; }")):
        _w(C / "0" / name, f"""{_HDR.format(c=cls, o=name)}
dimensions {dims}; internalField uniform {internal};
boundaryField {{
  {left}      {side("left", coupled, wall)}
  {right} {side("right", coupled, wall)}
  bottom     {bottom}
  atmosphere {atm}
  frontAndBack {fb}
}}""")
    if len(arms) == 1:
        a = arms[0]
        patch = "inlet" if a.side == "left" else "outlet"
        iface = f"patch {patch}; preciceMesh {a.vof_mesh}; domainHeight {ly};"
    else:
        iface = "interfaces\n    {\n" + "\n".join(
            f"      {a.side}  {{ patch {'inlet' if a.side == 'left' else 'outlet'};  "
            f"mesh {a.vof_mesh};  domainHeight {ly}; }}" for a in arms) + "\n    }"
    fo = dict(tank.fo)
    head = [f"{k} {fo.pop(k)};" for k in ("executeAtStart",) if k in fo]
    fo_lines = " ".join(f"{k} {v};" for k, v in fo.items())
    _w(C / "system/controlDict", f"""FoamFile {{ format ascii; class dictionary; object controlDict; }}
application foamRun; solver incompressibleVoF;
startFrom startTime; startTime 0; stopAt endTime; endTime {t_end};
deltaT {window}; writeControl timeStep; writeInterval 1000000;
purgeWrite 0; writeFormat ascii; writePrecision 8; timeFormat general;
runTimeModifiable yes; adjustTimeStep no;   // the function object owns the CFL step
functions
{{
  sweCoupling
  {{
    type swePreciceCoupling; libs ("libswePreciceCoupling.so");
    {" ".join(head)}
    {iface}
    outputInterval {write_interval};
    {fo_lines}
    preciceConfig "{config}"; preciceParticipant Vof;
    preciceReadData  ( {" ".join(f"{f}_S" for f in FIELDS)} );
    preciceWriteData ( {" ".join(f"{f}_V" for f in FIELDS)} );
  }}
}}""")
    return C


def _image_id():
    """The OpenFOAM image's identity in the build hash (inside the image: its environment)."""
    from zoomy_foam.symsolver import _image_id as image_id
    return image_id(SIF)


def coupling_library(sm, *, cache=None):
    """The directory holding `libswePreciceCoupling.so` built against the reduced model `sm`
    (its SystemModel: `Model.H` / `NumericsKernels.H` / `MarchConstants.H` emitted with the
    Foam printers next to a copy of this package's C++ sources), built once per hash of the
    sources, the headers and the image."""
    from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
    from zoomy_core.transformation.to_openfoam import FoamNumericsPrinter, FoamSystemModelPrinter
    from zoomy_foam._constants import write_march_constants
    cache = Path(cache) if cache is not None else _BINCACHE
    cache.mkdir(parents=True, exist_ok=True)
    sources = sorted(FOAM_ROOT.glob("*.H")) + sorted((FOAM_ROOT / "swePreciceCoupling").glob(
        "*.[CH]")) + [FOAM_ROOT / "swePreciceCoupling/Make" / f for f in ("files", "options")]
    with tempfile.TemporaryDirectory(dir=cache) as d:
        d = Path(d)
        num = PositiveNonconservativeRusanov(model=sm)
        FoamSystemModelPrinter.write_code(sm, d / "Model.H")
        FoamNumericsPrinter.write_code(num, d / "NumericsKernels.H")
        write_march_constants(num, d / "MarchConstants.H")
        emitted = {n: (d / n).read_text() for n in ("Model.H", "NumericsKernels.H",
                                                     "MarchConstants.H")}
    h = hashlib.sha256(_image_id().encode())
    for p in sources:
        h.update(str(p.relative_to(FOAM_ROOT)).encode() + p.read_bytes())
    for n in sorted(emitted):
        h.update(n.encode() + emitted[n].encode())
    build = cache / f"swecoupling_{h.hexdigest()[:16]}"
    if (build / "lib" / "libswePreciceCoupling.so").exists():
        return build / "lib"
    # built aside and moved in whole: concurrent runs of the same model may build at once
    tmp = Path(tempfile.mkdtemp(prefix=build.name + "_", dir=cache))
    for p in sources:
        dst = tmp / p.relative_to(FOAM_ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    for n, text in emitted.items():
        (tmp / n).write_text(text)
    r = _apptainer(f"export FOAM_USER_LIBBIN={tmp}/lib; cd {tmp}/swePreciceCoupling; "
                   "wmake libso > wmake.log 2>&1; rc=$?; tail -30 wmake.log; [ $rc -eq 0 ]",
                   binds=[tmp], capture_output=True, text=True)
    if r.returncode != 0 or not (tmp / "lib" / "libswePreciceCoupling.so").exists():
        raise RuntimeError(f"swePreciceCoupling wmake failed ({tmp}):\n{r.stdout}\n{r.stderr}")
    try:
        os.rename(tmp, build)
    except OSError:                      # another run moved its build in first
        shutil.rmtree(tmp, ignore_errors=True)
    return build / "lib"


def start(case, lib, *, log=None):
    """`blockMesh`, `setFields`, then `foamRun` of the tank `case` with the function object
    from `lib`, in the background; returns the process (its output in `log`, default
    `<case>/../log.vof`)."""
    case = Path(case).resolve()
    log = Path(log) if log is not None else case.parent / "log.vof"
    script = (f"export LD_LIBRARY_PATH={lib}:$LD_LIBRARY_PATH; unset FOAM_SIGFPE FOAM_SETNAN; "
              f"cd {case} && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 "
              f"&& foamRun -case {case}")
    f = open(log, "w")
    return subprocess.Popen(_apptainer_cmd(script, binds=[case.parent, Path(lib)]),
                            stdout=f, stderr=subprocess.STDOUT, start_new_session=True)


def to_hdf5(case, out):
    """The tank's written time directories (`foamToVTK`) as a zoomy HDF5 store `out`."""
    case = Path(case).resolve()
    _apptainer(f"cd {case} && foamToVTK -ascii > log.foamToVTK 2>&1", binds=[case],
               check=True)
    from zoomy_prepost import vtk_to_hdf5
    frames = sorted((case / "VTK").glob(f"{case.name}_*.vtk"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not frames:
        raise RuntimeError(f"foamToVTK wrote no frames in {case / 'VTK'}")
    vtk_to_hdf5([str(p) for p in frames], str(out))
    return Path(out)


__all__ = ["Arm", "FIELDS", "FO_STARSTATE", "OF_BASHRC", "Tank", "coupling_library", "start",
           "to_hdf5", "write_case", "write_config"]
