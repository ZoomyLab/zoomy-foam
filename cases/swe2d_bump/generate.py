#!/usr/bin/env python
"""Generate the swe2d_bump run: 2D SWE inflow channel with an OFF-CENTER
submerged Gaussian bump in participant 1 — the wake/deflection crossing
the interface is genuinely transverse-nonuniform, so the coupled pair
tests cell-by-cell transfer (validated against the 2D monolithic).

  RUN/mono/   : monolithic [0,LX]x[0,LY]
  RUN/part1/  : [0,XMID]   (inflow left, coupled right, bump inside)
  RUN/part2/  : [XMID,LX]  (coupled left, outflow right)
  RUN/precice-config.xml

Env: RUNDIR overrides the run folder.
"""
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CHERE = Path(str(HERE).replace("/mnt/userdrive/Users/home/", "/Users/"))
RUN = Path(os.environ.get("RUNDIR", str(CHERE / "run")))
CFG = RUN / "precice-config.xml"

# ── geometry / physics ──────────────────────────────────────────────────────
LX, LY, XMID = 2.0, 0.5, 1.0
NXH, NY = 100, 50                       # per-half cells; dx = dy = 0.01
H_IN, Q_IN = 0.2, 0.1                   # inflow depth / discharge (u = 0.5)
BX, BY, BS = 0.6, 0.35, 0.08            # bump: center (off-axis!), sigma
# A=0.08 keeps the crest subcritical (h_crest=0.12, Fr~0.8): strong
# deflection without a transonic jump.  BUMP_A=0 -> flat channel (the
# uniform-flow steady state, the scheme's y-stability control).
BA = float(os.environ.get("BUMP_A", "0.08"))
T_END = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
WINDOW = float(sys.argv[2]) if len(sys.argv) > 2 else 5e-3
DT0 = 2e-3
FIELDS = ["b", "h", "u", "v", "w", "p"]
DATA_1 = [f + "_1" for f in FIELDS]
DATA_2 = [f + "_2" for f in FIELDS]

HDR = "FoamFile {{ format ascii; class {c}; object {o}; }}"


def w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip("\n"))


def bump(x, y):
    if os.environ.get("BUMP_MODE", "bump") == "ridge":   # y-uniform control
        return BA * np.exp(-(((x - BX) ** 2) / BS ** 2))
    return BA * np.exp(-(((x - BX) ** 2 + (y - BY) ** 2) / BS ** 2))


def case(name, x0, x1, nx, left, right, precice=None):
    C = RUN / name
    w(C / "system/blockMeshDict", f"""
{HDR.format(c="dictionary", o="blockMeshDict")}
convertToMeters 1;
vertices (
  ({x0} 0 0) ({x1} 0 0) ({x1} {LY} 0) ({x0} {LY} 0)
  ({x0} 0 0.1) ({x1} 0 0.1) ({x1} {LY} 0.1) ({x0} {LY} 0.1)
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {NY} 1) simpleGrading (1 1 1) );
boundary (
  {left}  {{ type patch; faces ((0 4 7 3)); }}
  {right} {{ type patch; faces ((1 2 6 5)); }}
  sides   {{ type patch; faces ((0 1 5 4) (3 7 6 2)); }}
  frontAndBack {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();
""")
    pc = ""
    if precice:
        part, mesh, wdat, rdat = precice
        pc = f"""
preciceParticipant {part};
preciceConfig "{CFG}";
preciceMeshes ( {mesh} );
preciceWriteData ( {' '.join(wdat)} );
preciceReadData ( {' '.join(rdat)} );
preciceGhost fullstate;
preciceZSamples 1;
preciceFrozenMass yes;
"""
    w(C / "system/controlDict", f"""
{HDR.format(c="dictionary", o="controlDict")}
application zoomyFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {T_END};
deltaT {DT0}; writeControl adjustableRunTime; writeInterval 0.1;
purgeWrite 0; writeFormat ascii; writePrecision 10; timeFormat general;
runTimeModifiable true; adjustTimeStep yes; maxCo 0.9; reconstructionOrder 1;
modelParameters {{ h_in {H_IN}; q_in {Q_IN}; }}
{pc}""")
    w(C / "system/fvSchemes", f"""
{HDR.format(c="dictionary", o="fvSchemes")}
ddtSchemes {{ default Euler; }}
gradSchemes {{ default cellLimited Gauss linear 1; }}
divSchemes {{ default Gauss linear; }}
laplacianSchemes {{ default Gauss linear corrected; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default corrected; }}
""")
    w(C / "system/fvSolution", f"""
{HDR.format(c="dictionary", o="fvSolution")}
solvers {{ "Q.*" {{ solver diagonal; }} }}
""")
    (C / "constant").mkdir(parents=True, exist_ok=True)

    # ── nonuniform ICs written directly (no setFields: the bump is smooth)
    xc = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    yc = (np.arange(NY) + 0.5) * LY / NY
    X, Y = np.meshgrid(xc, yc)              # row-major: y outer, x inner = OF order
    b = bump(X, Y).ravel()
    h = H_IN - b                             # flat free surface
    # start from the DEVELOPED uniform flow (hu = q_in everywhere, matching
    # the inflow BC): an impulsive start against a lake at rest drives a
    # bore that dries the crest (measured: pow(h,7/3) FPE at t=0.29)

    def field(fname, vals):
        body = "\n".join(f"{v:.10g}" for v in vals)
        w(C / "0" / fname, f"""
{HDR.format(c="volScalarField", o=fname)}
dimensions [0 0 0 0 0 0 0];
internalField nonuniform List<scalar> {len(vals)} ( {body} );
boundaryField {{
  {left} {{ type zeroGradient; }}
  {right} {{ type zeroGradient; }}
  sides {{ type zeroGradient; }}
  frontAndBack {{ type empty; }}
}}
""")
    field("Q0", b)
    field("Q1", h)
    field("Q2", np.full(nx * NY, Q_IN))
    field("Q3", np.zeros(nx * NY))


def precice_config():
    alldata = "\n  ".join(f'<data:scalar name="{d}"/>' for d in DATA_1 + DATA_2)
    use = "\n    ".join(f'<use-data name="{d}"/>' for d in DATA_1 + DATA_2)
    w1 = "\n      ".join(f'<write-data name="{d}" mesh="Swe1Mesh"/>' for d in DATA_1)
    r1 = "\n      ".join(f'<read-data name="{d}" mesh="Swe1Mesh"/>' for d in DATA_2)
    w2 = "\n      ".join(f'<write-data name="{d}" mesh="Swe2Mesh"/>' for d in DATA_2)
    r2 = "\n      ".join(f'<read-data name="{d}" mesh="Swe2Mesh"/>' for d in DATA_1)
    ex1 = "\n    ".join(f'<exchange data="{d}" mesh="Swe1Mesh" from="Swe1" to="Swe2"/>' for d in DATA_1)
    ex2 = "\n    ".join(f'<exchange data="{d}" mesh="Swe2Mesh" from="Swe2" to="Swe1"/>' for d in DATA_2)
    w(CFG, f"""
<?xml version="1.0" encoding="UTF-8" ?>
<precice-configuration>
  {alldata}

  <mesh name="Swe1Mesh" dimensions="3">
    {use}
  </mesh>
  <mesh name="Swe2Mesh" dimensions="3">
    {use}
  </mesh>

  <participant name="Swe1">
      <provide-mesh name="Swe1Mesh"/>
      <receive-mesh name="Swe2Mesh" from="Swe2"/>
      {w1}
      {r1}
      <mapping:nearest-neighbor direction="read" from="Swe2Mesh" to="Swe1Mesh" constraint="consistent"/>
  </participant>

  <participant name="Swe2">
      <provide-mesh name="Swe2Mesh"/>
      <receive-mesh name="Swe1Mesh" from="Swe1"/>
      {w2}
      {r2}
      <mapping:nearest-neighbor direction="read" from="Swe1Mesh" to="Swe2Mesh" constraint="consistent"/>
  </participant>

  <m2n:sockets acceptor="Swe1" connector="Swe2" exchange-directory="{RUN}"/>

  <coupling-scheme:parallel-explicit>
    <participants first="Swe1" second="Swe2"/>
    <max-time value="{T_END}"/>
    <time-window-size value="{WINDOW}"/>
    {ex1}
    {ex2}
  </coupling-scheme:parallel-explicit>
</precice-configuration>
""")


if __name__ == "__main__":
    RUN.mkdir(parents=True, exist_ok=True)
    case("mono",  0.0, LX,  2 * NXH, "inflow", "outflow")
    case("part1", 0.0, XMID, NXH, "inflow", "coupled",
         ("Swe1", "Swe1Mesh", DATA_1, DATA_2))
    case("part2", XMID, LX, NXH, "coupled", "outflow",
         ("Swe2", "Swe2Mesh", DATA_2, DATA_1))
    precice_config()
    print(f"swe2d_bump generated in {RUN}: [0,{LX}]x[0,{LY}] split at {XMID}, "
          f"bump at ({BX},{BY}) A={BA} sigma={BS}, h_in={H_IN} q_in={Q_IN}, "
          f"T={T_END} window={WINDOW}")
