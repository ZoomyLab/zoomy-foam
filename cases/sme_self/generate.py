#!/usr/bin/env python
"""Generate the SME(N)↔SME(N) SAME-LEVEL coupling case (1D Stoker dam break).

Control experiment for interface reflection: both participants run the SAME
model, so the monolithic reference is exact and ANY coupled-vs-mono deviation
is a pure coupling artifact (no model-impedance contribution).

Layout (case convention):
  part1/  zoomyFoam SME(N), x ∈ [0, 25], dam at 25 = the interface
  part2/  zoomyFoam SME(N), x ∈ [25, 50]
  mono/   zoomyFoam SME(N) monolithic reference, x ∈ [0, 50]
  precice-config.xml  canonical [b,h,u,v,w,p] two-way (scheme selectable)

Fixed dt on all three so the coupled pair and the monolithic reference share
one time grid (exact comparability — adaptive dt would confound the metric
with time-discretization differences).  preciceZSamples > 2·N so the
ζ-column exchange (interpolate_to_3d → project_from_3d) carries the moments
through the interface without truncation.

Usage: generate.py [LEVEL] [SCHEME] [ZSAMPLES]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# container-visible alias of the same bind-mounted dir (apptainer sees /Users/...)
CHERE = Path(str(HERE).replace("/mnt/userdrive/Users/home/", "/Users/"))
LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SCHEME = sys.argv[2] if len(sys.argv) > 2 else "parallel-explicit"
ZS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
DT = float(sys.argv[4]) if len(sys.argv) > 4 else 5e-3   # coupling window (exchange interval); solvers CFL-adapt below it
T_END = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

X_MIN, X_MID, X_MAX, N = 0.0, 25.0, 50.0, 200
NH = N // 2
H_L, H_R = 0.5, 0.1            # dam AT the interface (Stoker)
WRITE = DT if T_END <= 400*DT else 0.02   # step-resolved output for short probes
LAMBDA_S, NU = 0.5, 1e-3      # bottom friction: excites q_1..q_N
NQ = 3 + LEVEL
FIELDS = ["b", "h", "u", "v", "w", "p"]
DATA_1 = [f + "_1" for f in FIELDS]
DATA_2 = [f + "_2" for f in FIELDS]
CFG = CHERE / "precice-config.xml"
HDR = "FoamFile {{ format ascii; class {c}; object {o}; }}"


def w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip("\n"))


def case(name, xmin, xmax, n, patches, precice=None):
    C = HERE / name
    w(C / "system/blockMeshDict", f"""
{HDR.format(c="dictionary", o="blockMeshDict")}
convertToMeters 1;
vertices (
  ({xmin} 0 0) ({xmax} 0 0) ({xmax} 1 0) ({xmin} 1 0)
  ({xmin} 0 1) ({xmax} 0 1) ({xmax} 1 1) ({xmin} 1 1)
);
blocks ( hex (0 1 2 3 4 5 6 7) ({n} 1 1) simpleGrading (1 1 1) );
boundary (
  {patches}
  sides        {{ type empty; faces ((0 1 5 4) (3 7 6 2)); }}
  topAndBottom {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();
""")
    pc = ""
    if precice:
        pc = f"""
preciceParticipant {precice['participant']};
preciceConfig "{CFG}";
preciceMeshes ( {precice['mesh']} );
preciceWriteData ( {' '.join(precice['write'])} );
preciceReadData ( {' '.join(precice['read'])} );
preciceGhost fullstate;
preciceFrozenMass yes;
preciceZSamples {ZS};
"""
    w(C / "system/controlDict", f"""
{HDR.format(c="dictionary", o="controlDict")}
application zoomyFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {T_END};
deltaT {DT}; writeControl adjustableRunTime; writeInterval {WRITE};
purgeWrite 0; writeFormat ascii; writePrecision 12; timeFormat general;
runTimeModifiable true; adjustTimeStep no; maxCo 0.9; reconstructionOrder 1;
modelParameters {{ lambda_s {LAMBDA_S}; nu {NU}; }}
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
    w(C / "system/setFieldsDict", f"""
{HDR.format(c="dictionary", o="setFieldsDict")}
defaultFieldValues ( volScalarFieldValue Q1 {H_R} );
regions ( boxToCell {{ box ({X_MIN-1} -100 -100) ({X_MID} 100 100);
  fieldValues ( volScalarFieldValue Q1 {H_L} ); }} );
""")
    (C / "constant").mkdir(parents=True, exist_ok=True)
    bset = " ".join(f"{p} {{ type zeroGradient; }}"
                    for p in ([pp.split()[0] for pp in patches.split("}") if pp.strip()]))
    for qi in range(NQ):
        val = H_R if qi == 1 else 0.0
        w(C / "0" / f"Q{qi}", f"""
{HDR.format(c="volScalarField", o=f"Q{qi}")}
dimensions [0 0 0 0 0 0 0]; internalField uniform {val};
boundaryField {{
  {bset}
  sides {{ type empty; }}
  topAndBottom {{ type empty; }}
}}
""")


case("part1", X_MIN, X_MID, NH,
     'outer        { type patch; faces ((0 4 7 3)); }\n'
     '  coupled      { type patch; faces ((1 2 6 5)); }',
     precice=dict(participant="SmeA", mesh="MeshA",
                  write=DATA_1, read=DATA_2))
case("part2", X_MID, X_MAX, NH,
     'coupled      { type patch; faces ((0 4 7 3)); }\n'
     '  outer        { type patch; faces ((1 2 6 5)); }',
     precice=dict(participant="SmeB", mesh="MeshB",
                  write=DATA_2, read=DATA_1))
case("mono", X_MIN, X_MAX, N,
     'outer        { type patch; faces ((0 4 7 3) (1 2 6 5)); }')

alldata = "\n  ".join(f'<data:scalar name="{d}"/>' for d in DATA_1 + DATA_2)
use = "\n    ".join(f'<use-data name="{d}"/>' for d in DATA_1 + DATA_2)
w1 = "\n      ".join(f'<write-data name="{d}" mesh="MeshA"/>' for d in DATA_1)
r1 = "\n      ".join(f'<read-data name="{d}" mesh="MeshA"/>' for d in DATA_2)
w2 = "\n      ".join(f'<write-data name="{d}" mesh="MeshB"/>' for d in DATA_2)
r2 = "\n      ".join(f'<read-data name="{d}" mesh="MeshB"/>' for d in DATA_1)
ex1 = "\n    ".join(f'<exchange data="{d}" mesh="MeshA" from="SmeA" to="SmeB" initialize="true"/>' for d in DATA_1)
ex2 = "\n    ".join(f'<exchange data="{d}" mesh="MeshB" from="SmeB" to="SmeA" initialize="true"/>' for d in DATA_2)
if SCHEME == "parallel-explicit":
    cs = f"""<coupling-scheme:parallel-explicit>
    <participants first="SmeA" second="SmeB"/>
    <max-time value="{T_END}"/>
    <time-window-size value="{DT}"/>
    {ex1}
    {ex2}
  </coupling-scheme:parallel-explicit>"""
else:
    cs = f"""<coupling-scheme:parallel-implicit>
    <participants first="SmeA" second="SmeB"/>
    <max-time value="{T_END}"/>
    <time-window-size value="{DT}"/>
    <max-iterations value="5"/>
    <relative-convergence-measure data="h_1" mesh="MeshA" limit="1e-6"/>
    <relative-convergence-measure data="h_2" mesh="MeshB" limit="1e-6"/>
    {ex1}
    {ex2}
    <acceleration:constant><relaxation value="0.5"/></acceleration:constant>
  </coupling-scheme:parallel-implicit>"""
w(HERE / "precice-config.xml", f"""
<?xml version="1.0" encoding="UTF-8" ?>
<precice-configuration>
  {alldata}

  <mesh name="MeshA" dimensions="3">
    {use}
  </mesh>
  <mesh name="MeshB" dimensions="3">
    {use}
  </mesh>

  <participant name="SmeA">
      <provide-mesh name="MeshA"/>
      <receive-mesh name="MeshB" from="SmeB"/>
      {w1}
      {r1}
      <mapping:nearest-neighbor direction="read" from="MeshB" to="MeshA" constraint="consistent"/>
  </participant>

  <participant name="SmeB">
      <provide-mesh name="MeshB"/>
      <receive-mesh name="MeshA" from="SmeA"/>
      {w2}
      {r2}
      <mapping:nearest-neighbor direction="read" from="MeshA" to="MeshB" constraint="consistent"/>
  </participant>

  <m2n:sockets acceptor="SmeA" connector="SmeB" exchange-directory="{CHERE}"/>

  {cs}
</precice-configuration>
""")
print(f"generated sme_self L{LEVEL} ({SCHEME}, zsamples={ZS}) in {HERE}")
