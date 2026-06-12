#!/usr/bin/env python
"""Generate the SME(0)↔SME(1) inter-level coupling case (1D Stoker dam break).

Layout (per the case convention):
  part1/  zoomyFoam SME(level 0), x ∈ [0, 25], dam at 25 = the interface
  part2/  zoomyFoam SME(level 1), x ∈ [25, 50]
  mono/   zoomyFoam SME(level 0) monolithic reference, x ∈ [0, 50]
  precice-config.xml  canonical [b,h,u,v,w,p] two-way (scheme selectable)
  run.sh  emit both levels, build + stash both binaries, mesh, run all three

The two participants have different n_dof_q → different binaries; they exchange
the CANONICAL interface fields (model-owned interpolate_to_3d / project_from_3d),
not the raw state. Inviscid dam break ⇒ q_1 stays 0 ⇒ the joined solution must
match the monolithic SME(0) reference (and Stoker).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEME = sys.argv[1] if len(sys.argv) > 1 else "parallel-explicit"

X_MIN, X_MID, X_MAX, N = 0.0, 25.0, 50.0, 200
NH = N // 2
H_L, H_R = 0.5, 0.1            # dam AT the interface (Stoker, same as self_coupling)
T_END, DT = 1.0, 5e-4
WRITE = 0.02
LAMBDA_S, NU = 0.5, 1e-3      # bottom friction: slip + viscosity (excites q_1)
FIELDS = ["b", "h", "u", "v", "w", "p"]
DATA_1 = [f + "_1" for f in FIELDS]    # written by part1 (L0)
DATA_2 = [f + "_2" for f in FIELDS]    # written by part2 (L1)
CFG = HERE / "precice-config.xml"
HDR = "FoamFile {{ format ascii; class {c}; object {o}; }}"


def w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip("\n"))


def case(name, xmin, xmax, n, nq, patches, precice=None):
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
preciceZSamples 1;
"""
    w(C / "system/controlDict", f"""
{HDR.format(c="dictionary", o="controlDict")}
application zoomyFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {T_END};
deltaT {DT}; writeControl adjustableRunTime; writeInterval {WRITE};
purgeWrite 0; writeFormat ascii; writePrecision 12; timeFormat general;
runTimeModifiable true; adjustTimeStep no; maxCo 0.4; reconstructionOrder 1;
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
    for qi in range(nq):
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


# part1: SME(0), interface on the right
case("part1", X_MIN, X_MID, NH, 3,
     'outer        { type patch; faces ((0 4 7 3)); }\n'
     '  coupled      { type patch; faces ((1 2 6 5)); }',
     precice=dict(participant="Sme0", mesh="Mesh0",
                  write=DATA_1, read=DATA_2))
# part2: SME(1), interface on the left
case("part2", X_MID, X_MAX, NH, 4,
     'coupled      { type patch; faces ((0 4 7 3)); }\n'
     '  outer        { type patch; faces ((1 2 6 5)); }',
     precice=dict(participant="Sme1", mesh="Mesh1",
                  write=DATA_2, read=DATA_1))
# mono: SME(0) and SME(1) references over the full domain (bracket the joined run)
case("mono", X_MIN, X_MAX, N, 3,
     'outer        { type patch; faces ((0 4 7 3) (1 2 6 5)); }')
case("mono_l1", X_MIN, X_MAX, N, 4,
     'outer        { type patch; faces ((0 4 7 3) (1 2 6 5)); }')

# precice config — canonical two-way
alldata = "\n  ".join(f'<data:scalar name="{d}"/>' for d in DATA_1 + DATA_2)
use = "\n    ".join(f'<use-data name="{d}"/>' for d in DATA_1 + DATA_2)
w1 = "\n      ".join(f'<write-data name="{d}" mesh="Mesh0"/>' for d in DATA_1)
r1 = "\n      ".join(f'<read-data name="{d}" mesh="Mesh0"/>' for d in DATA_2)
w2 = "\n      ".join(f'<write-data name="{d}" mesh="Mesh1"/>' for d in DATA_2)
r2 = "\n      ".join(f'<read-data name="{d}" mesh="Mesh1"/>' for d in DATA_1)
ex1 = "\n    ".join(f'<exchange data="{d}" mesh="Mesh0" from="Sme0" to="Sme1"/>' for d in DATA_1)
ex2 = "\n    ".join(f'<exchange data="{d}" mesh="Mesh1" from="Sme1" to="Sme0"/>' for d in DATA_2)
if SCHEME == "parallel-explicit":
    cs = f"""<coupling-scheme:parallel-explicit>
    <participants first="Sme0" second="Sme1"/>
    <max-time value="{T_END}"/>
    <time-window-size value="{DT}"/>
    {ex1}
    {ex2}
  </coupling-scheme:parallel-explicit>"""
else:
    cs = f"""<coupling-scheme:parallel-implicit>
    <participants first="Sme0" second="Sme1"/>
    <max-time value="{T_END}"/>
    <time-window-size value="{DT}"/>
    <max-iterations value="50"/>
    <relative-convergence-measure data="h_1" mesh="Mesh0" limit="1e-6"/>
    <relative-convergence-measure data="h_2" mesh="Mesh1" limit="1e-6"/>
    {ex1}
    {ex2}
    <acceleration:constant><relaxation value="0.5"/></acceleration:constant>
  </coupling-scheme:parallel-implicit>"""
w(CFG, f"""
<?xml version="1.0" encoding="UTF-8" ?>
<precice-configuration>
  {alldata}

  <mesh name="Mesh0" dimensions="3">
    {use}
  </mesh>
  <mesh name="Mesh1" dimensions="3">
    {use}
  </mesh>

  <participant name="Sme0">
      <provide-mesh name="Mesh0"/>
      <receive-mesh name="Mesh1" from="Sme1"/>
      {w1}
      {r1}
      <mapping:nearest-neighbor direction="read" from="Mesh1" to="Mesh0" constraint="consistent"/>
  </participant>

  <participant name="Sme1">
      <provide-mesh name="Mesh1"/>
      <receive-mesh name="Mesh0" from="Sme0"/>
      {w2}
      {r2}
      <mapping:nearest-neighbor direction="read" from="Mesh0" to="Mesh1" constraint="consistent"/>
  </participant>

  <m2n:sockets acceptor="Sme0" connector="Sme1" exchange-directory="{HERE}"/>

  {cs}
</precice-configuration>
""")
print(f"generated sme0_sme1 case ({SCHEME}) in {HERE}")
