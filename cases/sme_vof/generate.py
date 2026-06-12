#!/usr/bin/env python
"""Generate the two-way SWE<->VOF coupling run:
  RUN/swe_case/  : zoomyFoam 1-D SWE dam break (outer-left, coupled-right)
  RUN/vof_case/  : incompressibleVoF wave tank (coupled inlet, wall right)
  RUN/precice-config.xml : canonical [b,h,u,v,w,p] two-way, serial-explicit
Both participants run from RUN (shared exchange dir, absolute config path).
"""
import shutil
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
# container-visible alias (apptainer sees /Users/..., not /mnt/userdrive/...)
CHERE = Path(str(HERE).replace("/mnt/userdrive/Users/home/", "/Users/"))
# RUNDIR env overrides the run folder so independent runs go in PARALLEL
# (each run dir is self-contained: cases, precice config, exchange dir).
RUN = Path(os.environ.get("RUNDIR", str(CHERE / "run")))
VOF_SRC = CHERE / "vof_template"
CFG = RUN / "precice-config.xml"

# ── geometry / physics ──────────────────────────────────────────────────────
SWE_XMIN, SWE_XMAX, SWE_N = -0.6, 0.0, 120
X_DAM, H_L, H_R = -0.4, 0.18, 0.10
VOF_LX, VOF_LY, VOF_NX, VOF_NY, VOF_NZ = 1.5, 0.4, 120, 40, 1   # = vof_template blockMeshDict
import sys
SME_LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # reduced-model moment level
T_END = 4.0
# coupling window. Both solvers run adaptive CFL inside it (SME maxCo 0.9 and
# subcycles if its CFL dt drops below the window; VOF stable dt >> window) so
# the effective dt is min(CFL_SME, CFL_VOF, window).  parallel schemes in
# preCICE require a FIXED window (first-participant sizing is serial-only).
WINDOW = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-3
SCHEME = sys.argv[3] if len(sys.argv) > 3 else "parallel-explicit"  # | parallel-implicit
GHOST = sys.argv[4] if len(sys.argv) > 4 else "fullstate"           # | characteristic
FROZEN = sys.argv[5] if len(sys.argv) > 5 else "auto"   # auto: yes for explicit
LEDGER = sys.argv[6] if len(sys.argv) > 6 else "20"     # debt repayment windows; 1e18 = off
if FROZEN == "auto":
    FROZEN = "yes" if SCHEME == "parallel-explicit" else "no"
DT0 = 5e-4                 # initial solver dt (SME adapts up from here)
LAMBDA_S, NU = 0.5, 1e-3   # SWE bottom friction (VOF floor is noSlip -> consistency)
# zeta-column contract: ALWAYS exchange the full profile (one sample per VOF
# inlet face).  The old level-0 single-sample shortcut made the SME's ghost
# read the MID-COLUMN velocity u(0.5) instead of the depth average — a
# +2.5e-3 q* stream gap in sheared outflow (the last conservation leak).
NZ_SAMPLES = VOF_NY
FIELDS = ["b", "h", "u", "v", "w", "p"]
DATA_S = [f + "_S" for f in FIELDS]
DATA_V = [f + "_V" for f in FIELDS]


def w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip("\n"))


# ════════════════════ SWE case (zoomyFoam) ══════════════════════════════════
def swe_case():
    C = RUN / "swe_case"
    HDR = "FoamFile {{ format ascii; class {c}; object {o}; }}"
    w(C / "system/blockMeshDict", f"""
{HDR.format(c="dictionary", o="blockMeshDict")}
convertToMeters 1;
vertices (
  ({SWE_XMIN} 0 0) ({SWE_XMAX} 0 0) ({SWE_XMAX} 1 0) ({SWE_XMIN} 1 0)
  ({SWE_XMIN} 0 1) ({SWE_XMAX} 0 1) ({SWE_XMAX} 1 1) ({SWE_XMIN} 1 1)
);
blocks ( hex (0 1 2 3 4 5 6 7) ({SWE_N} 1 1) simpleGrading (1 1 1) );
boundary (
  outer        {{ type patch; faces ((0 4 7 3)); }}
  coupled      {{ type patch; faces ((1 2 6 5)); }}
  sides        {{ type empty; faces ((0 1 5 4) (3 7 6 2)); }}
  topAndBottom {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();
""")
    w(C / "system/controlDict", f"""
{HDR.format(c="dictionary", o="controlDict")}
application zoomyFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {T_END};
deltaT {DT0}; writeControl adjustableRunTime; writeInterval 0.05;
purgeWrite 0; writeFormat ascii; writePrecision 10; timeFormat general;
runTimeModifiable true; adjustTimeStep yes; maxCo 0.9; reconstructionOrder 1;
modelParameters {{ lambda_s {LAMBDA_S}; nu {NU}; }}
preciceParticipant Swe;
preciceConfig "{CFG}";
preciceMeshes ( SweMesh );
preciceWriteData ( {' '.join(DATA_S)} );
preciceReadData ( {' '.join(DATA_V)} );
preciceGhost {GHOST};
preciceZSamples {NZ_SAMPLES};
preciceFrozenMass {FROZEN};
""")
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
regions ( boxToCell {{ box ({SWE_XMIN-1} -100 -100) ({X_DAM} 100 100);
  fieldValues ( volScalarFieldValue Q1 {H_L} ); }} );
""")
    (C / "constant").mkdir(parents=True, exist_ok=True)

    def field(name, val):
        w(C / "0" / name, f"""
{HDR.format(c="volScalarField", o=name)}
dimensions [0 0 0 0 0 0 0]; internalField uniform {val};
boundaryField {{
  outer {{ type zeroGradient; }}
  coupled {{ type zeroGradient; }}
  sides {{ type empty; }}
  topAndBottom {{ type empty; }}
}}
""")
    field("Q0", 0.0)            # b
    field("Q1", H_R)            # h (setFields adds the dam step)
    for qi in range(2, 3 + SME_LEVEL):
        field(f"Q{qi}", 0.0)    # q_0 .. q_level


# ════════════════════ VOF case (incompressibleVoF) ═══════════════════════════
def vof_case():
    C = RUN / "vof_case"
    if C.exists():
        shutil.rmtree(C)
    shutil.copytree(VOF_SRC, C, ignore=shutil.ignore_patterns(
        "0.[0-9]*", "[1-9]*", "precice-run", "precice-*", "log.*",
        "dynamicCode", "0.orig", "processor*"))   # keep 0/ ; drop time/run dirs
    # rewrite the controlDict functions block for two-way canonical exchange
    w(C / "system/controlDict", f"""
FoamFile {{ format ascii; class dictionary; object controlDict; }}
application foamRun; solver incompressibleVoF;
startFrom startTime; startTime 0; stopAt endTime; endTime {T_END};
deltaT {WINDOW}; writeControl timeStep; writeInterval 1000000;
purgeWrite 0; writeFormat ascii; writePrecision 8; timeFormat general;
runTimeModifiable yes; adjustTimeStep no;   // FO owns the CFL step
functions
{{
  sweCoupling
  {{
    type swePreciceCoupling; libs ("libswePreciceCoupling.so");
    patch inlet; domainHeight {VOF_LY}; relax 1.0; outputInterval 0.05;
    maxCo 0.45; maxAlphaCo 0.45;   // FO-owned adaptive dt below the window
    debtRepayWindows {LEDGER}; ledgerLog yes; writeColumns yes;
    preciceConfig "{CFG}"; preciceParticipant Vof; preciceMesh VofInletMesh;
    preciceReadData  ( {' '.join(DATA_S)} );
    preciceWriteData ( {' '.join(DATA_V)} );
  }}
}}
""")


# ════════════════════ precice-config (canonical two-way) ═════════════════════
def precice_config():
    alldata = "\n  ".join(f'<data:scalar name="{d}"/>' for d in DATA_S + DATA_V)
    use = "\n    ".join(f'<use-data name="{d}"/>' for d in DATA_S + DATA_V)
    wS = "\n      ".join(f'<write-data name="{d}" mesh="SweMesh"/>' for d in DATA_S)
    rS = "\n      ".join(f'<read-data name="{d}" mesh="SweMesh"/>' for d in DATA_V)
    wV = "\n      ".join(f'<write-data name="{d}" mesh="VofInletMesh"/>' for d in DATA_V)
    rV = "\n      ".join(f'<read-data name="{d}" mesh="VofInletMesh"/>' for d in DATA_S)
    exS = "\n    ".join(f'<exchange data="{d}" mesh="SweMesh" from="Swe" to="Vof"/>' for d in DATA_S)
    exV = "\n    ".join(f'<exchange data="{d}" mesh="VofInletMesh" from="Vof" to="Swe"/>' for d in DATA_V)
    if SCHEME == "parallel-explicit":
        cs = f"""<coupling-scheme:parallel-explicit>
    <participants first="Swe" second="Vof"/>
    <max-time value="{T_END}"/>
    <time-window-size value="{WINDOW}"/>
    {exS}
    {exV}
  </coupling-scheme:parallel-explicit>"""
    else:
        cs = f"""<coupling-scheme:parallel-implicit>
    <participants first="Swe" second="Vof"/>
    <max-time value="{T_END}"/>
    <time-window-size value="{WINDOW}"/>
    <max-iterations value="30"/>
    <relative-convergence-measure data="h_V" mesh="VofInletMesh" limit="1e-8"/>
    <relative-convergence-measure data="u_V" mesh="VofInletMesh" limit="1e-7"/>
    {exS}
    {exV}
    <acceleration:constant><relaxation value="0.5"/></acceleration:constant>
  </coupling-scheme:parallel-implicit>"""
    w(CFG, f"""
<?xml version="1.0" encoding="UTF-8" ?>
<precice-configuration>
  {alldata}

  <mesh name="SweMesh" dimensions="3">
    {use}
  </mesh>
  <mesh name="VofInletMesh" dimensions="3">
    {use}
  </mesh>

  <participant name="Swe">
      <provide-mesh name="SweMesh"/>
      <receive-mesh name="VofInletMesh" from="Vof"/>
      {wS}
      {rS}
      <mapping:nearest-neighbor direction="read" from="VofInletMesh" to="SweMesh" constraint="consistent"/>
  </participant>

  <participant name="Vof">
      <provide-mesh name="VofInletMesh"/>
      <receive-mesh name="SweMesh" from="Swe"/>
      {wV}
      {rV}
      <mapping:nearest-neighbor direction="read" from="SweMesh" to="VofInletMesh" constraint="consistent"/>
  </participant>

  <m2n:sockets acceptor="Swe" connector="Vof" exchange-directory="{RUN}"/>

  {cs}
</precice-configuration>
""")


if __name__ == "__main__":
    RUN.mkdir(parents=True, exist_ok=True)
    swe_case(); vof_case(); precice_config()
    print(f"two-way run generated in {RUN} (SME level {SME_LEVEL}, {SCHEME})")
    print(f"  SWE [{SWE_XMIN},{SWE_XMAX}] dam@{X_DAM} ({H_L}->{H_R}) | "
          f"VOF [0,{VOF_LX}] wall-right | T={T_END} window={WINDOW}")
