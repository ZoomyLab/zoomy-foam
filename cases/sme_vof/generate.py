#!/usr/bin/env python
"""Generate the two-way SME<->VOF coupling run:
  RUN/swe_case/  : zoomyFoam 1-D SME dam break (outer-left, coupled-right)
  RUN/vof_case/  : incompressibleVoF wave tank (coupled inlet; wall right in
                   pair mode, coupled outlet -> second SME in triple mode)
  RUN/swe2_case/ : (triple mode) second 1-D SME, coupled-left, outer-right
  RUN/precice-config.xml : canonical [b,h,u,v,w,p] exchange
Both/all participants run from RUN (shared exchange dir, absolute config path).

Args   : LEVEL WINDOW SCHEME GHOST FROZEN LEDGER   (positional, as before)
Env    : RUNDIR     run folder override (parallel batches)
         NZ         VOF transverse cells (default 1 = 2D-thin; >1 = real 3D)
         TRANSVERSE front/back BC for NZ>1: cyclic (default) | wall
         MODE       pair (default) | triple (SME-VOF-SME, coupling-scheme:multi)
"""
import shutil
import os
import sys
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
VOF_LX, VOF_LY, VOF_NX, VOF_NY = 1.5, 0.4, 120, 40
VOF_NZ = int(os.environ.get("NZ", "1"))
VOF_LZ = 0.02 if VOF_NZ == 1 else 0.01*VOF_NZ      # dz = dy for real 3D
TRANSVERSE = os.environ.get("TRANSVERSE", "cyclic")  # cyclic | wall (NZ>1)
MODE = os.environ.get("MODE", "pair")                # pair | triple
SWE2_XMAX = VOF_LX + 0.6                             # triple: right SME domain
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

HDR = "FoamFile {{ format ascii; class {c}; object {o}; }}"


def w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip("\n"))


# ════════════════════ SME cases (zoomyFoam) ══════════════════════════════════
def swe_case(name, xmin, xmax, coupled_side, participant, mesh_name,
             dam=None):
    """One 1-D SME participant.  coupled_side: 'right' | 'left'."""
    C = RUN / name
    left_patch = "outer" if coupled_side == "right" else "coupled"
    right_patch = "coupled" if coupled_side == "right" else "outer"
    w(C / "system/blockMeshDict", f"""
{HDR.format(c="dictionary", o="blockMeshDict")}
convertToMeters 1;
vertices (
  ({xmin} 0 0) ({xmax} 0 0) ({xmax} 1 0) ({xmin} 1 0)
  ({xmin} 0 1) ({xmax} 0 1) ({xmax} 1 1) ({xmin} 1 1)
);
blocks ( hex (0 1 2 3 4 5 6 7) ({SWE_N} 1 1) simpleGrading (1 1 1) );
boundary (
  {left_patch}  {{ type patch; faces ((0 4 7 3)); }}
  {right_patch} {{ type patch; faces ((1 2 6 5)); }}
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
preciceParticipant {participant};
preciceConfig "{CFG}";
preciceMeshes ( {mesh_name} );
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
    if dam is not None:
        w(C / "system/setFieldsDict", f"""
{HDR.format(c="dictionary", o="setFieldsDict")}
defaultFieldValues ( volScalarFieldValue Q1 {H_R} );
regions ( boxToCell {{ box ({xmin-1} -100 -100) ({dam} 100 100);
  fieldValues ( volScalarFieldValue Q1 {H_L} ); }} );
""")
    (C / "constant").mkdir(parents=True, exist_ok=True)

    def field(fname, val):
        w(C / "0" / fname, f"""
{HDR.format(c="volScalarField", o=fname)}
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
    # template provides constant/ (g, transport, phases) + fvSchemes/fvSolution;
    # mesh, ICs and BCs are GENERATED (no donor-mesh dependency).
    shutil.copytree(VOF_SRC, C, ignore=shutil.ignore_patterns(
        "0", "0.[0-9]*", "[1-9]*", "polyMesh", "precice-run", "precice-*",
        "log.*", "dynamicCode", "0.orig", "processor*", "blockMeshDict",
        "setFieldsDict"))

    triple = (MODE == "triple")
    right_name = "outlet" if triple else "wall_right"
    right_type = "patch" if triple else "wall"
    if VOF_NZ == 1:
        fb_mesh = "frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }"
        fb_U = fb_s = fb_p = "frontAndBack { type empty; }"
    elif TRANSVERSE == "cyclic":
        fb_mesh = ("front { type cyclic; neighbourPatch back;"
                   " faces ((0 3 2 1)); }\n"
                   "  back  { type cyclic; neighbourPatch front;"
                   " faces ((4 5 6 7)); }")
        fb_U = fb_s = fb_p = "front { type cyclic; } back { type cyclic; }"
    else:   # wall flume
        fb_mesh = ("front { type wall; faces ((0 3 2 1)); }\n"
                   "  back  { type wall; faces ((4 5 6 7)); }")
        fb_U = "front { type noSlip; } back { type noSlip; }"
        fb_s = "front { type zeroGradient; } back { type zeroGradient; }"
        fb_p = ("front { type fixedFluxPressure; value uniform 0; } "
                "back { type fixedFluxPressure; value uniform 0; }")

    w(C / "system/blockMeshDict", f"""
{HDR.format(c="dictionary", o="blockMeshDict")}
convertToMeters 1;
vertices (
  (0 0 0) ({VOF_LX} 0 0) ({VOF_LX} {VOF_LY} 0) (0 {VOF_LY} 0)
  (0 0 {VOF_LZ}) ({VOF_LX} 0 {VOF_LZ}) ({VOF_LX} {VOF_LY} {VOF_LZ}) (0 {VOF_LY} {VOF_LZ})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({VOF_NX} {VOF_NY} {VOF_NZ}) simpleGrading (1 1 1) );
boundary (
  inlet        {{ type patch; faces ((0 4 7 3)); }}
  {right_name} {{ type {right_type};  faces ((1 2 6 5)); }}
  bottom       {{ type wall;  faces ((0 1 5 4)); }}
  atmosphere   {{ type patch; faces ((3 7 6 2)); }}
  {fb_mesh}
);
mergePatchPairs ();
""")
    w(C / "system/setFieldsDict", f"""
{HDR.format(c="dictionary", o="setFieldsDict")}
defaultFieldValues ( volScalarFieldValue alpha.water 0 );
regions ( boxToCell {{ box (-1 -1 -1) ({VOF_LX+1} {H_R} {VOF_LZ+1});
  fieldValues ( volScalarFieldValue alpha.water 1 ); }} );
""")
    # 0/ fields.  Coupled patches (inlet [+ outlet in triple]) are fixedValue
    # so the FO can impose; everything else is the standard VoF tank set.
    coupled_U = "{ type fixedValue; value uniform (0 0 0); }"
    coupled_a = "{ type fixedValue; value uniform 0; }"
    coupled_p = "{ type fixedFluxPressure; value uniform 0; }"
    right_U = coupled_U if triple else "{ type noSlip; }"
    right_a = coupled_a if triple else "{ type zeroGradient; }"
    right_p = coupled_p if triple else \
        "{ type fixedFluxPressure; value uniform 0; }"
    w(C / "0/U", f"""
{HDR.format(c="volVectorField", o="U")}
dimensions [0 1 -1 0 0 0 0]; internalField uniform (0 0 0);
boundaryField {{
  inlet      {coupled_U}
  {right_name} {right_U}
  bottom     {{ type noSlip; }}
  atmosphere {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}
  {fb_U}
}}
""")
    w(C / "0/alpha.water", f"""
{HDR.format(c="volScalarField", o="alpha.water")}
dimensions []; internalField uniform 0;
boundaryField {{
  inlet      {coupled_a}
  {right_name} {right_a}
  bottom     {{ type zeroGradient; }}
  atmosphere {{ type inletOutlet; inletValue uniform 0; value uniform 0; }}
  {fb_s}
}}
""")
    w(C / "0/p_rgh", f"""
{HDR.format(c="volScalarField", o="p_rgh")}
dimensions [1 -1 -2 0 0 0 0]; internalField uniform 0;
boundaryField {{
  inlet      {coupled_p}
  {right_name} {right_p}
  bottom     {{ type fixedFluxPressure; value uniform 0; }}
  atmosphere {{ type totalPressure; p0 uniform 0; value uniform 0; }}
  {fb_p}
}}
""")
    if triple:
        iface = f"""interfaces
    {{
      left  {{ patch inlet;  mesh VofInletMesh;  domainHeight {VOF_LY}; }}
      right {{ patch outlet; mesh VofOutletMesh; domainHeight {VOF_LY}; }}
    }}"""
    else:
        iface = f"patch inlet; preciceMesh VofInletMesh; domainHeight {VOF_LY};"
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
    {iface}
    relax 1.0; outputInterval 0.05;
    maxCo 0.45; maxAlphaCo 0.45;   // FO-owned adaptive dt below the window
    debtRepayWindows {LEDGER}; ledgerLog yes; writeColumns yes;
    preciceConfig "{CFG}"; preciceParticipant Vof;
    preciceReadData  ( {' '.join(DATA_S)} );
    preciceWriteData ( {' '.join(DATA_V)} );
  }}
}}
""")


# ════════════════════ precice-config ═════════════════════════════════════════
def precice_config():
    triple = (MODE == "triple")
    meshes = [("SweMesh", "Swe", "VofInletMesh")]
    if triple:
        meshes.append(("Swe2Mesh", "Swe2", "VofOutletMesh"))

    alldata = "\n  ".join(f'<data:scalar name="{d}"/>' for d in DATA_S + DATA_V)
    use = "\n    ".join(f'<use-data name="{d}"/>' for d in DATA_S + DATA_V)

    mesh_xml = "\n".join(f"""  <mesh name="{m}" dimensions="3">
    {use}
  </mesh>""" for pair in meshes for m in (pair[0], pair[2]))

    # reduced participants
    parts = []
    for sm, sp, vm in meshes:
        wS = "\n      ".join(f'<write-data name="{d}" mesh="{sm}"/>' for d in DATA_S)
        rS = "\n      ".join(f'<read-data name="{d}" mesh="{sm}"/>' for d in DATA_V)
        parts.append(f"""  <participant name="{sp}">
      <provide-mesh name="{sm}"/>
      <receive-mesh name="{vm}" from="Vof"/>
      {wS}
      {rS}
      <mapping:nearest-neighbor direction="read" from="{vm}" to="{sm}" constraint="consistent"/>
  </participant>""")
    # the VOF participant provides every interface mesh
    vof_lines = []
    for sm, sp, vm in meshes:
        vof_lines += [f'<provide-mesh name="{vm}"/>',
                      f'<receive-mesh name="{sm}" from="{sp}"/>']
        vof_lines += [f'<write-data name="{d}" mesh="{vm}"/>' for d in DATA_V]
        vof_lines += [f'<read-data name="{d}" mesh="{vm}"/>' for d in DATA_S]
        vof_lines += [f'<mapping:nearest-neighbor direction="read" '
                      f'from="{sm}" to="{vm}" constraint="consistent"/>']
    parts.append("  <participant name=\"Vof\">\n      "
                 + "\n      ".join(vof_lines) + "\n  </participant>")

    m2n = "\n  ".join(
        f'<m2n:sockets acceptor="{sp}" connector="Vof" exchange-directory="{RUN}"/>'
        for _, sp, _ in meshes)

    def exchanges(sm, sp, vm):
        exS = "\n    ".join(
            f'<exchange data="{d}" mesh="{sm}" from="{sp}" to="Vof"/>'
            for d in DATA_S)
        exV = "\n    ".join(
            f'<exchange data="{d}" mesh="{vm}" from="Vof" to="{sp}"/>'
            for d in DATA_V)
        return exS + "\n    " + exV

    if triple:
        # one multi scheme, VOF = controller (the hub of the star topology)
        ex = "\n    ".join(exchanges(*m) for m in meshes)
        ctrl = "\n    ".join(
            [f'<participant name="{sp}"/>' for _, sp, _ in meshes]
            + ['<participant name="Vof" control="yes"/>'])
        # multi is implicit-class in preCICE; max-iterations 1 makes it the
        # explicit-equivalent single pass (parallel-explicit-first policy).
        cs = f"""<coupling-scheme:multi>
    {ctrl}
    <max-time value="{T_END}"/>
    <time-window-size value="{WINDOW}"/>
    <max-iterations value="1"/>
    {ex}
  </coupling-scheme:multi>"""
    elif SCHEME == "parallel-explicit":
        cs = f"""<coupling-scheme:parallel-explicit>
    <participants first="Swe" second="Vof"/>
    <max-time value="{T_END}"/>
    <time-window-size value="{WINDOW}"/>
    {exchanges(*meshes[0])}
  </coupling-scheme:parallel-explicit>"""
    else:
        cs = f"""<coupling-scheme:parallel-implicit>
    <participants first="Swe" second="Vof"/>
    <max-time value="{T_END}"/>
    <time-window-size value="{WINDOW}"/>
    <max-iterations value="30"/>
    <relative-convergence-measure data="h_V" mesh="VofInletMesh" limit="1e-8"/>
    <relative-convergence-measure data="u_V" mesh="VofInletMesh" limit="1e-7"/>
    {exchanges(*meshes[0])}
    <acceleration:constant><relaxation value="0.5"/></acceleration:constant>
  </coupling-scheme:parallel-implicit>"""

    w(CFG, f"""
<?xml version="1.0" encoding="UTF-8" ?>
<precice-configuration>
  {alldata}

{mesh_xml}

{chr(10).join(parts)}

  {m2n}

  {cs}
</precice-configuration>
""")


if __name__ == "__main__":
    RUN.mkdir(parents=True, exist_ok=True)
    swe_case("swe_case", SWE_XMIN, SWE_XMAX, "right", "Swe", "SweMesh",
             dam=X_DAM)
    if MODE == "triple":
        swe_case("swe2_case", VOF_LX, SWE2_XMAX, "left", "Swe2", "Swe2Mesh")
    vof_case()
    precice_config()
    print(f"two-way run generated in {RUN} (SME level {SME_LEVEL}, {SCHEME}, "
          f"mode={MODE}, NZ={VOF_NZ}{'' if VOF_NZ == 1 else ' ' + TRANSVERSE})")
    print(f"  SWE [{SWE_XMIN},{SWE_XMAX}] dam@{X_DAM} ({H_L}->{H_R}) | "
          f"VOF [0,{VOF_LX}]x{VOF_LZ} "
          f"{'-> SWE2 [' + str(VOF_LX) + ',' + str(SWE2_XMAX) + ']' if MODE == 'triple' else 'wall-right'}"
          f" | T={T_END} window={WINDOW}")
