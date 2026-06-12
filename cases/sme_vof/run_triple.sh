#!/usr/bin/env bash
# SME -> VOF -> SME (three participants, coupling-scheme:multi, closed box).
# Usage: run_triple.sh LEVEL [SNAP_SUFFIX] [WINDOW]
# Env  : NZ=1 TRANSVERSE=cyclic
set -e
LEVEL=${1:?level}
SNAP=${2:-}
WINDOW=${3:-2e-3}
HERE="$(cd "$(dirname "$0")" && pwd)"
SIF=/Users/adam-obbpb5az1dhsjzf/of_build/zoomy_openfoam.sif
RUN=$HERE/run${SNAP:+_$SNAP}
export RUNDIR=$RUN
export MODE=triple
BIN=$HERE/../sme0_sme1/bin/zoomyFoam_L${LEVEL}w   # wall outer (closed system)
PY=/Users/adam-obbpb5az1dhsjzf/micromamba/envs/zoomy/bin/python

rm -rf "$RUN"
$PY "$HERE/generate.py" "$LEVEL" "$WINDOW" parallel-explicit fullstate yes 20

T0=$(date +%s.%N)
apptainer exec "$SIF" bash -c "
set --
set +e; source /opt/openfoam13/etc/bashrc; set -e
for c in swe_case swe2_case vof_case; do
  cd $RUN/\$c
  blockMesh > log.blockMesh 2>&1
  [ -f system/setFieldsDict ] && setFields > log.setFields 2>&1
done
cd $RUN
$BIN -case swe_case  > log.swe  2>&1 &  P1=\$!
$BIN -case swe2_case > log.swe2 2>&1 &  P2=\$!
foamRun -case vof_case > log.vof 2>&1 &  P3=\$!
wait \$P1; R1=\$?
wait \$P2; R2=\$?
wait \$P3; R3=\$?
echo \"swe rc=\$R1 swe2 rc=\$R2 vof rc=\$R3\"
exit \$((R1 + R2 + R3))
" || { echo "RUN_FAILED triple level=$LEVEL"; exit 1; }
T1=$(date +%s.%N)
echo "RUN_DONE triple level=$LEVEL window=$WINDOW NZ=${NZ:-1}"
echo "WALL_TRIPLE  $(echo "$T1 - $T0" | bc) s (three participants)"
if [ -n "$SNAP" ]; then
  rm -rf "$HERE/snap_$SNAP"
  mv "$RUN" "$HERE/snap_$SNAP"
  echo "snapshotted -> snap_$SNAP"
fi
