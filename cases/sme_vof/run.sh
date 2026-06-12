#!/usr/bin/env bash
# Run the two-way SME(level)<->VOF coupled case end-to-end inside the container.
# Usage: run.sh LEVEL [SNAP_SUFFIX] [WINDOW] [SCHEME] [GHOST] [OUTER]
# Env  : NZ=1 TRANSVERSE=cyclic   (3D VOF: NZ>1, front/back cyclic|wall)
set -e
LEVEL=${1:?level}
SNAP=${2:-}
WINDOW=${3:-2e-3}
SCHEME=${4:-parallel-explicit}
GHOST=${5:-fullstate}
OUTER=${6:-wall}   # wall (closed system, default) | extrapolation
FROZEN=${7:-auto}
LEDGER=${8:-20}
HERE="$(cd "$(dirname "$0")" && pwd)"
SIF=/Users/adam-obbpb5az1dhsjzf/of_build/zoomy_openfoam.sif
RUN=$HERE/run${SNAP:+_$SNAP}          # per-snapshot run dir -> parallel batches
export RUNDIR=$RUN
SUFFIX=""; [ "$OUTER" = "wall" ] && SUFFIX="w"
BIN=$HERE/bin/zoomyFoam_L$LEVEL$SUFFIX
[ -x "$BIN" ] || { echo "missing $BIN — run: bash compile.sh $LEVEL $OUTER"; exit 1; }
PY=/Users/adam-obbpb5az1dhsjzf/micromamba/envs/zoomy/bin/python

rm -rf "$RUN"
$PY "$HERE/generate.py" "$LEVEL" "$WINDOW" "$SCHEME" "$GHOST" "$FROZEN" "$LEDGER"

T0=$(date +%s.%N)
apptainer exec "$SIF" bash -c "
set --
set +e; source /opt/openfoam13/etc/bashrc; set -e
cd $RUN/swe_case
blockMesh > log.blockMesh 2>&1
setFields  > log.setFields 2>&1
cd $RUN/vof_case
blockMesh > log.blockMesh 2>&1
setFields  > log.setFields 2>&1
cd $RUN
$BIN -case swe_case > log.swe 2>&1 &
SWE_PID=\$!
foamRun -case vof_case > log.vof 2>&1 &
VOF_PID=\$!
wait \$SWE_PID; SWE_RC=\$?
wait \$VOF_PID; VOF_RC=\$?
echo \"swe rc=\$SWE_RC vof rc=\$VOF_RC\"
exit \$((SWE_RC + VOF_RC))
" || { echo "RUN_FAILED level=$LEVEL"; exit 1; }
T1=$(date +%s.%N)
echo "RUN_DONE level=$LEVEL scheme=$SCHEME window=$WINDOW NZ=${NZ:-1}"
echo "WALL_PAIR  $(echo "$T1 - $T0" | bc) s (both participants, incl. mesh+startup)"
echo "SWE solver: $(grep ExecutionTime "$RUN/log.swe" | tail -1)"
echo "VOF solver: $(grep ExecutionTime "$RUN/log.vof" | tail -1)"
if [ -n "$SNAP" ]; then
  rm -rf "$HERE/snap_$SNAP"
  mv "$RUN" "$HERE/snap_$SNAP"
  echo "snapshotted -> snap_$SNAP"
fi
