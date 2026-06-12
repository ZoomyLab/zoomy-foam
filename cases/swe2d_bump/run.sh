#!/usr/bin/env bash
# Run swe2d_bump end-to-end: mono (standalone) + coupled pair (preCICE).
# Prerequisite: bash compile.sh   (model.py -> headers -> bin/zoomyFoam_swe2d)
# Usage: run.sh [T_END] [WINDOW]
set -e
T_END=${1:-6.0}
WINDOW=${2:-5e-3}
HERE="$(cd "$(dirname "$0")" && pwd)"
SIF=/Users/adam-obbpb5az1dhsjzf/of_build/zoomy_openfoam.sif
RUN=$HERE/run
export RUNDIR=$RUN
BIN=$HERE/bin/zoomyFoam_swe2d
PY=/Users/adam-obbpb5az1dhsjzf/micromamba/envs/zoomy/bin/python

[ -x "$BIN" ] || { echo "missing $BIN — run compile.sh first"; exit 1; }
rm -rf "$RUN"
$PY "$HERE/generate.py" "$T_END" "$WINDOW"

T0=$(date +%s.%N)
apptainer exec "$SIF" bash -c "
set --
set +e; source /opt/openfoam13/etc/bashrc; set -e
for c in mono part1 part2; do
  cd $RUN/\$c && blockMesh > log.blockMesh 2>&1
done
cd $RUN
$BIN -case mono > log.mono 2>&1
echo \"mono rc=\$?\"
$BIN -case part1 > log.part1 2>&1 &  P1=\$!
$BIN -case part2 > log.part2 2>&1 &  P2=\$!
wait \$P1; R1=\$?
wait \$P2; R2=\$?
echo \"part1 rc=\$R1 part2 rc=\$R2\"
exit \$((R1 + R2))
" || { echo "RUN_FAILED"; exit 1; }
T1=$(date +%s.%N)
echo "RUN_DONE T=$T_END window=$WINDOW"
echo "WALL  $(echo "$T1 - $T0" | bc) s (mono + coupled pair)"
