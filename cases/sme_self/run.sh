#!/usr/bin/env bash
# SME(N)<->SME(N) same-level coupling vs monolithic reference.
# Usage: run.sh LEVEL [SCHEME] [ZSAMPLES]   (binaries: ../sme0_sme1/bin/zoomyFoam_L$LEVEL)
set -e
LEVEL=${1:?level}
SCHEME=${2:-parallel-explicit}
ZS=${3:-16}
DT=${4:-5e-4}
TEND=${5:-1.0}
HERE="$(cd "$(dirname "$0")" && pwd)"
SIF=/Users/adam-obbpb5az1dhsjzf/of_build/zoomy_openfoam.sif
BIN="$HERE/../sme0_sme1/bin/zoomyFoam_L$LEVEL"
PY=/Users/adam-obbpb5az1dhsjzf/micromamba/envs/zoomy/bin/python

rm -rf "$HERE"/{part1,part2,mono,precice-run} "$HERE"/precice-*.log
$PY "$HERE/generate.py" "$LEVEL" "$SCHEME" "$ZS" "$DT" "$TEND"

T0=$(date +%s.%N)
apptainer exec "$SIF" bash -c "
set --
set +e; source /opt/openfoam13/etc/bashrc; set -e
for c in part1 part2 mono; do
  cd $HERE/\$c && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1
done
cd $HERE
$BIN -case mono > mono/log.run 2>&1
$BIN -case part1 > part1/log.run 2>&1 &
P1=\$!
$BIN -case part2 > part2/log.run 2>&1 &
P2=\$!
wait \$P1; R1=\$?
wait \$P2; R2=\$?
echo \"part1 rc=\$R1 part2 rc=\$R2\"
exit \$((R1 + R2))
" || { echo "RUN_FAILED L$LEVEL"; exit 1; }
T1=$(date +%s.%N)
echo "RUN_DONE sme_self L$LEVEL $SCHEME zs=$ZS  wall=$(echo "$T1 - $T0" | bc) s"
