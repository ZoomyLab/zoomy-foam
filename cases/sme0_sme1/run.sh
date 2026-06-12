#!/bin/bash
# SME(0) <-> SME(1) inter-level coupling: emit both levels, build + stash both
# binaries, mesh, run mono reference + the coupled pair.
# Usage: run.sh [parallel-explicit|parallel-implicit]   (inside the OF container)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
FOAM_ROOT="$(cd "$HERE/../.." && pwd)"
SCHEME="${1:-parallel-explicit}"
PYBIN="${ZOOMY_PY:-python}"

set --   # clear positional args: the OF bashrc chokes on a sourced-with-args shell
set +e; source /opt/openfoam13/etc/bashrc; set -e

"$PYBIN" "$HERE/generate.py" "$SCHEME"

mkdir -p "$HERE/bin"
for L in 0 1; do
  "$PYBIN" "$FOAM_ROOT/create_model.py" --level $L
  ( cd "$FOAM_ROOT" && wmake > "$HERE/log.wmake_L$L" 2>&1 )
  cp "$(command -v zoomyFoam)" "$HERE/bin/zoomyFoam_L$L"
done
# leave the library at level 0 (the default/coupling baseline)
"$PYBIN" "$FOAM_ROOT/create_model.py" --level 0
( cd "$FOAM_ROOT" && wmake > /dev/null 2>&1 )

for c in mono mono_l1 part1 part2; do
  blockMesh -case "$HERE/$c" > "$HERE/$c/log.blockMesh" 2>&1
  setFields -case "$HERE/$c" > "$HERE/$c/log.setFields" 2>&1
done

unset FOAM_SIGFPE FOAM_SETNAN
"$HERE/bin/zoomyFoam_L0" -case "$HERE/mono" > "$HERE/mono/log.zoomyFoam" 2>&1
"$HERE/bin/zoomyFoam_L1" -case "$HERE/mono_l1" > "$HERE/mono_l1/log.zoomyFoam" 2>&1

( cd "$HERE/part1" && "$HERE/bin/zoomyFoam_L0" > log.zoomyFoam 2>&1 ) &
( cd "$HERE/part2" && "$HERE/bin/zoomyFoam_L1" > log.zoomyFoam 2>&1 ) &
wait
echo "sme0_sme1 ($SCHEME) done"
