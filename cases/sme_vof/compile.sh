#!/usr/bin/env bash
# Build the SME participant binaries this case runs (sequentially — one
# shared wmake tree).  Usage: compile.sh [LEVELS...] [OUTER]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
OUTER=wall
LEVELS=()
for a in "$@"; do
  case "$a" in wall|extrapolation) OUTER=$a;; *) LEVELS+=("$a");; esac
done
[ ${#LEVELS[@]} -gt 0 ] || LEVELS=(0)
for L in "${LEVELS[@]}"; do
  bash "$HERE/../compile_sme.sh" "$L" "$OUTER" "$HERE"
done
