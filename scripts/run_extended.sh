#!/usr/bin/env bash
# AI-assisted: written with Claude (Anthropic) via Claude Code; reviewed and tested by the team.
# Runs (or resumes) the pre-registered extended experiments, 4 jobs at a time.
# Each line of scripts/extended_jobs.txt is one job; finished configurations already in
# docs/results/extended/*.csv are kept (--resume), so rerunning this script continues where it stopped.
#   bash scripts/run_extended.sh            (from the repo root, after `make`)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=python
OUT=docs/results/extended
mkdir -p "$OUT"
xargs -P "${JOBS:-4}" -L 1 sh -c '
  name=$(echo "$@" | sed "s/.*--out out\///; s/\.csv.*//")
  python3 apps/fp4_experiment.py --dataset cifar10 --resume $(echo "$@" | sed "s#--out out/#--out '"$OUT"'/#") > "'"$OUT"'/$name.log" 2>&1
' _ < scripts/extended_jobs.txt
echo ALLDONE
