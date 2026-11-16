#!/usr/bin/env bash
# LBS reproduction, one KV copy per beam: OPT-2.7B, step 32, prompt 128, 1920 generated, M = 8 GiB,
# 8 / 16 / 24 / 32 beams (baseline and LBS; 8 beams also in-GPU). About 25 minutes on an RTX 5060 Ti.
# Each run is its own process; finished runs are skipped, so re-running resumes.
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
mkdir -p logs/repro
run() {  # mode beams
  local log="logs/repro/percopy_$1_N$2.log"
  echo "=== $(date +%T) $1 N=$2 -> $log"
  $PY scripts/run.py --exp repro --store percopy --mode "$1" --beams "$2" --mem-gib 8 > "$log" 2>&1
  local code=$?
  tail -n 2 "$log"
  if [ $code -eq 3 ]; then echo "not enough host memory for N=$2 ($1); stopping"; exit 3; fi
  if [ $code -ne 0 ]; then echo "run failed (exit $code); stopping"; exit $code; fi
}
BEAMS=("$@"); [ ${#BEAMS[@]} -eq 0 ] && BEAMS=(8 16 24 32)
for n in "${BEAMS[@]}"; do
  if [ "$n" -eq 8 ]; then run ingpu 8; fi
  run lbs "$n"
  run baseline "$n"
done
echo "=== $(date +%T) done"
