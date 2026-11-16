#!/usr/bin/env bash
# Deduplication experiment: OPT-2.7B, 32 beams, prompt 128, 1920 generated; step 32, 64, 128. Per step length:
#   M = 2.2 GiB: per-beam baseline (samples freely; its beam tree is the reference), per-beam LBS, dedup baseline, dedup LBS
#   M = 8 GiB  : dedup LBS and dedup baseline (control: no transfer expected)
#   M = 1 GiB  : dedup baseline and dedup LBS (supplementary, tighter budget), after all of the above
# Every run except the reference takes its tokens from the reference tree, so all four do the same work.
# About 1.5 hours on an RTX 5060 Ti (each per-beam baseline at M = 2.2 GiB takes ~17 minutes).
# Finished runs are skipped, so re-running resumes. Usage: scripts/run_dedup.sh [step ...]
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
mkdir -p logs/dedup
run() {  # store mode step M [force-tree]
  local name="$1_$2_N32_step$3_M$4"
  local log="logs/dedup/$name.log"
  echo "=== $(date +%T) $name -> $log"
  local force=()
  [ -n "$5" ] && force=(--force-tree "$5")
  $PY scripts/run.py --exp dedup --store "$1" --mode "$2" --step "$3" --mem-gib "$4" "${force[@]}" > "$log" 2>&1
  local code=$?
  tail -n 2 "$log"
  if [ $code -ne 0 ]; then echo "run failed (exit $code); stopping"; exit $code; fi
}
STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(32 64 128)
for t in "${STEPS[@]}"; do
  ref="results/dedup/trees/percopy_baseline_N32_step${t}_M2.2.npz"
  run percopy baseline "$t" 2.2
  run percopy lbs      "$t" 2.2 "$ref"
  run dedup   baseline "$t" 2.2 "$ref"
  run dedup   lbs      "$t" 2.2 "$ref"
  run dedup   lbs      "$t" 8   "$ref"
  run dedup   baseline "$t" 8   "$ref"
done
for t in "${STEPS[@]}"; do
  ref="results/dedup/trees/percopy_baseline_N32_step${t}_M2.2.npz"
  run dedup baseline "$t" 1 "$ref"
  run dedup lbs      "$t" 1 "$ref"
done
echo "=== $(date +%T) done"
