#!/usr/bin/env bash
# GQA comparison: Qwen2.5-3B FP16, step 32, prompt 128, 1920 generated, M = 8 GiB, one KV copy per beam.
#   fits in 8 GiB: 32, 64, 112 beams (all KV on the GPU)
#   does not fit : 128, 256 beams (baseline and LBS); 512 beams is attempted and recorded as skipped if host memory is short
# About 20 minutes on an RTX 5060 Ti. Finished runs are skipped, so re-running resumes.
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
mkdir -p logs/gqa
run() {  # mode beams
  local log="logs/gqa/percopy_$1_N$2.log"
  echo "=== $(date +%T) $1 N=$2 -> $log"
  $PY scripts/run.py --exp gqa --model qwen --store percopy --mode "$1" --beams "$2" --mem-gib 8 > "$log" 2>&1
  local code=$?
  tail -n 2 "$log"
  case $code in
    0) ;;
    3) echo "not enough host memory ($1 N=$2); recorded, continuing" ;;
    5) echo "GPU out of memory ($1 N=$2); recorded, continuing" ;;
    *) echo "run failed (exit $code); stopping"; exit $code ;;
  esac
}
for n in 32 64 112; do run ingpu $n; done
for n in 128 256; do run lbs $n; run baseline $n; done
run lbs 512
run baseline 512
echo "=== $(date +%T) done"
