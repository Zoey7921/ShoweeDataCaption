#!/usr/bin/env bash
set -eu

PROJECT_ROOT="/data1/shared_data/qwen3vl-showeeData"
RUN_DIR="$PROJECT_ROOT/runs/liuyichen/challenge2_egoconv_50_egoschema_lora_precision_guarded"
OUTPUT="$PROJECT_ROOT/eval/results/egoschema_lora_v001_challenge2_egoconv_50_precision_guarded_predictions.jsonl"
GOLDEN="$RUN_DIR/golden.jsonl"
REPORT="$RUN_DIR/run_report.json"
EVAL_OUTPUT="$PROJECT_ROOT/eval/results/egoschema_lora_v001_challenge2_egoconv_50_precision_guarded_results.json"

mkdir -p "$RUN_DIR" "$PROJECT_ROOT/eval/results"
cd "$PROJECT_ROOT"

echo "START_WAIT_FOR_GPU $(date)" | tee -a "$RUN_DIR/run.log"

while true; do
  gpu="$(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F, '{
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
        if (found == "" && $2 <= 2000 && $3 <= 10) { found = $1 }
      }
      END {
        if (found != "") { print found }
      }'
  )"
  if [[ -n "$gpu" ]]; then
    break
  fi
  echo "NO_FREE_GPU $(date)" | tee -a "$RUN_DIR/run.log"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee -a "$RUN_DIR/run.log"
  sleep 300
done

echo "SELECTED_GPU=$gpu $(date)" | tee -a "$RUN_DIR/run.log"
source "$PROJECT_ROOT/.venv/bin/activate"

CUDA_VISIBLE_DEVICES="$gpu" python scripts/qwen3vl_egoconv_smoke.py \
  --adapter models/lora/egoschema_v001_lora \
  --max-samples 50 \
  --max-turns 0 \
  --frames-per-interval 2 \
  --max-frames 12 \
  --prompt-style precision_guarded \
  --output "$OUTPUT" \
  --golden-output "$GOLDEN" \
  --report-output "$REPORT" \
  2>&1 | tee -a "$RUN_DIR/run.log"

status="${PIPESTATUS[0]}"
echo "INFERENCE_EXIT:$status $(date)" | tee -a "$RUN_DIR/run.log"

if [[ "$status" -eq 0 ]]; then
  echo "START_EVAL $(date)" | tee -a "$RUN_DIR/eval.log"
  python /data1/wearable_ai_challenge_data/starter_kit/run_evaluation.py \
    --task convqa \
    --eval-only \
    --no-llm-judge \
    --golden "$GOLDEN" \
    --predictions "$OUTPUT" \
    --output "$EVAL_OUTPUT" \
    2>&1 | tee -a "$RUN_DIR/eval.log"
  echo "EVAL_EXIT:${PIPESTATUS[0]} $(date)" | tee -a "$RUN_DIR/eval.log"
fi

echo "END $(date)" | tee -a "$RUN_DIR/run.log"
