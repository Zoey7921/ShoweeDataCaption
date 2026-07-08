#!/usr/bin/env bash
export HF_ENDPOINT=https://hf-mirror.com
set -eu

PROJECT_ROOT="/data1/shared_data/qwen3vl-showeeData"
STARTER_KIT="/data1/wearable_ai_challenge_data/starter_kit"
PYTHON="$PROJECT_ROOT/.venv-vllm-judge/bin/python"

RUN_DIR="$PROJECT_ROOT/runs/liuyichen/challenge2_egoconv_700_egoschema_lora_balanced_detail"
GOLDEN="$RUN_DIR/golden.jsonl"
PREDICTIONS="$PROJECT_ROOT/eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_predictions.jsonl"
OUTPUT="$PROJECT_ROOT/eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_llmjudge_results.json"
LOG="$RUN_DIR/llm_judge.log"

mkdir -p "$RUN_DIR" "$PROJECT_ROOT/eval/results"
cd "$STARTER_KIT"
export VLLM_LOG_DIR="$RUN_DIR"

echo "START_LLM_JUDGE_WAIT $(date)" | tee -a "$LOG"

while true; do
  if [[ -f "$PREDICTIONS" ]]; then
    lines="$(wc -l < "$PREDICTIONS")"
  else
    lines="0"
  fi
  if [[ "$lines" -ge 700 ]]; then
    break
  fi
  echo "WAIT_PREDICTIONS lines=$lines $(date)" | tee -a "$LOG"
  sleep 600
done

if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("vllm") else 1)
PY
then
  echo "MISSING_VLLM: official Maverick judge needs a Python environment with vLLM." | tee -a "$LOG"
  echo "Install/activate vLLM env, then rerun this script." | tee -a "$LOG"
  exit 2
fi

while true; do
  free_count="$(
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F, '{
        gsub(/ /, "", $1); gsub(/ /, "", $2);
        if ($1 <= 2000 && $2 <= 10) { count += 1 }
      } END { print count + 0 }'
  )"
  if [[ "$free_count" -ge 8 ]]; then
    break
  fi
  echo "WAIT_8_FREE_GPUS free=$free_count $(date)" | tee -a "$LOG"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee -a "$LOG"
  sleep 600
done

echo "START_OFFICIAL_LLM_JUDGE $(date)" | tee -a "$LOG"
"$PYTHON" run_evaluation.py \
  --task convqa \
  --eval-only \
  --llm-judge \
  --llm-judge-backend vllm \
  --llm-judge-model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --llm-judge-vllm-tp-size 8 \
  --llm-judge-vllm-online-quantization fp8 \
  --golden "$GOLDEN" \
  --predictions "$PREDICTIONS" \
  --output "$OUTPUT" \
  2>&1 | tee -a "$LOG"

status="${PIPESTATUS[0]}"
echo "LLM_JUDGE_EXIT:$status $(date)" | tee -a "$LOG"

summary="${OUTPUT%.json}_summary.json"
if [[ "$status" -eq 0 && -f "$summary" ]]; then
  score="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("llm_judge",""))' "$summary")"
  if [[ -n "$score" ]]; then
    cp "$PREDICTIONS" "${PREDICTIONS%.jsonl}_with_llm_judge.jsonl"
    printf '{"llm_judge": %.6f}\n' "$score" >> "${PREDICTIONS%.jsonl}_with_llm_judge.jsonl"
    echo "WROTE_SUBMISSION_WITH_SCORE ${PREDICTIONS%.jsonl}_with_llm_judge.jsonl score=$score" | tee -a "$LOG"
  fi
fi

exit "$status"
