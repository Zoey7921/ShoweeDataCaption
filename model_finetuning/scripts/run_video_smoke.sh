#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data1/shared_data/qwen3vl-showeeData
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

mkdir -p "$PROJECT_ROOT/runs/puzuo/video_smoke_infer"

nvidia-smi > "$PROJECT_ROOT/runs/puzuo/video_smoke_infer/nvidia-smi.log"
python "$PROJECT_ROOT/scripts/infer_video_smoke.py" \
  > "$PROJECT_ROOT/runs/puzuo/video_smoke_infer/run.log" \
  2>&1

cat "$PROJECT_ROOT/runs/puzuo/video_smoke_infer/output.json"
