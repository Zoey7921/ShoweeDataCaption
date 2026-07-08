# EgoConv 700 Submission Checklist

Date: 2026-07-07

## Current run

Configuration:

```text
base model: models/base/Qwen3-VL-8B-Instruct
adapter: models/lora/egoschema_v001_lora
prompt_style: balanced_detail
frames_per_interval: 2
max_frames: 12
```

Running tmux:

```text
egoconv_700_balanced_detail_wait
egoconv_700_llm_judge_wait
```

Main prediction file:

```text
eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_predictions.jsonl
```

## Check progress

```bash
cd /data1/shared_data/qwen3vl-showeeData
.venv/bin/python scripts/egoconv_run_status.py \
  --predictions eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_predictions.jsonl \
  --log runs/liuyichen/challenge2_egoconv_700_egoschema_lora_balanced_detail/run.log \
  --total 700
```

## Validate prediction format

Partial check while running:

```bash
cd /data1/shared_data/qwen3vl-showeeData
.venv/bin/python scripts/check_egoconv_predictions.py \
  --golden runs/liuyichen/challenge2_egoconv_700_egoschema_lora_balanced_detail/golden.jsonl \
  --predictions eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_predictions.jsonl \
  --allow-partial
```

Final check after 700 rows:

```bash
cd /data1/shared_data/qwen3vl-showeeData
.venv/bin/python scripts/check_egoconv_predictions.py \
  --golden runs/liuyichen/challenge2_egoconv_700_egoschema_lora_balanced_detail/golden.jsonl \
  --predictions eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_predictions.jsonl
```

## BLEU sanity check

This is not the official leaderboard metric, but it is useful for local sanity checking.

```bash
cd /data1/wearable_ai_challenge_data/starter_kit
/data1/shared_data/qwen3vl-showeeData/.venv/bin/python run_evaluation.py \
  --task convqa \
  --eval-only \
  --no-llm-judge \
  --golden /data1/shared_data/qwen3vl-showeeData/runs/liuyichen/challenge2_egoconv_700_egoschema_lora_balanced_detail/golden.jsonl \
  --predictions /data1/shared_data/qwen3vl-showeeData/eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_predictions.jsonl \
  --output /data1/shared_data/qwen3vl-showeeData/eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_results.json
```

## Official LLM judge

The official EgoConv leaderboard metric is LLM-as-Judge response accuracy. The vLLM judge environment is:

```text
/data1/shared_data/qwen3vl-showeeData/.venv-vllm-judge
```

The waiting script is already running in tmux:

```text
egoconv_700_llm_judge_wait
```

Manual command if needed:

```bash
cd /data1/shared_data/qwen3vl-showeeData
bash scripts/run_egoconv_700_llm_judge.sh
```

Expected outputs:

```text
eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_llmjudge_results.json
eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_llmjudge_results_summary.json
eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_predictions_with_llm_judge.jsonl
```

If model access fails, login with a Hugging Face account that has access to:

```text
meta-llama/Llama-4-Maverick-17B-128E-Instruct
```

Command:

```bash
source /data1/shared_data/qwen3vl-showeeData/.venv-vllm-judge/bin/activate
hf auth login
```

## Partial analysis artifacts

Generated while the 700-run was still in progress:

```text
eval/results/egoconv_experiment_summary_2026-07-07.json
eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_partial_228_results.json
eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_partial_228_results_summary.json
runs/liuyichen/challenge2_egoconv_700_egoschema_lora_balanced_detail/partial_228_analysis.md
runs/liuyichen/challenge2_egoconv_700_egoschema_lora_balanced_detail/partial_228_worst_turns.csv
runs/liuyichen/challenge2_egoconv_700_egoschema_lora_balanced_detail/partial_228_error_summary.json
```

Partial 228 result:

```text
BLEU: 0.1037
conversations: 228
turns: 1520
```
