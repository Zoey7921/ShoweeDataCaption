# Qwen3-VL Model Fine-Tuning

This module contains LoRA fine-tuning, inference, and lightweight evaluation utilities for Showee / wearable EgoConv experiments.

It is intentionally separated from `data_pipeline/`: the data pipeline builds annotation JSON/JSONL files, while this module consumes prepared training/eval files and trains or evaluates Qwen3-VL adapters.

## Layout

```text
model_finetuning/
  configs/       # LoRA experiment configs
  docs/          # LoRA plans, experiment logs, and progress notes
  eval_sets/     # small fixed eval sets safe to track in Git
  scripts/       # training, conversion, inference, and summarization scripts
```

## Important Paths

The copied experiment configs preserve the absolute paths used on the shared server, for example:

```text
/data1/shared_data/qwen3vl-showeeData/models/base/Qwen3-VL-8B-Instruct
/data1/shared_data/qwen3vl-showeeData/data/processed/
/data1/shared_data/qwen3vl-showeeData/models/lora/
```

Before running in a new clone, update the config fields:

- `model`
- `train_data`
- `output_dir`
- `run_dir`

Generated adapters, checkpoints, predictions, and run logs should remain outside Git.

## Current Main Experiment

Latest wearable multiturn training config:

```bash
configs/lora_wearable_egoconv_v001_multiturn.yaml
```

Training command used on the shared server:

```bash
python scripts/train_qwen3vl_lora.py \
  --config configs/lora_wearable_egoconv_v001_multiturn.yaml
```

Validation command:

```bash
python scripts/infer_qwen3vl_lora_wearable.py \
  --eval-set eval_sets/wearable_egoconv_v001_multiturn_val20.json \
  --adapter /data1/shared_data/qwen3vl-showeeData/models/lora/wearable_egoconv_v001_multiturn_lora \
  --output /data1/shared_data/qwen3vl-showeeData/eval/results/lora_wearable_egoconv_v001_multiturn_val20_predictions.jsonl \
  --num-frames 8 \
  --max-new-tokens 160
```

Summary of the latest result:

- Train data: line 1-100 from `wearable_egoconv_v001.jsonl`, all 3-turn conversations.
- Validation: line 101-120 expanded to 60 turns.
- Train loss: `3.3761 -> 1.2607`.
- Val non-empty: `60/60`.
- Val ROUGE-L F1: `0.4608`.
- First-turn task hit: `18/20`.

See `docs/LoRA实验记录.md` for the full experiment record.

## Dependencies

Install the project-specific dependencies into an existing PyTorch/CUDA environment:

```bash
pip install -r requirements.txt
```

The shared server experiment used bf16 LoRA with PEFT. It did not require bitsandbytes, TRL, DeepSpeed, or flash-attn for the small runs recorded here.
