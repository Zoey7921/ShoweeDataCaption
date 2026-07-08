# ShoweeDataCaption

This repository now contains two peer modules:

- `data_pipeline/`: ShoweeHandv2 annotation and data conversion pipeline.
- `model_finetuning/`: Qwen3-VL LoRA fine-tuning, inference, and evaluation utilities.

Raw videos, processed datasets, model weights, LoRA adapters, and run outputs are intentionally kept out of Git.

## Layout

```text
ShoweeDataCaption/
  data_pipeline/
    configs/
    docs/
    examples/
    scripts/
    README.md
    requirements.txt
  model_finetuning/
    configs/
    docs/
    eval_sets/
    scripts/
    README.md
    requirements.txt
```

## Data Pipeline

Use `data_pipeline/` to build annotation datasets and export Wearable AI EgoConv-style JSONL:

```bash
cd data_pipeline
python scripts/build_temporal_caption_dataset.py --config configs/temporal_caption_v001.yaml
python scripts/export_wearable_egoconv.py --config configs/wearable_egoconv_v001.yaml
```

See `data_pipeline/README.md` for details.

## Model Fine-Tuning

Use `model_finetuning/` for Qwen3-VL LoRA experiments:

```bash
cd model_finetuning
python scripts/train_qwen3vl_lora.py --config configs/lora_wearable_egoconv_v001_multiturn.yaml
python scripts/infer_qwen3vl_lora_wearable.py \
  --eval-set eval_sets/wearable_egoconv_v001_multiturn_val20.json \
  --adapter /path/to/lora_adapter \
  --output /path/to/predictions.jsonl
```

Most config files preserve the absolute paths used in the shared server experiments. Update `model`, `train_data`, `output_dir`, and `run_dir` before running in a new environment.

See `model_finetuning/README.md` and `model_finetuning/docs/LoRA实验记录.md` for details.

## Git Hygiene

Do not commit:

- ShoweeHandv2 raw videos or processed training JSON with internal absolute paths.
- Qwen3-VL base model weights or LoRA checkpoints.
- `runs/`, `logs/`, `tmp/`, `eval/results/`, caches, and virtual environments.
- Generated contact sheets or review images.
