# Qwen3-VL Showee Annotation Pipeline

This directory contains the lightweight annotation pipeline used to build Qwen3-VL video QA data from ShoweeHandv2.

It is intentionally separated from the working experiment directory so it can be pushed to GitHub without raw videos, processed frames, model weights, checkpoints, or run outputs.

## What This Pipeline Does

1. Index readable ShoweeHandv2 `showee_head` videos and their `metadata.json`.
2. Build a small v001 AI-assisted annotation set with Qwen3-VL.
3. Expand to v002 with metadata-seeded samples and fixed train/val/test splits.
4. Generate open-description eval sets and candidate-task choice eval sets.
5. Add choice-style training samples for every v002 train video.
6. Add coarse temporal action segments while preserving the multi-turn format.
7. Build a reusable temporal caption dataset independent of v001/v002 batch names.
8. Build streaming interaction samples from temporal captions.
9. Summarize open/choice prediction files.

This is not the full ShoweeHandv2 raw-to-processed data pipeline. It consumes the dataset's raw videos and metadata to create Qwen3-VL fine-tuning/evaluation JSON files.

## Directory Layout

```text
annotation_pipeline/
  README.md
  requirements.txt
  scripts/
    build_showee_ai_dataset_v001.py
    build_showee_dataset_v002.py
    build_showee_choice_train_v002.py
    add_temporal_segments.py
    build_temporal_caption_dataset.py
    apply_temporal_review.py
    build_streaming_interaction_dataset.py
    summarize_showee_eval.py
  docs/
    ShoweeData_v001_标注说明.md
    ShoweeData_v002_标注说明.md
    TemporalCaption_标注规范.md
    StreamingInteraction_标注规范.md
  configs/
    temporal_caption_v001.yaml
    streaming_interaction_v001.yaml
  examples/
    sharegpt_sample.json
    choice_messages_sample.json
```

Generated files are written under `data/`, `eval/`, and `runs/` relative to the repository root. Those directories are ignored by Git.

## Path Configuration

The scripts default to paths inside this repository. On a shared server, set these variables to use an existing dataset/model:

```bash
export SHOWEE_PIPELINE_ROOT=<repo_root>
export SHOWEE_RAW_ROOT=<path_to_ShoweeHandv2>/raw
export QWEN3VL_MODEL=<path_to_Qwen3-VL-8B-Instruct>
```

## Usage

Index available videos only:

```bash
python scripts/build_showee_ai_dataset_v001.py --index-only
```

Build v001 AI-assisted annotations:

```bash
python scripts/build_showee_ai_dataset_v001.py \
  --raw-root "$SHOWEE_RAW_ROOT" \
  --model "$QWEN3VL_MODEL" \
  --max-samples 20 \
  --num-frames 8
```

Build v002 from reviewed v001 plus metadata-seeded expansion:

```bash
python scripts/build_showee_dataset_v002.py
```

Build v002 choice training samples:

```bash
python scripts/build_showee_choice_train_v002.py
```

Add coarse start/end timestamps to existing ShareGPT-style samples:

```bash
python scripts/add_temporal_segments.py \
  data/processed/showee_train_v002.json \
  data/processed/showee_val_v002.json \
  data/processed/showee_test_v002.json
```

Build the reusable temporal caption dataset:

```bash
python scripts/build_temporal_caption_dataset.py \
  --config configs/temporal_caption_v001.yaml
```

Apply manual temporal review updates back to JSON:

```bash
python scripts/apply_temporal_review.py --dry-run
python scripts/apply_temporal_review.py
```

Build streaming interaction samples from temporal captions:

```bash
python scripts/build_streaming_interaction_dataset.py \
  --config configs/streaming_interaction_v001.yaml
```

Summarize an eval prediction file:

```bash
python scripts/summarize_showee_eval.py eval/results/predictions.jsonl
```

## Output Format

Training data uses a ShareGPT-like video conversation format:

```json
{
  "id": "showee_0006_midair_1_asl_y_head",
  "video": "/path/to/video.mkv",
  "metadata": {
    "task_id": "asl_y",
    "task_name": "ASL Y",
    "split": "train",
    "review_status": "edited"
  },
  "conversations": [
    {
      "from": "user",
      "value": "<video>\n请描述视频中正在执行的手势任务。"
    },
    {
      "from": "assistant",
      "value": "视频展示的是 ASL Y 手势任务。..."
    }
  ]
}
```

Choice samples use the same `conversations` format for training and also provide a `messages` variant for inspection or other trainers.

Temporal annotations are stored in `metadata.temporal_segments` and the existing "动作从头到尾有什么变化" assistant answer is rewritten to mention timestamps. Current timestamps are coarse task-level boundaries derived from metadata duration, not frame-level human boundaries.

The reusable temporal caption pipeline writes a new dataset such as `temporal_caption_v001.json` and can inherit prior annotations when configured, but it does not depend on v001/v002 naming or split logic.

For manual review, add a `review_update:` YAML block under a sample in `eval/human_review/temporal_caption_v001_review.md`, then run `apply_temporal_review.py`. The script updates the full JSON, train/val/test split JSON files, and the index CSV.

The streaming interaction pipeline converts temporal segments into time-node guidance turns. It produces both canonical `turns` JSON and trainer-friendly `messages` JSON.

## Git Hygiene

Do not commit:

- ShoweeHandv2 raw videos or processed data.
- Qwen3-VL model weights or LoRA checkpoints.
- Generated JSON datasets that contain absolute internal paths, unless they are intentionally sanitized examples.
- `runs/`, `logs/`, `eval/results/`, caches, and virtual environments.
