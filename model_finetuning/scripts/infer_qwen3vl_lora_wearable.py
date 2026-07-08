#!/usr/bin/env python3
"""Run Qwen3-VL or Qwen3-VL+LoRA inference on wearable EgoConv eval rows."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PROJECT_ROOT = Path("/data1/shared_data/qwen3vl-showeeData")
DEFAULT_MODEL = PROJECT_ROOT / "models/base/Qwen3-VL-8B-Instruct"
DEFAULT_ADAPTER = PROJECT_ROOT / "models/lora/wearable_egoconv_v001_reviewed_lora"
DEFAULT_EVAL = PROJECT_ROOT / "eval/sets/wearable_egoconv_v001_reviewed_val20.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval/results/lora_wearable_egoconv_v001_reviewed_val20_predictions.jsonl"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_video_frames(path: Path, num_frames: int) -> np.ndarray:
    reader = VideoReader(str(path), ctx=cpu(0))
    total_frames = len(reader)
    if total_frames <= 0:
        raise ValueError(f"Video has no decodable frames: {path}")
    sample_count = min(num_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, sample_count, dtype=np.int64)
    return reader.get_batch(indices).asnumpy()


def build_messages(row: dict[str, Any], frames: np.ndarray) -> list[dict[str, Any]]:
    source_messages = row.get("messages") or [{"role": "user", "text": row["question"]}]
    messages: list[dict[str, Any]] = []
    video_attached = False
    for turn in source_messages:
        role = str(turn.get("role", "user"))
        text = str(turn.get("text", ""))
        if role == "assistant":
            messages.append({"role": "assistant", "content": text})
            continue

        content: list[dict[str, Any]] = []
        if not video_attached:
            content.append({"type": "video", "video": frames})
            video_attached = True
        content.append({"type": "text", "text": text})
        messages.append({"role": "user", "content": content})
    return messages


def generate_one(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    num_frames: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    video_path = Path(row["video"])
    frames = load_video_frames(video_path, num_frames)
    messages = build_messages(row, frames)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    start = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    prediction = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return {
        "id": row.get("id"),
        "video": row["video"],
        "question": row.get("question"),
        "messages": row.get("messages"),
        "prediction": prediction,
        "reference_answer": row.get("reference_answer"),
        "metadata": row.get("metadata", {}),
        "num_frames": num_frames,
        "elapsed_seconds": round(time.time() - start, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model),
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True,
    )
    processor_source = args.adapter if args.adapter else args.model
    processor = AutoProcessor.from_pretrained(str(processor_source), local_files_only=True)
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True)
    model.eval()

    rows = load_json(args.eval_set)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    start = time.time()
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            pred = generate_one(model, processor, row, args.num_frames, args.max_new_tokens)
            pred["model"] = str(args.model)
            pred["adapter"] = str(args.adapter) if args.adapter else None
            predictions.append(pred)
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({"id": pred["id"], "prediction": pred["prediction"]}, ensure_ascii=False), flush=True)

    summary = {
        "eval_set": str(args.eval_set),
        "output": str(args.output),
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "count": len(predictions),
        "elapsed_seconds": round(time.time() - start, 3),
        "num_frames": args.num_frames,
        "max_new_tokens": args.max_new_tokens,
    }
    with args.output.with_suffix(args.output.suffix + ".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
