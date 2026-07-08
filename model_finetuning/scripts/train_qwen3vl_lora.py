#!/usr/bin/env python3
"""Train a small Qwen3-VL LoRA adapter on Showee ShareGPT-style video data."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from decord import VideoReader, cpu
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PROJECT_ROOT = Path("/data1/shared_data/qwen3vl-showeeData")
DEFAULT_CONFIG = PROJECT_ROOT / "configs/lora_showee_smoke.yaml"


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_video_frames(path: Path, num_frames: int) -> np.ndarray:
    reader = VideoReader(str(path), ctx=cpu(0))
    total_frames = len(reader)
    if total_frames <= 0:
        raise ValueError(f"Video has no decodable frames: {path}")
    sample_count = min(num_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, sample_count, dtype=np.int64)
    return reader.get_batch(indices).asnumpy()


def conversation_to_messages(sample: dict[str, Any], frames: np.ndarray) -> tuple[list[dict[str, Any]], list[str]]:
    messages: list[dict[str, Any]] = []
    assistant_values: list[str] = []
    video_attached = False

    for turn in sample["conversations"]:
        role = "assistant" if turn["from"] == "assistant" else "user"
        value = str(turn["value"])
        if role == "assistant":
            assistant_values.append(value)
            content: Any = value
        elif not video_attached:
            content = [
                {"type": "video", "video": frames},
                {"type": "text", "text": value.replace("<video>\n", "")},
            ]
            video_attached = True
        else:
            content = value.replace("<video>\n", "")
        messages.append({"role": role, "content": content})

    if not assistant_values:
        raise ValueError(f"Sample has no assistant messages: {sample.get('id')}")
    return messages, assistant_values


def find_subsequence(haystack: list[int], needle: list[int], start: int) -> int:
    if not needle:
        return -1
    limit = len(haystack) - len(needle) + 1
    for idx in range(start, limit):
        if haystack[idx : idx + len(needle)] == needle:
            return idx
    return -1


def build_labels(input_ids: torch.Tensor, assistant_values: list[str], processor: Any) -> torch.Tensor:
    ids = input_ids[0].tolist()
    labels = torch.full_like(input_ids, -100)
    cursor = 0

    for value in assistant_values:
        answer_ids = processor.tokenizer(value, add_special_tokens=False).input_ids
        start = find_subsequence(ids, answer_ids, cursor)
        if start < 0:
            preview = value[:80].replace("\n", "\\n")
            raise ValueError(f"Could not align assistant answer tokens: {preview}")
        end = start + len(answer_ids)
        labels[0, start:end] = input_ids[0, start:end]
        cursor = end

    if int((labels != -100).sum().item()) == 0:
        raise ValueError("No supervised assistant tokens were found")
    return labels


def build_batch(sample: dict[str, Any], processor: Any, num_frames: int, device: torch.device) -> dict[str, torch.Tensor]:
    frames = load_video_frames(Path(sample["video"]), num_frames)
    messages, assistant_values = conversation_to_messages(sample, frames)
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    labels = build_labels(batch["input_ids"], assistant_values, processor)
    batch["labels"] = labels

    out: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
    return out


def trainable_parameter_report(model: torch.nn.Module) -> dict[str, Any]:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": round(trainable / total * 100, 4) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=None, help="Override config max_steps.")
    parser.add_argument("--num-frames", type=int, default=None, help="Override config num_frames.")
    args = parser.parse_args()

    config = read_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.num_frames is not None:
        config["num_frames"] = args.num_frames

    set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model_path = Path(config["model"])
    train_data = Path(config["train_data"])
    output_dir = Path(config["output_dir"])
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(train_data)
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=dtype,
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    if config.get("use_gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config["target_modules"]),
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    epochs = int(config.get("epochs", 1))
    max_steps = int(config.get("max_steps", 0))
    grad_accum = int(config.get("gradient_accumulation_steps", 1))
    log_every = int(config.get("log_every", 1))
    save_every = int(config.get("save_every", 0))
    num_frames = int(config["num_frames"])
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    target_steps = max_steps if max_steps > 0 else epochs * len(samples)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    logs: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    step = 0
    update_step = 0

    while step < target_steps:
        for sample in samples:
            if step >= target_steps:
                break
            batch_start = time.time()
            batch = build_batch(sample, processor, num_frames, device)
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()

            do_update = (step + 1) % grad_accum == 0 or step + 1 == target_steps
            if do_update:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

            raw_loss = float(loss.detach().cpu().item() * grad_accum)
            log_row = {
                "step": step + 1,
                "update_step": update_step,
                "sample_id": sample.get("id"),
                "loss": raw_loss,
                "tokens": int(batch["input_ids"].numel()),
                "supervised_tokens": int((batch["labels"] != -100).sum().item()),
                "elapsed_seconds": round(time.time() - batch_start, 3),
            }
            logs.append(log_row)
            if (step + 1) % log_every == 0:
                print(json.dumps(log_row, ensure_ascii=False), flush=True)
            if save_every and (step + 1) % save_every == 0:
                checkpoint_dir = output_dir / f"checkpoint-step-{step + 1}"
                model.save_pretrained(checkpoint_dir)
                processor.save_pretrained(checkpoint_dir)

            step += 1

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    peak_memory_gb = None
    if torch.cuda.is_available():
        peak_memory_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    report = {
        "config": config,
        "train_data": str(train_data),
        "sample_count": len(samples),
        "target_steps": target_steps,
        "updates": update_step,
        "elapsed_seconds": round(time.time() - start_time, 3),
        "peak_memory_gb": peak_memory_gb,
        "parameter_report": trainable_parameter_report(model),
        "output_dir": str(output_dir),
        "final_loss": logs[-1]["loss"] if logs else None,
        "first_loss": logs[0]["loss"] if logs else None,
    }
    write_json(run_dir / "train_log.json", logs)
    write_json(run_dir / "run_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
