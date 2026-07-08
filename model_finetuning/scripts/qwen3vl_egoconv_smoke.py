#!/usr/bin/env python3
"""Run a small Qwen3-VL smoke test for Wearable AI EgoConv.

The script selects validation rows whose videos are already downloaded, runs
sequential multi-turn inference, and writes the official ConvQA prediction
format:

    {"video_path": "...mp4", "answers": ["...", "..."]}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/data1/shared_data/qwen3vl-showeeData")
DATA_ROOT = Path("/data1/wearable_ai_challenge_data")

DEFAULT_MODEL = PROJECT_ROOT / "models/base/Qwen3-VL-8B-Instruct"
DEFAULT_INPUT = DATA_ROOT / "egoconv/wearable_ai_2026_egoconv_val_700.jsonl"
DEFAULT_VIDEO_DIR = DATA_ROOT / "egoconv/val"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs/liuyichen/challenge2_egoconv_smoke"
DEFAULT_GOLDEN = DEFAULT_RUN_DIR / "golden.jsonl"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "eval/results/baseline_qwen3vl_challenge2_egoconv_smoke_predictions.jsonl"
)
DEFAULT_REPORT = DEFAULT_RUN_DIR / "run_report.json"

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about an egocentric video. "
    "Answer directly and concisely using the video, prior conversation, and "
    "general knowledge when the question asks for facts. For visual questions, "
    "ground the answer in visible evidence. If the answer cannot be determined "
    "from the available context, say so."
)

PROMPT_STYLES = {
    "default": SYSTEM_PROMPT,
    "less_refusal": (
        "You answer multi-turn questions about an egocentric video. Use the "
        "visible video evidence, previous questions, and your previous answers. "
        "Answer the current question directly in one short sentence. Prefer the "
        "most likely answer when there is reasonable visual evidence. Do not "
        "start with uncertainty phrases such as 'I cannot determine' unless the "
        "video gives no useful evidence at all. Resolve references such as "
        "'this', 'that', 'here', and 'it' from the conversation context."
    ),
    "balanced_detail": (
        "You answer multi-turn questions about an egocentric video. Use the "
        "visible video evidence, previous questions, previous answers, and "
        "general knowledge when the question asks for facts, reasons, or "
        "instructions. Answer in one or two concise sentences, including key "
        "visible details such as names, numbers, colors, object labels, places, "
        "or actions when they are visible. Prefer the most likely answer when "
        "there is reasonable evidence, but do not invent exact names, locations, "
        "models, or causes that are not visible or strongly implied. For "
        "how-to, why, and when questions, answer the question directly even if "
        "the current video frame is not essential. Resolve references such as "
        "'this', 'that', 'here', and 'it' from the conversation context."
    ),
    "precision_guarded": (
        "You answer multi-turn questions about an egocentric video. Use the "
        "visible video evidence, previous questions, previous answers, and "
        "general knowledge when the question asks for facts, reasons, or "
        "instructions. Answer the current question directly in one or two "
        "concise sentences. First identify what the user is referring to from "
        "the current and previous turns. Give exact names, dates, counts, "
        "models, locations, or artist names only when they are readable, "
        "clearly visible, or strongly established by prior context. If an "
        "exact detail is not visible, give the safest useful answer without "
        "inventing a specific value. For yes/no, how-to, why, and common "
        "knowledge questions, answer plainly and include the key reason or "
        "instruction. Avoid saying that the video gives no information unless "
        "the question truly cannot be answered from the visual context, "
        "conversation, or common knowledge."
    ),
    "balanced_no_hallucination": (
        "You answer multi-turn questions about an egocentric video. Use the "
        "visible video evidence, previous questions, previous answers, and "
        "general knowledge when the question asks for facts, reasons, or "
        "instructions. Answer the current question directly in one or two "
        "concise sentences. Resolve references such as 'this', 'that', "
        "'here', and 'it' from the conversation context. Include visible "
        "details such as colors, objects, signs, actions, places, numbers, "
        "and names when they are clear. For exact names, dates, counts, "
        "brands, models, artists, or locations, use a specific value only "
        "when it is visible, readable, strongly implied by the scene, or "
        "commonly known from the question context. If the exact value is not "
        "clear, still answer helpfully with the best supported description "
        "instead of refusing. For yes/no, how-to, why, and common-knowledge "
        "questions, answer plainly and include the key reason or instruction."
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_downloaded_rows(
    rows: list[dict[str, Any]],
    video_dir: Path,
    max_samples: int,
    max_turns: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        video_path = video_dir / str(row["video_path"])
        if not video_path.exists():
            skipped.append(
                {
                    "index": index,
                    "video_path": row["video_path"],
                    "reason": "missing_video",
                }
            )
            continue
        item = dict(row)
        item["_source_index"] = index
        if max_turns is not None and max_turns > 0:
            item["questions"] = item["questions"][:max_turns]
            item["answers"] = item["answers"][:max_turns]
            item["video_intervals"] = item["video_intervals"][:max_turns]
        selected.append(item)
        if len(selected) >= max_samples:
            break
    return selected, skipped


def sample_interval_frames(
    reader: Any,
    fps: float,
    total_frames: int,
    interval: list[float],
    frames_per_interval: int,
) -> np.ndarray:
    start_sec, end_sec = float(interval[0]), float(interval[1])
    if fps <= 0 or total_frames <= 0:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)

    video_end_sec = total_frames / fps
    start_sec = max(0.0, min(start_sec, video_end_sec))
    end_sec = max(start_sec, min(end_sec, video_end_sec))
    start_frame = int(start_sec * fps)
    end_frame = min(max(start_frame, int(end_sec * fps)), total_frames - 1)
    if end_frame < start_frame:
        end_frame = start_frame

    count = min(frames_per_interval, max(1, end_frame - start_frame + 1))
    indices = np.linspace(start_frame, end_frame, count, dtype=np.int64)
    return reader.get_batch(indices).asnumpy()


def load_accumulated_frames(
    video_path: Path,
    intervals: list[list[float]],
    turn: int,
    frames_per_interval: int,
    max_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from decord import VideoReader, cpu

    reader = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(reader)
    fps = float(reader.get_avg_fps())

    chunks: list[np.ndarray] = []
    for interval in intervals[: turn + 1]:
        frames = sample_interval_frames(
            reader, fps, total_frames, interval, frames_per_interval
        )
        if frames.size:
            chunks.append(frames)
    if not chunks:
        raise ValueError(f"No frames extracted from {video_path}")

    frames = np.concatenate(chunks, axis=0)
    original_count = int(frames.shape[0])
    if original_count > max_frames:
        indices = np.linspace(0, original_count - 1, max_frames, dtype=np.int64)
        frames = frames[indices]

    return frames, {
        "fps": fps,
        "total_frames": int(total_frames),
        "raw_accumulated_frames": original_count,
        "used_frames": int(frames.shape[0]),
    }


def build_messages(
    frames: np.ndarray,
    questions: list[str],
    generated_answers: list[str],
    current_turn: int,
    system_prompt: str,
) -> list[dict[str, Any]]:
    first_question = questions[0]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": first_question},
            ],
        },
    ]
    for idx in range(current_turn):
        messages.append({"role": "assistant", "content": generated_answers[idx]})
        messages.append({"role": "user", "content": questions[idx + 1]})
    return messages


def generate_answer(
    model: Any,
    processor: Any,
    frames: np.ndarray,
    questions: list[str],
    generated_answers: list[str],
    turn: int,
    max_new_tokens: int,
    system_prompt: str,
) -> str:
    import torch

    messages = build_messages(
        frames, questions, generated_answers, turn, system_prompt
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--golden-output", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=3,
        help="Maximum turns per conversation. Use 0 or a negative value for all turns.",
    )
    parser.add_argument("--frames-per-interval", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--prompt-style",
        choices=sorted(PROMPT_STYLES),
        default="default",
        help="System prompt variant to use for inference.",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if args.adapter:
        from peft import PeftModel

    start = time.time()
    rows = load_jsonl(args.input)
    selected, skipped = select_downloaded_rows(
        rows, args.video_dir, args.max_samples, args.max_turns
    )
    if not selected:
        raise RuntimeError(f"No downloaded videos found under {args.video_dir}")

    write_jsonl(args.golden_output, selected)

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

    predictions: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []

    for row_index, row in enumerate(selected):
        sample_start = time.time()
        video_path = args.video_dir / str(row["video_path"])
        questions = [str(q) for q in row["questions"]]
        answers: list[str] = []
        frame_stats: list[dict[str, Any]] = []

        for turn in range(len(questions)):
            turn_start = time.time()
            frames, stats = load_accumulated_frames(
                video_path,
                row["video_intervals"],
                turn,
                args.frames_per_interval,
                args.max_frames,
            )
            answer = generate_answer(
                model,
                processor,
                frames,
                questions,
                answers,
                turn,
                args.max_new_tokens,
                PROMPT_STYLES[args.prompt_style],
            )
            answers.append(answer)
            stats["elapsed_seconds"] = round(time.time() - turn_start, 3)
            frame_stats.append(stats)

        pred = {"video_path": row["video_path"], "answers": answers}
        predictions.append(pred)
        write_jsonl(args.output, predictions)

        per_sample.append(
            {
                "row": row_index,
                "source_index": row.get("_source_index"),
                "video_path": row["video_path"],
                "turns": len(questions),
                "elapsed_seconds": round(time.time() - sample_start, 3),
                "frame_stats": frame_stats,
            }
        )
        print(json.dumps(per_sample[-1], ensure_ascii=False), flush=True)

    report = {
        "input": str(args.input),
        "video_dir": str(args.video_dir),
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "output": str(args.output),
        "golden_output": str(args.golden_output),
        "selected_count": len(selected),
        "skipped_missing_before_selection": len(skipped),
        "max_samples": args.max_samples,
        "max_turns": args.max_turns,
        "frames_per_interval": args.frames_per_interval,
        "max_frames": args.max_frames,
        "prompt_style": args.prompt_style,
        "system_prompt": PROMPT_STYLES[args.prompt_style],
        "elapsed_seconds": round(time.time() - start, 3),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "samples": per_sample,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
