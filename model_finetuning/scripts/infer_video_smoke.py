#!/usr/bin/env python3
"""Run one Qwen3-VL video inference smoke test on ShoweeHandv2.

This script reads the first item from data/processed/smoke_train.json and asks
Qwen3-VL to describe the showee_head video. It is intentionally small and keeps
all paths explicit for easy reproduction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path("/data1/shared_data/qwen3vl-showeeData")
DEFAULT_MODEL = PROJECT_ROOT / "models/base/Qwen3-VL-8B-Instruct"
DEFAULT_DATA = PROJECT_ROOT / "data/processed/smoke_train.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs/puzuo/video_smoke_infer/output.json"


def load_first_sample(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    return data[0]


def load_video_frames(path: Path, num_frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode a small, fixed set of RGB frames for a stable smoke test."""
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0))
    total_frames = len(reader)
    if total_frames == 0:
        raise ValueError(f"Video has no decodable frames: {path}")

    sample_count = min(num_frames, total_frames)
    fps = float(reader.get_avg_fps())
    indices = np.linspace(0, total_frames - 1, sample_count, dtype=np.int64)
    frame_times = [
        {
            "sample_index": sample_index,
            "frame_index": int(frame_index),
            "time_sec": float(frame_index / fps) if fps > 0 else None,
        }
        for sample_index, frame_index in enumerate(indices.tolist())
    ]
    metadata = {
        "total_frames": int(total_frames),
        "fps": fps,
        "duration_seconds_est": float(total_frames / fps) if fps > 0 else None,
        "sampled_frames": frame_times,
    }
    return reader.get_batch(indices).asnumpy(), metadata


def build_temporal_prompt(video_metadata: dict[str, Any]) -> str:
    sampled_times = ", ".join(
        f"{frame['sample_index']}={frame['time_sec']:.2f}s"
        for frame in video_metadata["sampled_frames"]
        if frame["time_sec"] is not None
    )
    duration = video_metadata["duration_seconds_est"]
    return (
        "请按时间段描述这个视频中正在执行的手势任务。"
        "你看到的是按时间顺序均匀抽样的帧，抽样帧时间点如下："
        f"{sampled_times}。"
        f"视频总时长约 {duration:.2f} 秒。"
        "请只输出 JSON，不要输出 Markdown。"
        "必须严格输出 4 个 segments，时间边界固定为："
        f"0.00-5.00、5.00-10.00、10.00-15.00、15.00-{duration:.2f}。"
        "JSON 格式必须是："
        "{\"segments\":["
        "{\"start_sec\":0.0,\"end_sec\":5.0,\"action\":\"\",\"evidence\":\"\"},"
        "{\"start_sec\":5.0,\"end_sec\":10.0,\"action\":\"\",\"evidence\":\"\"},"
        "{\"start_sec\":10.0,\"end_sec\":15.0,\"action\":\"\",\"evidence\":\"\"},"
        f"{{\"start_sec\":15.0,\"end_sec\":{duration:.2f},\"action\":\"\",\"evidence\":\"\"}}],"
        "\"summary\":\"\"}。"
        "每段 action 和 evidence 都限制在 30 个汉字以内。"
        "即使动作没有变化，也必须按这 4 个固定时间段分别说明持续姿态。"
    )


def parse_temporal_output(output_text: list[str]) -> dict[str, Any] | None:
    if not output_text:
        return None
    text = output_text[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if not isinstance(parsed.get("segments"), list):
        return None
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=16)
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    sample = load_first_sample(args.data)
    video_path = Path(sample["video"])
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    video_frames, video_metadata = load_video_frames(video_path, args.num_frames)

    prompt = build_temporal_prompt(video_metadata)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    start = time.time()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model),
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)

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
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    temporal_output = parse_temporal_output(output_text)

    elapsed = time.time() - start
    result = {
        "sample_id": sample.get("id"),
        "model": str(args.model),
        "video": str(video_path),
        "num_frames": int(video_frames.shape[0]),
        "video_total_frames": video_metadata["total_frames"],
        "video_fps": video_metadata["fps"],
        "video_duration_seconds_est": video_metadata["duration_seconds_est"],
        "sampled_frames": video_metadata["sampled_frames"],
        "prompt": prompt,
        "output": output_text,
        "temporal_segments": temporal_output.get("segments") if temporal_output else None,
        "temporal_summary": temporal_output.get("summary") if temporal_output else None,
        "elapsed_seconds": elapsed,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
