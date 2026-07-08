#!/usr/bin/env python3
"""Build streaming interaction samples from temporal caption annotations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(
    os.environ.get("SHOWEE_PIPELINE_ROOT", Path(__file__).resolve().parents[1])
)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"config must be a mapping: {path}")
    return data


def input_config(config: dict[str, Any]) -> tuple[Path, str]:
    input_cfg = config.get("input", {})
    if input_cfg is None:
        input_cfg = {}
    if not isinstance(input_cfg, dict):
        raise TypeError("config.input must be a mapping when provided")

    dataset = input_cfg.get("dataset", config.get("source_dataset"))
    if not dataset:
        raise KeyError("config must set input.dataset or source_dataset")

    source_format = str(input_cfg.get("format", "temporal_segments"))
    if source_format != "temporal_segments":
        raise ValueError(
            "only input.format='temporal_segments' is supported; "
            f"got {source_format!r}"
        )
    return resolve_path(dataset), source_format


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def segment_midpoint(segment: dict[str, Any]) -> float:
    start = as_float(segment.get("start_sec"))
    end = as_float(segment.get("end_sec"))
    return round((start + end) / 2.0, 3)


def frame_index(time_sec: float, fps: float = 60.0) -> int:
    return int(round(time_sec * fps))


def first_user_question(turn_idx: int) -> str:
    if turn_idx == 0:
        return "我现在应该怎么做？"
    if turn_idx == 1:
        return "我已经做到这一步了，下一步呢？"
    return "这样继续做对吗？接下来该注意什么？"


def next_segment(segments: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    return segments[idx + 1] if idx + 1 < len(segments) else None


def assistant_answer(
    sample: dict[str, Any],
    segment: dict[str, Any],
    next_seg: dict[str, Any] | None,
    turn_idx: int,
) -> str:
    meta = sample["metadata"]
    task_name = str(meta.get("task_name") or meta.get("task_id"))
    current = str(segment.get("description") or segment.get("label") or task_name)
    confidence = str(segment.get("confidence", ""))
    source = str(segment.get("boundary_source", ""))
    prefix = "从当前时间点看，" if turn_idx == 0 else "结合前面的对话和当前画面，"
    if next_seg:
        nxt = str(next_seg.get("description") or next_seg.get("label") or "继续完成下一阶段动作")
        action = f"接下来进入下一阶段：{nxt}"
    else:
        action = "如果当前动作已经完成，可以保持稳定姿态；如果是重复练习，可以从起始动作再做一轮。"
    caution = ""
    if confidence != "verified":
        caution = " 当前时间点是初始粗标注，精确边界仍需人工复核。"
    return f"{prefix}你处在「{task_name}」任务的阶段：{current} {action}{caution}"


def build_turns(sample: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    segments = sample.get("metadata", {}).get("temporal_segments", [])
    if not isinstance(segments, list) or not segments:
        return []
    max_turns = int(config.get("turns", {}).get("max_turns_per_sample", 3))
    selected_segments = segments[:max_turns]
    turns: list[dict[str, Any]] = []
    history: list[dict[str, str]] = []
    for idx, segment in enumerate(selected_segments):
        time_sec = segment_midpoint(segment)
        user_text = first_user_question(idx)
        answer = assistant_answer(sample, segment, next_segment(segments, idx), idx)
        turn = {
            "turn_id": f"turn_{idx:03d}",
            "time_sec": time_sec,
            "frame_index": frame_index(time_sec),
            "segment_id": str(segment.get("segment_id", f"seg_{idx:03d}")),
            "history": list(history),
            "user": user_text,
            "assistant": answer,
        }
        turns.append(turn)
        history.append({"user": user_text, "assistant": answer})
    return turns


def make_stream_sample(sample: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    meta = json.loads(json.dumps(sample.get("metadata", {}), ensure_ascii=False))
    meta["streaming_source_id"] = sample.get("id", "")
    meta["review_status"] = str(config.get("review_status", "streaming_unreviewed"))
    meta["interaction_type"] = "time_node_guidance"
    turns = build_turns(sample, config)
    return {
        "id": f"{sample['id']}_stream",
        "video": sample["video"],
        "metadata": meta,
        "turns": turns,
    }


def to_messages_sample(stream_sample: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    for turn in stream_sample["turns"]:
        messages.append(
            {
                "role": "user",
                "content": f"<video time={turn['time_sec']:.2f}s frame={turn['frame_index']}>\n{turn['user']}",
            }
        )
        messages.append({"role": "assistant", "content": turn["assistant"]})
    return {
        "id": stream_sample["id"],
        "video": stream_sample["video"],
        "metadata": stream_sample["metadata"],
        "messages": messages,
    }


def write_review(path: Path, samples: list[dict[str, Any]]) -> None:
    lines = [
        "# Streaming Interaction v001 人工复核表",
        "",
        "状态可填：streaming_accepted / streaming_edited / streaming_rejected。",
        "重点检查：当前时间点画面、历史对话和下一步建议是否一致。",
        "",
    ]
    for idx, sample in enumerate(samples, start=1):
        meta = sample["metadata"]
        lines.extend(
            [
                f"## {idx}. {sample['id']}",
                "",
                f"- status: {meta.get('review_status', '')}",
                f"- video: {sample['video']}",
                f"- source_id: {meta.get('streaming_source_id', '')}",
                f"- task: {meta.get('task_id', '')} / {meta.get('task_name', '')}",
                "- turns:",
            ]
        )
        for turn in sample["turns"]:
            lines.extend(
                [
                    f"  - {turn['turn_id']} @ {turn['time_sec']:.2f}s frame={turn['frame_index']} segment={turn['segment_id']}",
                    f"    user: {turn['user']}",
                    f"    assistant: {turn['assistant']}",
                ]
            )
        lines.extend(["", "review_notes:", "", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/streaming_interaction_v001.yaml")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = read_yaml(config_path)
    version = str(config.get("version", "streaming_interaction_v001"))
    source_dataset, source_format = input_config(config)
    output = config.get("outputs", {})
    processed_dir = resolve_path(output.get("processed_dir", "data/processed"))
    review_dir = resolve_path(output.get("review_dir", "eval/human_review"))
    run_dir = resolve_path(output.get("run_dir", f"runs/{version}"))

    source_samples = read_json(source_dataset)
    if not isinstance(source_samples, list):
        raise TypeError(f"source dataset must be a JSON list: {source_dataset}")
    stream_samples = [make_stream_sample(sample, config) for sample in source_samples]
    stream_samples = [sample for sample in stream_samples if sample["turns"]]
    message_samples = [to_messages_sample(sample) for sample in stream_samples]

    write_json(processed_dir / f"{version}.json", stream_samples)
    write_json(processed_dir / f"{version}_messages.json", message_samples)
    write_review(review_dir / f"{version}_review.md", stream_samples)

    turn_counts = [len(sample["turns"]) for sample in stream_samples]
    report = {
        "version": version,
        "config": str(config_path),
        "source_dataset": str(source_dataset),
        "input_format": source_format,
        "source_count": len(source_samples),
        "sample_count": len(stream_samples),
        "message_sample_count": len(message_samples),
        "turn_count": sum(turn_counts),
        "min_turns_per_sample": min(turn_counts) if turn_counts else 0,
        "max_turns_per_sample": max(turn_counts) if turn_counts else 0,
        "outputs": {
            "canonical": str(processed_dir / f"{version}.json"),
            "messages": str(processed_dir / f"{version}_messages.json"),
            "review": str(review_dir / f"{version}_review.md"),
        },
    }
    write_json(run_dir / "build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
