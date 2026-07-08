#!/usr/bin/env python3
"""Export Showee streaming annotations to Wearable AI EgoConv JSONL."""

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


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"config must be a mapping: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mmss(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def video_path_for_sample(video: str, root: Path, mode: str) -> str:
    video_path = Path(video)
    if mode == "absolute":
        return str(video_path)
    if mode == "basename":
        return video_path.name
    if mode != "relative_to_root":
        raise ValueError(f"unsupported video.path_mode: {mode}")
    try:
        return str(video_path.relative_to(root))
    except ValueError:
        return str(video_path)


def segment_by_id(sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
    segments = sample.get("metadata", {}).get("temporal_segments", [])
    if not isinstance(segments, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for idx, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        seg_id = str(seg.get("segment_id", f"seg_{idx:03d}"))
        result[seg_id] = seg
    return result


def interval_for_turn(
    turn: dict[str, Any], segments: dict[str, dict[str, Any]]
) -> list[float]:
    if "interval_start_sec" in turn and "interval_end_sec" in turn:
        start = as_float(turn.get("interval_start_sec"))
        end = as_float(turn.get("interval_end_sec"))
    else:
        seg = segments.get(str(turn.get("segment_id", "")), {})
        if seg:
            start = as_float(seg.get("start_sec"))
            end = as_float(seg.get("end_sec"))
        else:
            time_sec = as_float(turn.get("time_sec"))
            start = max(0.0, time_sec - 1.0)
            end = time_sec + 1.0
    if end < start:
        start, end = end, start
    return [round(start, 3), round(end, 3)]


def duration_for_row(sample: dict[str, Any], intervals: list[list[float]]) -> float:
    segment_end = max((end for _, end in intervals), default=0.0)
    meta_duration = as_float(sample.get("metadata", {}).get("duration"))
    if meta_duration > 0 and meta_duration < segment_end:
        return round(segment_end, 3)
    return round(segment_end or meta_duration, 3)


def dialog_for_turns(
    turns: list[dict[str, Any]],
    intervals: list[list[float]],
    defaults: dict[str, Any],
) -> list[dict[str, str]]:
    dialog: list[dict[str, str]] = []
    user_role = str(defaults.get("user_role", "P1"))
    assistant_role = str(defaults.get("assistant_role", "Assistant"))
    question_type = str(defaults.get("question_type", "Multimodal_relevant"))

    for turn, (start, end) in zip(turns, intervals):
        midpoint = as_float(turn.get("time_sec"), (start + end) / 2.0)
        user_start = max(start, midpoint - 1.0)
        user_end = min(end, max(user_start, midpoint))
        assistant_start = min(end, max(user_end, midpoint))
        assistant_end = min(end, max(assistant_start, midpoint + 1.0))
        dialog.append(
            {
                "text": str(turn.get("user", "")),
                "role": user_role,
                "start_time": mmss(user_start),
                "end_time": mmss(user_end),
                "question_type": question_type,
            }
        )
        dialog.append(
            {
                "text": str(turn.get("assistant", "")),
                "role": assistant_role,
                "start_time": mmss(assistant_start),
                "end_time": mmss(assistant_end),
            }
        )
    return dialog


def make_egoconv_row(
    sample: dict[str, Any],
    video_root: Path,
    video_path_mode: str,
    defaults: dict[str, Any],
) -> dict[str, Any] | None:
    turns = sample.get("turns", [])
    if not isinstance(turns, list) or not turns:
        return None

    segments = segment_by_id(sample)
    intervals = [interval_for_turn(turn, segments) for turn in turns]
    questions = [str(turn.get("user", "")) for turn in turns]
    answers = [str(turn.get("assistant", "")) for turn in turns]
    meta = sample.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    task_field = str(defaults.get("task_field", "task_name"))
    task = str(meta.get(task_field) or meta.get("task_name") or meta.get("task_id") or "")

    return {
        "video_path": video_path_for_sample(str(sample["video"]), video_root, video_path_mode),
        "duration_in_sec": duration_for_row(sample, intervals),
        "video_intervals": intervals,
        "questions": questions,
        "answers": answers,
        "task": task,
        "dialog": dialog_for_turns(turns, intervals, defaults),
    }


def validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    required = {
        "video_path",
        "duration_in_sec",
        "video_intervals",
        "questions",
        "answers",
        "task",
        "dialog",
    }
    for idx, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            errors.append(f"row {idx}: missing {missing}")
            continue
        n = len(row["questions"])
        if len(row["answers"]) != n or len(row["video_intervals"]) != n:
            errors.append(f"row {idx}: questions/answers/video_intervals length mismatch")
        if len(row["dialog"]) < n * 2:
            errors.append(f"row {idx}: dialog shorter than expected")
        for interval in row["video_intervals"]:
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or float(interval[1]) < float(interval[0])
            ):
                errors.append(f"row {idx}: bad interval {interval}")
                break
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/wearable_egoconv_v001.yaml",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = read_yaml(config_path)
    input_cfg = config.get("input", {})
    if not isinstance(input_cfg, dict):
        raise TypeError("config.input must be a mapping")
    source_format = str(input_cfg.get("format", "streaming_turns"))
    if source_format != "streaming_turns":
        raise ValueError("only input.format='streaming_turns' is supported")
    source_dataset = resolve_path(input_cfg["dataset"])

    video_cfg = config.get("video", {})
    if not isinstance(video_cfg, dict):
        raise TypeError("config.video must be a mapping")
    video_root = resolve_path(video_cfg.get("root", "."))
    video_path_mode = str(video_cfg.get("path_mode", "relative_to_root"))

    output_cfg = config.get("output", {})
    if not isinstance(output_cfg, dict):
        raise TypeError("config.output must be a mapping")
    output_jsonl = resolve_path(output_cfg.get("jsonl", "data/processed/wearable_egoconv_v001.jsonl"))
    report_path = resolve_path(output_cfg.get("report", "runs/wearable_egoconv_v001/export_report.json"))
    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}

    source = read_json(source_dataset)
    if not isinstance(source, list):
        raise TypeError(f"source dataset must be a JSON list: {source_dataset}")

    rows = [
        row
        for row in (
            make_egoconv_row(sample, video_root, video_path_mode, defaults)
            for sample in source
        )
        if row is not None
    ]
    errors = validate_rows(rows)
    if errors:
        raise ValueError("EgoConv export validation failed:\n" + "\n".join(errors[:20]))

    write_jsonl(output_jsonl, rows)
    turn_counts = [len(row["questions"]) for row in rows]
    report = {
        "version": str(config.get("version", "wearable_egoconv_v001")),
        "config": str(config_path),
        "source_dataset": str(source_dataset),
        "source_format": source_format,
        "video_root": str(video_root),
        "video_path_mode": video_path_mode,
        "source_count": len(source),
        "row_count": len(rows),
        "turn_count": sum(turn_counts),
        "min_turns_per_row": min(turn_counts) if turn_counts else 0,
        "max_turns_per_row": max(turn_counts) if turn_counts else 0,
        "output_jsonl": str(output_jsonl),
        "validation_errors": 0,
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
