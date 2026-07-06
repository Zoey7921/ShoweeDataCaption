#!/usr/bin/env python3
"""Add coarse temporal action segments to Showee ShareGPT-style samples."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(
    os.environ.get("SHOWEE_PIPELINE_ROOT", Path(__file__).resolve().parents[1])
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def duration_sec(meta: dict[str, Any]) -> float:
    for key in ["duration_seconds_est", "duration_sec", "duration"]:
        value = as_float(meta.get(key), 0.0)
        if value > 0:
            return round(value, 3)
    return 0.0


def motion_hint(task_desc: str) -> str:
    hints: list[str] = []
    if any(x in task_desc for x in ["转动手腕", "手腕"]):
        hints.append("手腕转动或朝向变化")
    if any(x in task_desc for x in ["弯曲", "屈"]):
        hints.append("手指弯曲程度变化")
    if any(x in task_desc for x in ["伸展", "伸出", "张开"]):
        hints.append("手指伸展或张开")
    if any(x in task_desc for x in ["并拢", "分合", "分和"]):
        hints.append("手指分合")
    if any(x in task_desc for x in ["捏合", "触", "touch"]):
        hints.append("指尖接触或捏合")
    if any(x in task_desc for x in ["顺序", "波浪", "依次"]):
        hints.append("按序连续变化")
    return "、".join(hints) if hints else "手部姿态保持与轻微变化"


def make_segments(meta: dict[str, Any]) -> list[dict[str, Any]]:
    task_name = str(meta.get("task_name") or meta.get("task_id") or "unknown_task")
    task_desc = str(meta.get("task_desc") or task_name)
    end = duration_sec(meta)
    if end <= 0:
        end = 20.0
    hint = motion_hint(task_desc)
    return [
        {
            "segment_id": "seg_000",
            "start_sec": 0.0,
            "end_sec": round(end, 3),
            "label": task_name,
            "description": f"整段视频围绕{task_name}任务展开，主要观察点是{hint}。",
            "boundary_source": "metadata_duration",
            "confidence": "coarse",
            "notes": "粗粒度任务级时间段；尚未逐帧人工标注动作边界。",
        }
    ]


def temporal_answer(meta: dict[str, Any], original: str) -> str:
    if re.match(r"^\s*\d+(?:\.\d+)?-\d+(?:\.\d+)?s[：:]", original):
        return original
    segments = meta.get("temporal_segments") or make_segments(meta)
    lines = []
    for seg in segments:
        lines.append(
            f"{as_float(seg.get('start_sec')):.2f}-{as_float(seg.get('end_sec')):.2f}s："
            f"{seg.get('description', seg.get('label', '手势动作'))}"
        )
    suffix = "以上时间点为粗粒度任务级标注，精确起止边界仍需逐帧复核。"
    if original:
        return f"{' '.join(lines)} {original} {suffix}"
    return f"{' '.join(lines)} {suffix}"


def update_conversations(sample: dict[str, Any]) -> None:
    conversations = sample.get("conversations")
    if not isinstance(conversations, list):
        return
    for idx, turn in enumerate(conversations):
        if (
            isinstance(turn, dict)
            and turn.get("from") == "user"
            and "动作从头到尾有什么变化" in str(turn.get("value", ""))
            and idx + 1 < len(conversations)
            and isinstance(conversations[idx + 1], dict)
        ):
            original = str(conversations[idx + 1].get("value", ""))
            conversations[idx + 1]["value"] = temporal_answer(sample["metadata"], original)
            return


def add_segments(sample: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(sample, ensure_ascii=False))
    meta = out.setdefault("metadata", {})
    if "temporal_segments" not in meta:
        meta["temporal_segments"] = make_segments(meta)
    update_conversations(out)
    return out


def default_output(path: Path) -> Path:
    return path.with_name(re.sub(r"\.json$", "_temporal.json", path.name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    for input_path in args.inputs:
        path = input_path if input_path.is_absolute() else PROJECT_ROOT / input_path
        data = read_json(path)
        if not isinstance(data, list):
            raise TypeError(f"expected a JSON list: {path}")
        updated = [add_segments(sample) for sample in data]
        output = path if args.in_place else default_output(path)
        write_json(output, updated)
        print(json.dumps({"input": str(path), "output": str(output), "count": len(updated)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
