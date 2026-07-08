#!/usr/bin/env python3
"""Convert wearable EgoConv JSONL data to the local Qwen3-VL LoRA format."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/data1/shared_data/qwen3vl-showeeData")
RAW_ROOT = PROJECT_ROOT / "data/raw/ShoweeHandv2/raw"
DEFAULT_INPUT = PROJECT_ROOT / "data/processed/wearable_egoconv_v001.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/wearable_egoconv_v001_train.json"
DEFAULT_REPORT = PROJECT_ROOT / "runs/puzuo/wearable_egoconv_v001_build/build_report.json"


def slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_./-]+", "_", text)
    return text.replace("/", "_").replace(".", "_").strip("_")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def dialog_to_conversations(row: dict[str, Any]) -> list[dict[str, str]]:
    dialog = row.get("dialog") or []
    if not dialog:
        questions = row.get("questions") or []
        answers = row.get("answers") or []
        dialog = []
        for question, answer in zip(questions, answers):
            dialog.append({"role": "P1", "text": question})
            dialog.append({"role": "Assistant", "text": answer})

    conversations: list[dict[str, str]] = []
    video_attached = False
    for turn in dialog:
        role = str(turn.get("role", ""))
        text = str(turn.get("text", ""))
        if role == "Assistant":
            conversations.append({"from": "assistant", "value": text})
        else:
            if not video_attached:
                text = "<video>\n" + text
                video_attached = True
            conversations.append({"from": "user", "value": text})

    if not conversations or not any(turn["from"] == "assistant" for turn in conversations):
        raise ValueError(f"Row has no usable assistant turns: line {row.get('_line_no')}")
    return conversations


def convert_row(row: dict[str, Any], input_path: Path, review_status: str) -> dict[str, Any]:
    rel_video = Path(str(row["video_path"]))
    video = RAW_ROOT / rel_video
    if not video.is_file():
        raise FileNotFoundError(f"Video does not exist: {video}")
    intervals = row.get("video_intervals") or []
    return {
        "id": f"wearable_egoconv_v001_{row['_line_no']:04d}_{slug(str(row['video_path']))}",
        "video": str(video),
        "metadata": {
            "dataset": "ShoweeHandv2",
            "source_format": "wearable_egoconv_v001",
            "source_jsonl": str(input_path),
            "source_line_no": row["_line_no"],
            "relative_video_path": str(row["video_path"]),
            "task": row.get("task"),
            "duration_in_sec": row.get("duration_in_sec"),
            "video_intervals": intervals,
            "dialog_turns": len(row.get("dialog") or []),
            "annotation_source": "wearable_egoconv_v001_pipeline",
            "review_status": review_status,
        },
        "conversations": dialog_to_conversations(row),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--review-status", default="coarse_warmup")
    parser.add_argument("--start-line", type=int, default=None)
    parser.add_argument("--end-line", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.start_line is not None or args.end_line is not None:
        start = args.start_line or 1
        end = args.end_line or len(rows)
        rows = [row for row in rows if start <= int(row["_line_no"]) <= end]
    samples = [convert_row(row, args.input, args.review_status) for row in rows]
    write_json(args.output, samples)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "sample_count": len(samples),
        "source_line_range": [
            min((int(sample["metadata"]["source_line_no"]) for sample in samples), default=None),
            max((int(sample["metadata"]["source_line_no"]) for sample in samples), default=None),
        ],
        "single_turn_samples": sum(1 for sample in samples if len(sample["conversations"]) == 2),
        "multi_turn_samples": sum(1 for sample in samples if len(sample["conversations"]) > 2),
        "tasks": sorted({str(sample["metadata"].get("task")) for sample in samples}),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
