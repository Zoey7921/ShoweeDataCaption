#!/usr/bin/env python3
"""Build wearable EgoConv validation sets from the reviewed JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/data1/shared_data/qwen3vl-showeeData")
RAW_ROOT = PROJECT_ROOT / "data/raw/ShoweeHandv2/raw"
DEFAULT_INPUT = PROJECT_ROOT / "data/processed/wearable_egoconv_v001.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval/sets/wearable_egoconv_v001_reviewed_val20.json"


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


def row_dialog_pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
    questions = row.get("questions") or []
    answers = row.get("answers") or []
    pairs = [(str(q), str(a)) for q, a in zip(questions, answers)]
    if pairs:
        return pairs

    dialog = row.get("dialog") or []
    parsed: list[tuple[str, str]] = []
    pending_user: str | None = None
    for turn in dialog:
        role = str(turn.get("role", ""))
        text = str(turn.get("text", ""))
        if role == "Assistant" and pending_user is not None:
            parsed.append((pending_user, text))
            pending_user = None
        elif role != "Assistant":
            pending_user = text
    return parsed


def target_interval(row: dict[str, Any], turn_index: int) -> list[float] | None:
    intervals = row.get("video_intervals") or []
    if not intervals:
        return None
    idx = min(turn_index, len(intervals) - 1)
    interval = intervals[idx]
    if not isinstance(interval, list) or len(interval) != 2:
        return None
    return [float(interval[0]), float(interval[1])]


def build_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        pairs = row_dialog_pairs(row)
        if not pairs:
            raise ValueError(f"Row has no question/answer pairs: line {row['_line_no']}")

        video = RAW_ROOT / str(row["video_path"])
        if not video.is_file():
            raise FileNotFoundError(f"Video does not exist: {video}")

        history: list[dict[str, str]] = []
        for turn_index, (question, answer) in enumerate(pairs):
            messages = history + [{"role": "user", "text": question}]
            eval_rows.append(
                {
                    "id": f"wearable_egoconv_v001_line{row['_line_no']:04d}_turn{turn_index + 1}",
                    "video": str(video),
                    "question": question,
                    "reference_answer": answer,
                    "messages": messages,
                    "metadata": {
                        "dataset": "ShoweeHandv2",
                        "source_format": "wearable_egoconv_v001",
                        "source_jsonl": str(DEFAULT_INPUT),
                        "source_line_no": row["_line_no"],
                        "relative_video_path": row["video_path"],
                        "task": row.get("task"),
                        "duration_in_sec": row.get("duration_in_sec"),
                        "video_intervals": row.get("video_intervals") or [],
                        "target_interval": target_interval(row, turn_index),
                        "turn_index": turn_index,
                        "turn_count": len(pairs),
                        "review_status": "reviewed_val",
                    },
                }
            )
            history.extend(
                [
                    {"role": "user", "text": question},
                    {"role": "assistant", "text": answer},
                ]
            )
    return eval_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-line", type=int, default=None)
    parser.add_argument("--end-line", type=int, default=None)
    parser.add_argument("--tail", type=int, default=20)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.start_line is not None or args.end_line is not None:
        start = args.start_line or 1
        end = args.end_line or len(rows)
        selected = [row for row in rows if start <= int(row["_line_no"]) <= end]
    else:
        selected = rows[-args.tail :]

    eval_rows = build_eval_rows(selected)
    write_json(args.output, eval_rows)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "source_rows": len(selected),
        "eval_turns": len(eval_rows),
        "line_range": [selected[0]["_line_no"], selected[-1]["_line_no"]] if selected else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
