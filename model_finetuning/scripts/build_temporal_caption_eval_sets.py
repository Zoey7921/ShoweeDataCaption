#!/usr/bin/env python3
"""Build temporal-caption eval sets from ShareGPT-style conversations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/data1/shared_data/qwen3vl-showeeData")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def conversation_turns(sample: dict[str, Any]) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for turn in sample.get("conversations", []):
        role = str(turn.get("from", ""))
        value = str(turn.get("value", ""))
        turns.append((role, value))
    return turns


def build_rows(samples: list[dict[str, Any]], question_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        pairs: list[tuple[str, str]] = []
        pending_user: str | None = None
        for role, value in conversation_turns(sample):
            if role == "user":
                pending_user = value.replace("<video>\n", "")
            elif role == "assistant" and pending_user is not None:
                pairs.append((pending_user, value))
                pending_user = None

        if question_index >= len(pairs):
            raise ValueError(f"Sample lacks question index {question_index}: {sample.get('id')}")
        question, answer = pairs[question_index]
        rows.append(
            {
                "id": sample.get("id"),
                "video": sample["video"],
                "question": question,
                "reference_answer": answer,
                "metadata": sample.get("metadata", {}),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, default=PROJECT_ROOT / "data/processed/temporal_caption_v001_val.json")
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "data/processed/temporal_caption_v001_test.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "eval/sets")
    parser.add_argument(
        "--question-index",
        type=int,
        default=2,
        help="0-based user/assistant pair index. Default 2 is the temporal segment question.",
    )
    parser.add_argument("--prefix", default="temporal_caption_v001")
    args = parser.parse_args()

    for split, path in [("val", args.val), ("test", args.test)]:
        rows = build_rows(load_json(path), args.question_index)
        output = args.output_dir / f"{args.prefix}_{split}_temporal.json"
        write_json(output, rows)
        print(json.dumps({"split": split, "count": len(rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
