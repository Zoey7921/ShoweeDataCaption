#!/usr/bin/env python3
"""Validate EgoConv prediction JSONL before evaluation/submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    golden = load_jsonl(args.golden)
    preds = load_jsonl(args.predictions)

    errors: list[str] = []
    warnings: list[str] = []

    if len(preds) != len(golden):
        msg = f"row_count predictions={len(preds)} golden={len(golden)}"
        if args.allow_partial and len(preds) < len(golden):
            warnings.append(msg)
        else:
            errors.append(msg)

    n = min(len(golden), len(preds))
    total_turns = 0
    empty_answers = 0
    mismatched_paths = 0
    mismatched_turn_counts = 0

    for i in range(n):
        g = golden[i]
        p = preds[i]
        if p.get("video_path") != g.get("video_path"):
            mismatched_paths += 1
            if mismatched_paths <= 5:
                errors.append(
                    "video_path mismatch at row "
                    f"{i}: pred={p.get('video_path')} gold={g.get('video_path')}"
                )

        gold_turns = len(g.get("questions", []))
        pred_answers = p.get("answers")
        if not isinstance(pred_answers, list):
            errors.append(f"row {i}: answers is not a list")
            continue
        if len(pred_answers) != gold_turns:
            mismatched_turn_counts += 1
            if mismatched_turn_counts <= 5:
                errors.append(
                    f"row {i}: answer_count={len(pred_answers)} expected={gold_turns}"
                )

        total_turns += len(pred_answers)
        for answer in pred_answers:
            if not str(answer).strip():
                empty_answers += 1

    report = {
        "golden_rows": len(golden),
        "prediction_rows": len(preds),
        "checked_rows": n,
        "prediction_turns_checked": total_turns,
        "empty_answers": empty_answers,
        "mismatched_paths": mismatched_paths,
        "mismatched_turn_counts": mismatched_turn_counts,
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
