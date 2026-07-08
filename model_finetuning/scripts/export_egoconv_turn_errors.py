#!/usr/bin/env python3
"""Export per-turn EgoConv errors to CSV for manual review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()

    data = json.load(args.results.open("r", encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for conv in data.get("per_row", []):
        questions = conv.get("questions", [])
        gold = conv.get("gold_answers", [])
        pred = conv.get("pred_answers", [])
        bleu = conv.get("bleu_per_turn", [])
        for turn, question in enumerate(questions):
            rows.append(
                {
                    "conversation_index": conv.get("index"),
                    "video_path": conv.get("video_path"),
                    "category": conv.get("category") or conv.get("task"),
                    "turn": turn,
                    "turn_bleu": bleu[turn] if turn < len(bleu) else "",
                    "question": question,
                    "gold_answer": gold[turn] if turn < len(gold) else "",
                    "pred_answer": pred[turn] if turn < len(pred) else "",
                }
            )

    rows.sort(key=lambda r: (float(r["turn_bleu"]) if r["turn_bleu"] != "" else -1.0))
    if args.limit > 0:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "conversation_index",
                "video_path",
                "category",
                "turn",
                "turn_bleu",
                "question",
                "gold_answer",
                "pred_answer",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"wrote": str(args.output), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
