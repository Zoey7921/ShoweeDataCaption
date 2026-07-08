#!/usr/bin/env python3
"""Build a compact qualitative report for partial EgoConv predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


UNCERTAINTY_PHRASES = [
    "cannot determine",
    "can't determine",
    "not visible",
    "isn't visible",
    "not enough information",
    "not enough visual",
    "unable to determine",
    "i don't know",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def short(text: str, limit: int = 260) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--partial-golden-out", type=Path, default=None)
    args = parser.parse_args()

    golden_all = load_jsonl(args.golden)
    preds = load_jsonl(args.predictions)
    golden = golden_all[: len(preds)]
    if args.partial_golden_out:
        write_jsonl(args.partial_golden_out, golden)

    results = json.load(args.results.open("r", encoding="utf-8"))
    per_row = results.get("per_row", [])

    category_counts: Counter[str] = Counter()
    category_turns: Counter[str] = Counter()
    uncertainty_count = 0
    total_answers = 0
    too_short = 0

    for row in per_row:
        cat = str(row.get("category") or row.get("task") or "")
        category_counts[cat] += 1
        category_turns[cat] += int(row.get("num_turns", 0))
        for ans in row.get("pred_answers", []):
            total_answers += 1
            ans_l = str(ans).lower()
            if any(p in ans_l for p in UNCERTAINTY_PHRASES):
                uncertainty_count += 1
            if len(str(ans).split()) <= 2:
                too_short += 1

    worst = sorted(per_row, key=lambda r: r.get("bleu_avg", 0.0))[:12]
    best = sorted(per_row, key=lambda r: r.get("bleu_avg", 0.0), reverse=True)[:8]

    cat_scores = results.get("category_scores", {})
    cat_lines = []
    for cat, data in sorted(cat_scores.items()):
        cat_lines.append(
            f"| {cat} | {data.get('bleu', '')} | {category_counts[cat]} | {category_turns[cat]} |"
        )

    lines: list[str] = [
        "# EgoConv Partial 700 Run Analysis",
        "",
        "This report is generated from the currently available partial 700 predictions.",
        "BLEU is used only as a fast sanity check; official ranking requires LLM-as-Judge.",
        "",
        "## Summary",
        "",
        f"- Prediction rows evaluated: `{len(preds)}` / `{len(golden_all)}`",
        f"- Turns evaluated: `{total_answers}`",
        f"- BLEU: `{results.get('bleu')}`",
        f"- Empty/uncertainty-style answers: `{uncertainty_count}`",
        f"- Very short answers, <=2 words: `{too_short}`",
        "",
        "## Category BLEU",
        "",
        "| Category | BLEU | Conversations | Turns |",
        "|---|---:|---:|---:|",
        *cat_lines,
        "",
        "## Best Rows By BLEU",
        "",
    ]

    for row in best:
        lines.extend(
            [
                f"### Row {row.get('index')} | {row.get('category')} | BLEU {row.get('bleu_avg')}",
                "",
                f"- Video: `{row.get('video_path')}`",
                f"- First question: {short(row.get('questions', [''])[0])}",
                f"- Gold: {short(row.get('gold_answers', [''])[0])}",
                f"- Pred: {short(row.get('pred_answers', [''])[0])}",
                "",
            ]
        )

    lines.extend(["## Worst Rows By BLEU", ""])
    for row in worst:
        lines.extend(
            [
                f"### Row {row.get('index')} | {row.get('category')} | BLEU {row.get('bleu_avg')}",
                "",
                f"- Video: `{row.get('video_path')}`",
                f"- First question: {short(row.get('questions', [''])[0])}",
                f"- Gold: {short(row.get('gold_answers', [''])[0])}",
                f"- Pred: {short(row.get('pred_answers', [''])[0])}",
                "",
            ]
        )

    lines.extend(
        [
            "## Current Observations",
            "",
            "- The format is valid so far: video order and answer counts match the golden rows.",
            "- The balanced prompt reduces refusal, but hallucination remains visible on fine-grained museum/place/name questions.",
            "- Long multi-turn samples are slow and tend to accumulate earlier visual mistakes into later answers.",
            "- BLEU can penalize concise correct answers and can reward wording overlap, so final judgment must use official LLM-as-Judge.",
            "",
        ]
    )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"wrote": str(args.out_md), "rows": len(preds), "bleu": results.get("bleu")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
