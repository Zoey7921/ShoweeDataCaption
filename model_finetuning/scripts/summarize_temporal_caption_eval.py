#!/usr/bin/env python3
"""Summarize temporal-caption prediction files with lightweight string metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def time_variants(start: float, end: float) -> set[str]:
    return {
        f"{start:.2f}-{end:.2f}s",
        f"{start:.1f}-{end:.1f}s",
        f"{start:g}-{end:g}s",
        f"{start:.2f}秒-{end:.2f}秒",
        f"{start:g}秒-{end:g}秒",
    }


def contains_any(text: str, variants: set[str]) -> bool:
    norm = normalize(text)
    return any(normalize(item) in norm for item in variants)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    task_hits = 0
    all_time_hits = 0
    avg_time_hit_rate = 0.0
    all_label_hits = 0
    avg_label_hit_rate = 0.0

    for row in rows:
        pred = str(row.get("prediction", ""))
        meta = row.get("metadata") or {}
        task_name = str(meta.get("task_name", ""))
        task_id = str(meta.get("task_id", ""))
        segments = meta.get("temporal_segments") or []

        task_hit = bool(task_name and normalize(task_name) in normalize(pred)) or bool(
            task_id and normalize(task_id) in normalize(pred)
        )
        task_hits += int(task_hit)

        time_hits = 0
        label_hits = 0
        for seg in segments:
            start = float(seg.get("start_sec", 0.0))
            end = float(seg.get("end_sec", 0.0))
            time_hits += int(contains_any(pred, time_variants(start, end)))
            label = str(seg.get("label", ""))
            label_hits += int(bool(label and normalize(label) in normalize(pred)))

        seg_count = len(segments)
        time_hit_rate = time_hits / seg_count if seg_count else 0.0
        label_hit_rate = label_hits / seg_count if seg_count else 0.0
        all_time_hit = bool(seg_count and time_hits == seg_count)
        all_label_hit = bool(seg_count and label_hits == seg_count)
        all_time_hits += int(all_time_hit)
        all_label_hits += int(all_label_hit)
        avg_time_hit_rate += time_hit_rate
        avg_label_hit_rate += label_hit_rate

        details.append(
            {
                "id": row.get("id"),
                "task_id": task_id,
                "task_name": task_name,
                "segment_count": seg_count,
                "task_hit": task_hit,
                "time_hits": time_hits,
                "time_hit_rate": time_hit_rate,
                "all_time_hit": all_time_hit,
                "label_hits": label_hits,
                "label_hit_rate": label_hit_rate,
                "all_label_hit": all_label_hit,
            }
        )

    count = len(rows)
    return {
        "eval_type": "temporal_caption",
        "count": count,
        "task_hits": task_hits,
        "task_hit_rate": task_hits / count if count else 0.0,
        "all_time_hits": all_time_hits,
        "all_time_hit_rate": all_time_hits / count if count else 0.0,
        "avg_time_hit_rate": avg_time_hit_rate / count if count else 0.0,
        "all_label_hits": all_label_hits,
        "all_label_hit_rate": all_label_hits / count if count else 0.0,
        "avg_label_hit_rate": avg_label_hit_rate / count if count else 0.0,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_jsonl", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = summarize(read_jsonl(args.prediction_jsonl))
    summary["prediction_jsonl"] = str(args.prediction_jsonl)
    output = args.output or args.prediction_jsonl.with_suffix(args.prediction_jsonl.suffix + ".metrics.json")
    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
