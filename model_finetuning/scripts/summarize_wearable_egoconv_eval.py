#!/usr/bin/env python3
"""Summarize wearable EgoConv validation predictions with lightweight metrics."""

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
    return re.sub(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()【】\\[\\]{}<>《》-]+", "", text).lower()


def seq(text: str) -> list[str]:
    return list(normalize(text))


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> dict[str, float]:
    p = seq(pred)
    r = seq(ref)
    lcs = lcs_len(p, r)
    precision = lcs / len(p) if p else 0.0
    recall = lcs / len(r) if r else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def time_variants(start: float, end: float) -> set[str]:
    return {
        f"{start:.3f}-{end:.3f}",
        f"{start:.2f}-{end:.2f}",
        f"{start:.1f}-{end:.1f}",
        f"{start:g}-{end:g}",
        f"{start:.3f}秒-{end:.3f}秒",
        f"{start:.2f}秒-{end:.2f}秒",
        f"{start:g}秒-{end:g}秒",
        f"{start:.3f}s-{end:.3f}s",
        f"{start:.2f}s-{end:.2f}s",
        f"{start:g}s-{end:g}s",
    }


def contains_any(text: str, variants: set[str]) -> bool:
    norm = normalize(text)
    return any(normalize(item) in norm for item in variants)


def task_hit(pred: str, task: str) -> bool:
    if not task:
        return False
    norm_pred = normalize(pred)
    norm_task = normalize(task)
    if norm_task and norm_task in norm_pred:
        return True
    m = re.fullmatch(r"asl([0-9a-z])", norm_task)
    if m:
        label = m.group(1)
        return f"asl{label}" in norm_pred or f"手形{label}" in norm_pred
    return False


def refusal_like(pred: str) -> bool:
    markers = ["无法", "不能确定", "看不到", "没有看到", "作为ai", "我不能", "无法判断"]
    norm_pred = normalize(pred)
    return any(normalize(marker) in norm_pred for marker in markers)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    rouge_f1_sum = 0.0
    rouge_recall_sum = 0.0
    task_hits = 0
    interval_hits = 0
    refusals = 0
    non_empty = 0

    for row in rows:
        pred = str(row.get("prediction", ""))
        ref = str(row.get("reference_answer", ""))
        meta = row.get("metadata") or {}
        task = str(meta.get("task") or "")
        interval = meta.get("target_interval")
        scores = rouge_l(pred, ref)
        hit_task = task_hit(pred, task)
        hit_interval = False
        if isinstance(interval, list) and len(interval) == 2:
            hit_interval = contains_any(pred, time_variants(float(interval[0]), float(interval[1])))
        is_refusal = refusal_like(pred)
        is_non_empty = bool(normalize(pred))

        rouge_f1_sum += scores["f1"]
        rouge_recall_sum += scores["recall"]
        task_hits += int(hit_task)
        interval_hits += int(hit_interval)
        refusals += int(is_refusal)
        non_empty += int(is_non_empty)
        details.append(
            {
                "id": row.get("id"),
                "source_line_no": meta.get("source_line_no"),
                "turn_index": meta.get("turn_index"),
                "task": task,
                "rouge_l_f1": scores["f1"],
                "rouge_l_recall": scores["recall"],
                "task_hit": hit_task,
                "interval_hit": hit_interval,
                "refusal_like": is_refusal,
                "prediction": pred,
                "reference_answer": ref,
            }
        )

    count = len(rows)
    return {
        "eval_type": "wearable_egoconv_reviewed_val",
        "count": count,
        "non_empty": non_empty,
        "non_empty_rate": non_empty / count if count else 0.0,
        "task_hits": task_hits,
        "task_hit_rate": task_hits / count if count else 0.0,
        "interval_hits": interval_hits,
        "interval_hit_rate": interval_hits / count if count else 0.0,
        "refusal_like": refusals,
        "refusal_like_rate": refusals / count if count else 0.0,
        "avg_rouge_l_f1": rouge_f1_sum / count if count else 0.0,
        "avg_rouge_l_recall": rouge_recall_sum / count if count else 0.0,
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
