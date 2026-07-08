#!/usr/bin/env python3
"""Summarize Showee open-description or candidate-choice prediction files."""

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


def candidate_ids(row: dict[str, Any]) -> set[str]:
    return set(candidate_map(row))


def candidate_map(row: dict[str, Any]) -> dict[str, str]:
    candidates = row.get("candidate_tasks")
    out: dict[str, str] = {}
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, dict) and item.get("task_id"):
                out[str(item["task_id"]).strip()] = str(item.get("task_name", "")).strip()
    question = str(row.get("question", ""))
    for match in re.finditer(r"\d+\.\s*([^\n(]+)\(([A-Za-z0-9_]+)\)", question):
        out[match.group(2).strip()] = match.group(1).strip()
    return out


def parse_prediction_task_id(
    text: str,
    allowed_ids: set[str] | None = None,
    aliases: dict[str, str] | None = None,
) -> str | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    allowed_ids = allowed_ids or set()
    aliases = aliases or {}

    def match_alias(source: str) -> str | None:
        norm_source = normalize(source)
        for task_id, task_name in aliases.items():
            if task_name and normalize(task_name) in norm_source:
                return task_id
            if task_id.startswith("asl_"):
                suffix = task_id.removeprefix("asl_").upper()
                if normalize(f"ASL {suffix}") in norm_source:
                    return task_id
        return None

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            raw_id = str(parsed.get("task_id", "")).strip()
            if raw_id and not raw_id.isdigit() and (not allowed_ids or raw_id in allowed_ids):
                return raw_id
            text_fields = " ".join(
                str(parsed.get(key, "")) for key in ["task_name", "reason"]
            )
            for task_id in sorted(allowed_ids, key=len, reverse=True):
                if task_id and task_id in text_fields:
                    return task_id
            alias_id = match_alias(text_fields)
            if alias_id:
                return alias_id
            if not allowed_ids:
                match = re.search(r"\b([a-z]+(?:_[a-z0-9]+)+)\b", text_fields)
                if match:
                    return match.group(1)
    except json.JSONDecodeError:
        pass
    for task_id in sorted(allowed_ids, key=len, reverse=True):
        if task_id and task_id in text:
            return task_id
    alias_id = match_alias(text)
    if alias_id:
        return alias_id
    match = re.search(r'"?task_id"?\s*[:：]\s*"?([A-Za-z0-9_]+)"?', text)
    if match and (not allowed_ids or match.group(1) in allowed_ids):
        return match.group(1)
    match = re.search(r"\b([a-z]+(?:_[a-z0-9]+)+)\b", text)
    if match:
        task_id = match.group(1)
        if not allowed_ids or task_id in allowed_ids:
            return task_id
    return None


def expected_task(row: dict[str, Any]) -> tuple[str, str]:
    reference = row.get("reference_answer")
    if isinstance(reference, dict):
        return str(reference.get("task_id", "")), str(reference.get("task_name", ""))
    meta = row.get("metadata", {})
    return str(meta.get("task_id", "")), str(meta.get("task_name", ""))


def summarize_open(rows: list[dict[str, Any]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    hits = 0
    for row in rows:
        task_id, task_name = expected_task(row)
        pred = str(row.get("prediction", ""))
        norm_pred = normalize(pred)
        task_name_hit = bool(task_name and normalize(task_name) in norm_pred)
        task_id_hit = bool(task_id and normalize(task_id) in norm_pred)
        hit = task_name_hit or task_id_hit
        hits += int(hit)
        details.append(
            {
                "id": row.get("id"),
                "task_id": task_id,
                "task_name": task_name,
                "task_name_or_id_hit": hit,
                "task_name_hit": task_name_hit,
                "task_id_hit": task_id_hit,
            }
        )
    return {
        "eval_type": "open_description",
        "count": len(rows),
        "task_name_or_id_hits": hits,
        "task_name_or_id_hit_rate": hits / len(rows) if rows else 0.0,
        "details": details,
    }


def summarize_choice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    hits = 0
    parse_failures = 0
    for row in rows:
        expected_id, expected_name = expected_task(row)
        pred = str(row.get("prediction", ""))
        candidates = candidate_map(row)
        pred_id = parse_prediction_task_id(pred, set(candidates), candidates)
        hit = pred_id == expected_id
        hits += int(hit)
        parse_failures += int(pred_id is None)
        details.append(
            {
                "id": row.get("id"),
                "expected_task_id": expected_id,
                "expected_task_name": expected_name,
                "predicted_task_id": pred_id,
                "choice_hit": hit,
            }
        )
    return {
        "eval_type": "candidate_task_choice",
        "count": len(rows),
        "choice_hits": hits,
        "choice_hit_rate": hits / len(rows) if rows else 0.0,
        "parse_failures": parse_failures,
        "details": details,
    }


def infer_eval_type(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "open"
    if isinstance(rows[0].get("reference_answer"), dict):
        return "choice"
    question = str(rows[0].get("question", ""))
    if "候选任务" in question or "task_id" in question:
        return "choice"
    return "open"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_jsonl", type=Path)
    parser.add_argument("--eval-type", choices=["auto", "open", "choice"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.prediction_jsonl)
    eval_type = infer_eval_type(rows) if args.eval_type == "auto" else args.eval_type
    summary = summarize_choice(rows) if eval_type == "choice" else summarize_open(rows)
    summary["prediction_jsonl"] = str(args.prediction_jsonl)

    output = args.output or args.prediction_jsonl.with_suffix(args.prediction_jsonl.suffix + ".metrics.json")
    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
