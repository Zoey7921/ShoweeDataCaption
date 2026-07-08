#!/usr/bin/env python3
"""Summarize EgoConv run progress and ETA from prediction/log files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_samples(log_path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if not log_path.exists():
        return samples
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "elapsed_seconds" in row and "row" in row:
                samples.append(row)
    return samples


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--total", type=int, default=700)
    args = parser.parse_args()

    pred_rows = count_lines(args.predictions)
    samples = parse_samples(args.log)
    times = [float(s["elapsed_seconds"]) for s in samples]
    recent = times[-20:] if times else []
    remaining = max(args.total - pred_rows, 0)

    status: dict[str, Any] = {
        "prediction_rows": pred_rows,
        "total_rows": args.total,
        "remaining_rows": remaining,
        "logged_samples": len(samples),
    }
    if times:
        avg = mean(times)
        status["avg_sec_per_sample"] = round(avg, 2)
        status["eta_hours_by_all_avg"] = round(remaining * avg / 3600, 2)
        status["last_sample"] = {
            "row": samples[-1].get("row"),
            "video_path": samples[-1].get("video_path"),
            "turns": samples[-1].get("turns"),
            "elapsed_seconds": samples[-1].get("elapsed_seconds"),
        }
    if recent:
        recent_avg = mean(recent)
        status["recent20_sec_per_sample"] = round(recent_avg, 2)
        status["eta_hours_by_recent20"] = round(remaining * recent_avg / 3600, 2)

    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
