#!/usr/bin/env python3
"""Apply temporal review_update blocks back to temporal caption JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(
    os.environ.get("SHOWEE_PIPELINE_ROOT", Path(__file__).resolve().parents[1])
)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected numeric timestamp, got {value!r}") from exc


def review_sections(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^##\s+\d+\.\s+(\S+)\s*$", text, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        sample_id = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections[sample_id] = text[start:end]
    return sections


def extract_review_update(section: str) -> dict[str, Any] | None:
    lines = section.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() != "review_update:":
            continue
        block_lines = ["review_update:"]
        for next_line in lines[idx + 1 :]:
            if next_line and not next_line.startswith((" ", "\t")):
                break
            block_lines.append(next_line)
        parsed = yaml.safe_load("\n".join(block_lines))
        if not parsed:
            return None
        update = parsed.get("review_update")
        if update is None:
            return None
        if not isinstance(update, dict):
            raise TypeError("review_update must be a mapping")
        return update
    return None


def parse_review_updates(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    updates: dict[str, dict[str, Any]] = {}
    for sample_id, section in review_sections(text).items():
        update = extract_review_update(section)
        if update:
            updates[sample_id] = update
    return updates


def validate_segments(segments: Any, sample_id: str) -> list[dict[str, Any]]:
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{sample_id}: temporal_segments must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    previous_end = 0.0
    for idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise TypeError(f"{sample_id}: segment {idx} must be a mapping")
        start = as_float(segment.get("start_sec"))
        end = as_float(segment.get("end_sec"))
        if end <= start:
            raise ValueError(f"{sample_id}: segment {idx} has end_sec <= start_sec")
        if idx > 0 and start < previous_end:
            raise ValueError(f"{sample_id}: segment {idx} overlaps previous segment")
        previous_end = end
        out = dict(segment)
        out["segment_id"] = str(out.get("segment_id") or f"seg_{idx:03d}")
        out["start_sec"] = round(start, 3)
        out["end_sec"] = round(end, 3)
        out["label"] = str(out.get("label", ""))
        out["description"] = str(out.get("description", ""))
        out["boundary_source"] = str(out.get("boundary_source", "human_review"))
        out["confidence"] = str(out.get("confidence", "verified"))
        out["notes"] = str(out.get("notes", ""))
        normalized.append(out)
    return normalized


def segment_answer(segments: list[dict[str, Any]]) -> str:
    parts = []
    for segment in segments:
        parts.append(
            f"{float(segment['start_sec']):.2f}-{float(segment['end_sec']):.2f}s："
            f"{segment.get('description') or segment.get('label') or '手势动作'}"
        )
    parts.append("以上时间段已根据人工复核更新。")
    return " ".join(parts)


def update_temporal_turn(sample: dict[str, Any], segments: list[dict[str, Any]]) -> None:
    conversations = sample.setdefault("conversations", [])
    if not isinstance(conversations, list):
        sample["conversations"] = []
        conversations = sample["conversations"]
    for idx, turn in enumerate(conversations):
        if (
            isinstance(turn, dict)
            and turn.get("from") == "user"
            and ("动作从头到尾" in str(turn.get("value", "")) or "时间段" in str(turn.get("value", "")))
            and idx + 1 < len(conversations)
            and isinstance(conversations[idx + 1], dict)
        ):
            turn["value"] = "动作从头到尾有什么变化？请明确时间段。"
            conversations[idx + 1]["value"] = segment_answer(segments)
            return
    conversations.extend(
        [
            {
                "from": "user",
                "value": "动作从头到尾有什么变化？请明确时间段。",
            },
            {
                "from": "assistant",
                "value": segment_answer(segments),
            },
        ]
    )


def apply_update(sample: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(sample, ensure_ascii=False))
    meta = out.setdefault("metadata", {})
    if "status" in update:
        meta["review_status"] = str(update["status"])
    if "review_status" in update:
        meta["review_status"] = str(update["review_status"])
    if "review_notes" in update:
        meta["review_notes"] = str(update["review_notes"])
    if "temporal_segments" in update:
        segments = validate_segments(update["temporal_segments"], str(out.get("id", "")))
        meta["temporal_segments"] = segments
        update_temporal_turn(out, segments)
    return out


def split_samples(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for sample in samples:
        split = str(sample.get("metadata", {}).get("split", ""))
        if split not in out:
            out[split] = []
        out[split].append(sample)
    return out


def write_index(path: Path, samples: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "split",
        "task_id",
        "task_name",
        "subject_id",
        "session_repeat",
        "annotation_source",
        "review_status",
        "segment_count",
        "video",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            meta = sample.get("metadata", {})
            writer.writerow(
                {
                    "sample_id": sample.get("id", ""),
                    "split": meta.get("split", ""),
                    "task_id": meta.get("task_id", ""),
                    "task_name": meta.get("task_name", ""),
                    "subject_id": meta.get("subject_id", ""),
                    "session_repeat": meta.get("session_repeat", ""),
                    "annotation_source": meta.get("annotation_source", ""),
                    "review_status": meta.get("review_status", ""),
                    "segment_count": len(meta.get("temporal_segments", [])),
                    "video": sample.get("video", ""),
                }
            )


def default_split_path(dataset: Path, split: str) -> Path:
    stem = dataset.stem
    if stem.endswith(f"_{split}"):
        return dataset
    return dataset.with_name(f"{stem}_{split}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/temporal_caption_v001.json"))
    parser.add_argument("--review", type=Path, default=Path("eval/human_review/temporal_caption_v001_review.md"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/temporal_caption_v001_index.csv"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_path = resolve_path(args.dataset)
    review_path = resolve_path(args.review)
    index_path = resolve_path(args.index)
    processed_dir = resolve_path(args.processed_dir)
    splits_dir = resolve_path(args.splits_dir)

    samples = read_json(dataset_path)
    if not isinstance(samples, list):
        raise TypeError(f"dataset must be a JSON list: {dataset_path}")
    updates = parse_review_updates(review_path)
    sample_ids = {str(sample.get("id", "")) for sample in samples}
    unknown_updates = sorted(set(updates) - sample_ids)
    if unknown_updates:
        raise KeyError(f"review updates refer to unknown sample ids: {unknown_updates[:10]}")

    updated_samples = []
    applied = 0
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        if sample_id in updates:
            updated_samples.append(apply_update(sample, updates[sample_id]))
            applied += 1
        else:
            updated_samples.append(sample)

    split_map = split_samples(updated_samples)
    report = {
        "dataset": str(dataset_path),
        "review": str(review_path),
        "sample_count": len(samples),
        "review_update_blocks": len(updates),
        "applied_updates": applied,
        "split_counts": {key: len(value) for key, value in split_map.items()},
        "dry_run": args.dry_run,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    write_json(dataset_path, updated_samples)
    for split, split_samples_for_name in split_map.items():
        if not split_samples_for_name:
            continue
        write_json(processed_dir / default_split_path(dataset_path, split).name, split_samples_for_name)
        write_json(splits_dir / default_split_path(dataset_path, split).name, split_samples_for_name)
    write_index(index_path, updated_samples)


if __name__ == "__main__":
    main()
