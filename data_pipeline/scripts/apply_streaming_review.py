#!/usr/bin/env python3
"""Apply streaming review_update blocks and refresh EgoConv export."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

import export_wearable_egoconv as egoconv
from build_streaming_interaction_dataset import to_messages_sample


PROJECT_ROOT = Path(
    os.environ.get("SHOWEE_PIPELINE_ROOT", Path(__file__).resolve().parents[1])
)
CODE_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def resolve_config_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    code_path = CODE_ROOT / p
    if code_path.exists():
        return code_path
    return resolve_path(p)


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
        raise ValueError(f"expected numeric value, got {value!r}") from exc


def frame_index(time_sec: float, fps: float) -> int:
    return int(round(time_sec * fps))


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


def validate_review_status(status: str) -> str:
    allowed = {
        "streaming_unreviewed",
        "streaming_accepted",
        "streaming_edited",
        "streaming_rejected",
    }
    if status not in allowed:
        raise ValueError(f"invalid streaming review status {status!r}; expected one of {sorted(allowed)}")
    return status


def segment_ids(sample: dict[str, Any]) -> set[str]:
    segments = sample.get("metadata", {}).get("temporal_segments", [])
    if not isinstance(segments, list):
        return set()
    out: set[str] = set()
    for idx, segment in enumerate(segments):
        if isinstance(segment, dict):
            out.add(str(segment.get("segment_id", f"seg_{idx:03d}")))
    return out


def normalize_turn_update(update: Any, sample_id: str) -> dict[str, Any]:
    if not isinstance(update, dict):
        raise TypeError(f"{sample_id}: each turn update must be a mapping")
    if "turn_id" not in update:
        raise KeyError(f"{sample_id}: each turn update must include turn_id")
    return update


def update_turns(
    sample: dict[str, Any],
    turn_updates: Any,
    fps: float,
) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(turn_updates, list):
        raise TypeError(f"{sample.get('id', '')}: review_update.turns must be a list")
    turns = sample.get("turns", [])
    if not isinstance(turns, list):
        raise TypeError(f"{sample.get('id', '')}: sample.turns must be a list")

    by_id = {str(turn.get("turn_id", "")): turn for turn in turns if isinstance(turn, dict)}
    known_segments = segment_ids(sample)
    changed = 0

    for raw_update in turn_updates:
        update = normalize_turn_update(raw_update, str(sample.get("id", "")))
        turn_id = str(update["turn_id"])
        if turn_id not in by_id:
            raise KeyError(f"{sample.get('id', '')}: unknown turn_id {turn_id!r}")
        turn = by_id[turn_id]
        for key in ("user", "assistant", "segment_id"):
            if key in update:
                value = str(update[key])
                if key == "segment_id" and known_segments and value not in known_segments:
                    raise KeyError(f"{sample.get('id', '')}: unknown segment_id {value!r}")
                if turn.get(key) != value:
                    turn[key] = value
                    changed += 1
        if "time_sec" in update:
            time_sec = round(as_float(update["time_sec"]), 3)
            if time_sec < 0:
                raise ValueError(f"{sample.get('id', '')}: time_sec must be >= 0")
            if turn.get("time_sec") != time_sec:
                turn["time_sec"] = time_sec
                changed += 1
            if "frame_index" not in update:
                turn["frame_index"] = frame_index(time_sec, fps)
        if "frame_index" in update:
            idx = int(as_float(update["frame_index"]))
            if idx < 0:
                raise ValueError(f"{sample.get('id', '')}: frame_index must be >= 0")
            if turn.get("frame_index") != idx:
                turn["frame_index"] = idx
                changed += 1

    rebuild_history(turns)
    return turns, changed


def rebuild_history(turns: list[dict[str, Any]]) -> None:
    history: list[dict[str, str]] = []
    for turn in turns:
        turn["history"] = list(history)
        history.append(
            {
                "user": str(turn.get("user", "")),
                "assistant": str(turn.get("assistant", "")),
            }
        )


def apply_update(
    sample: dict[str, Any],
    update: dict[str, Any],
    fps: float,
) -> tuple[dict[str, Any], int]:
    out = json.loads(json.dumps(sample, ensure_ascii=False))
    meta = out.setdefault("metadata", {})
    changed = 0

    status = update.get("review_status", update.get("status"))
    if status is not None:
        status = validate_review_status(str(status))
        if meta.get("review_status") != status:
            meta["review_status"] = status
            changed += 1

    if "review_notes" in update and meta.get("review_notes") != str(update["review_notes"]):
        meta["review_notes"] = str(update["review_notes"])
        changed += 1

    metadata_update = update.get("metadata")
    if metadata_update is not None:
        if not isinstance(metadata_update, dict):
            raise TypeError(f"{out.get('id', '')}: review_update.metadata must be a mapping")
        for key, value in metadata_update.items():
            if meta.get(key) != value:
                meta[key] = value
                changed += 1

    if "turns" in update:
        _, turn_changes = update_turns(out, update["turns"], fps)
        changed += turn_changes

    return out, changed


def export_egoconv_from_config(config_path: Path) -> dict[str, Any]:
    config = egoconv.read_yaml(config_path)
    input_cfg = config.get("input", {})
    if not isinstance(input_cfg, dict):
        raise TypeError("egoconv config.input must be a mapping")
    source_dataset = egoconv.resolve_path(input_cfg["dataset"])
    source_format = str(input_cfg.get("format", "streaming_turns"))
    if source_format != "streaming_turns":
        raise ValueError("only egoconv input.format='streaming_turns' is supported")

    video_cfg = config.get("video", {})
    if not isinstance(video_cfg, dict):
        raise TypeError("egoconv config.video must be a mapping")
    video_root = egoconv.resolve_path(video_cfg.get("root", "."))
    video_path_mode = str(video_cfg.get("path_mode", "relative_to_root"))

    output_cfg = config.get("output", {})
    if not isinstance(output_cfg, dict):
        raise TypeError("egoconv config.output must be a mapping")
    output_jsonl = egoconv.resolve_path(
        output_cfg.get("jsonl", "data/processed/wearable_egoconv_v001.jsonl")
    )
    report_path = egoconv.resolve_path(
        output_cfg.get("report", "runs/wearable_egoconv_v001/export_report.json")
    )
    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}

    source = egoconv.read_json(source_dataset)
    if not isinstance(source, list):
        raise TypeError(f"source dataset must be a JSON list: {source_dataset}")

    rows = [
        row
        for row in (
            egoconv.make_egoconv_row(sample, video_root, video_path_mode, defaults)
            for sample in source
        )
        if row is not None
    ]
    errors = egoconv.validate_rows(rows)
    if errors:
        raise ValueError("EgoConv export validation failed:\n" + "\n".join(errors[:20]))

    egoconv.write_jsonl(output_jsonl, rows)
    turn_counts = [len(row["questions"]) for row in rows]
    report = {
        "version": str(config.get("version", "wearable_egoconv_v001")),
        "config": str(config_path),
        "source_dataset": str(source_dataset),
        "source_format": source_format,
        "video_root": str(video_root),
        "video_path_mode": video_path_mode,
        "source_count": len(source),
        "row_count": len(rows),
        "turn_count": sum(turn_counts),
        "min_turns_per_row": min(turn_counts) if turn_counts else 0,
        "max_turns_per_row": max(turn_counts) if turn_counts else 0,
        "output_jsonl": str(output_jsonl),
        "validation_errors": 0,
    }
    egoconv.write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/streaming_interaction_v001.json"))
    parser.add_argument("--messages", type=Path, default=Path("data/processed/streaming_interaction_v001_messages.json"))
    parser.add_argument("--review", type=Path, default=Path("eval/human_review/streaming_interaction_v001_review.md"))
    parser.add_argument("--egoconv-config", type=Path, default=Path("configs/wearable_egoconv_v001.yaml"))
    parser.add_argument("--report", type=Path, default=Path("runs/streaming_interaction_v001/review_apply_report.json"))
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-egoconv-export", action="store_true")
    args = parser.parse_args()

    dataset_path = resolve_path(args.dataset)
    messages_path = resolve_path(args.messages)
    review_path = resolve_path(args.review)
    egoconv_config_path = resolve_config_path(args.egoconv_config)
    report_path = resolve_path(args.report)

    samples = read_json(dataset_path)
    if not isinstance(samples, list):
        raise TypeError(f"dataset must be a JSON list: {dataset_path}")
    updates = parse_review_updates(review_path)
    sample_ids = {str(sample.get("id", "")) for sample in samples}
    unknown_updates = sorted(set(updates) - sample_ids)
    if unknown_updates:
        raise KeyError(f"review updates refer to unknown sample ids: {unknown_updates[:10]}")

    updated_samples: list[dict[str, Any]] = []
    applied = 0
    changed_fields = 0
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        if sample_id in updates:
            updated_sample, changes = apply_update(sample, updates[sample_id], args.fps)
            updated_samples.append(updated_sample)
            applied += 1
            changed_fields += changes
        else:
            updated_samples.append(sample)

    report: dict[str, Any] = {
        "dataset": str(dataset_path),
        "messages": str(messages_path),
        "review": str(review_path),
        "sample_count": len(samples),
        "review_update_blocks": len(updates),
        "applied_updates": applied,
        "changed_fields": changed_fields,
        "dry_run": args.dry_run,
        "egoconv_export": None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    message_samples = [to_messages_sample(sample) for sample in updated_samples]
    write_json(dataset_path, updated_samples)
    write_json(messages_path, message_samples)

    if not args.skip_egoconv_export:
        report["egoconv_export"] = export_egoconv_from_config(egoconv_config_path)

    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
