#!/usr/bin/env python3
"""Build a reusable temporal caption dataset from ShoweeHandv2.

This pipeline is independent of prior experiment batches. It can optionally
inherit existing annotations, then adds temporal_segments and timestamped
answers while preserving the ShareGPT-style multi-turn conversation format.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(
    os.environ.get("SHOWEE_PIPELINE_ROOT", Path(__file__).resolve().parents[1])
)
RAW_ROOT = Path(
    os.environ.get("SHOWEE_RAW_ROOT", PROJECT_ROOT / "data/raw/ShoweeHandv2/raw")
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"config must be a mapping: {path}")
    return data


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metadata_value(task: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = task.get(key)
        if value:
            return str(value)
    return default


def head_video(sample_dir: Path, source_view: str) -> Path | None:
    view_dir = sample_dir / source_view
    candidates = sorted(view_dir.glob("*.mkv"))
    return candidates[0] if candidates else None


def build_sample_id(session: dict[str, Any], task_id: str, source_view: str) -> str:
    return safe_id(
        "showee_"
        f"{session.get('subject_id', 'unknown')}_"
        f"{session.get('session_type', 'unknown')}_"
        f"{session.get('session_repeat', 'unknown')}_"
        f"{task_id}_"
        f"{source_view.removeprefix('showee_')}"
    )


def scan_raw_samples(raw_root: Path, source_view: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(raw_root.glob("*/*/metadata.json")):
        sample_dir = metadata_path.parent
        video = head_video(sample_dir, source_view)
        if video is None:
            continue
        try:
            metadata = read_json(metadata_path)
        except Exception as exc:
            rows.append(
                {
                    "id": safe_id(str(sample_dir.relative_to(raw_root))),
                    "readable": False,
                    "error": repr(exc),
                    "metadata_path": str(metadata_path),
                }
            )
            continue
        session = metadata.get("session", {})
        task = metadata.get("task", {})
        task_id = str(task.get("id") or sample_dir.name)
        task_name = metadata_value(task, "name", "name_en", default=task_id)
        task_desc = metadata_value(task, "description", "desc_en", default=task_name)
        rows.append(
            {
                "id": build_sample_id(session, task_id, source_view),
                "video": str(video),
                "metadata_path": str(metadata_path),
                "subject_id": str(session.get("subject_id", "")),
                "session_type": str(session.get("session_type", "")),
                "session_repeat": str(session.get("session_repeat", "")),
                "task_id": task_id,
                "task_name": task_name,
                "task_desc": task_desc,
                "duration": task.get("duration"),
                "source_view": source_view,
                "readable": video.exists(),
                "error": "",
            }
        )
    return [row for row in rows if row.get("readable")]


def normalize_video_key(path: str) -> str:
    return re.sub(r"^.*?/ShoweeHandv2/raw/", "ShoweeHandv2/raw/", path)


def load_existing_annotations(paths: list[str]) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    for rel in paths:
        path = resolve_path(rel)
        if not path.exists():
            continue
        data = read_json(path)
        if not isinstance(data, list):
            continue
        for sample in data:
            if not isinstance(sample, dict):
                continue
            keys = [
                str(sample.get("id", "")),
                normalize_video_key(str(sample.get("video", ""))),
            ]
            meta = sample.get("metadata", {})
            if isinstance(meta, dict):
                key = "|".join(
                    [
                        str(meta.get("subject_id", "")),
                        str(meta.get("session_type", "")),
                        str(meta.get("session_repeat", "")),
                        str(meta.get("task_id", "")),
                    ]
                )
                keys.append(key)
            for key in keys:
                if key and key not in existing:
                    existing[key] = sample
    return existing


def match_existing(row: dict[str, Any], existing: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    keys = [
        str(row["id"]),
        normalize_video_key(str(row["video"])),
        "|".join(
            [
                str(row.get("subject_id", "")),
                str(row.get("session_type", "")),
                str(row.get("session_repeat", "")),
                str(row.get("task_id", "")),
            ]
        ),
    ]
    for key in keys:
        if key in existing:
            return existing[key]
    return None


def rank_task(task_id: str, priority_tasks: list[str]) -> tuple[int, str]:
    return (0 if task_id in priority_tasks else 1, task_id)


def select_rows(
    rows: list[dict[str, Any]],
    existing: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    sample_count = int(config.get("sample_count", 100))
    sampling = config.get("sampling", {})
    priority_tasks = [str(x) for x in sampling.get("priority_tasks", [])]
    inherit_first = bool(sampling.get("inherit_existing_first", True))
    balance_by_task = bool(sampling.get("balance_by_task", True))

    annotated: list[dict[str, Any]] = []
    fresh: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        row["_existing"] = match_existing(row, existing)
        (annotated if row["_existing"] and inherit_first else fresh).append(row)
    pools = [annotated, fresh] if inherit_first else [annotated + fresh]

    selected: list[dict[str, Any]] = []
    seen_videos: set[str] = set()

    def add_row(row: dict[str, Any]) -> bool:
        video_key = normalize_video_key(str(row["video"]))
        if video_key in seen_videos:
            return False
        selected.append(row)
        seen_videos.add(video_key)
        return len(selected) >= sample_count

    for pool in pools:
        if not pool:
            continue
        pool = sorted(
            pool,
            key=lambda row: (
                rank_task(str(row["task_id"]), priority_tasks),
                str(row.get("subject_id", "")),
                str(row.get("session_repeat", "")),
                str(row["id"]),
            ),
        )
        if balance_by_task:
            by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in pool:
                by_task[str(row["task_id"])].append(row)
            task_order = sorted(by_task, key=lambda task: rank_task(task, priority_tasks))
            while task_order and len(selected) < sample_count:
                next_order: list[str] = []
                for task_id in task_order:
                    if by_task[task_id]:
                        done = add_row(by_task[task_id].pop(0))
                        if done:
                            break
                    if by_task[task_id]:
                        next_order.append(task_id)
                task_order = next_order
        else:
            for row in pool:
                if add_row(row):
                    break
        if len(selected) >= sample_count:
            break
    return selected


def motion_hint(task_desc: str) -> str:
    hints: list[str] = []
    if any(x in task_desc for x in ["转动手腕", "手腕"]):
        hints.append("手腕转动或朝向变化")
    if any(x in task_desc for x in ["弯曲", "屈", "收回"]):
        hints.append("手指弯曲程度变化")
    if any(x in task_desc for x in ["伸展", "伸出", "张开", "开掌"]):
        hints.append("手指伸展或张开")
    if any(x in task_desc for x in ["并拢", "分合", "分和"]):
        hints.append("手指分合")
    if any(x in task_desc for x in ["捏合", "触", "touch", "滑动"]):
        hints.append("指尖接触或滑动")
    if any(x in task_desc for x in ["顺序", "波浪", "依次"]):
        hints.append("按序连续变化")
    return "、".join(hints) if hints else "手部姿态保持与轻微变化"


def duration_sec(row: dict[str, Any], config: dict[str, Any]) -> float:
    default = as_float(config.get("temporal", {}).get("default_duration_sec"), 20.0)
    value = as_float(row.get("duration"), 0.0)
    return round(value if value > 0 else default, 3)


def should_use_phases(task_desc: str, task_id: str) -> bool:
    dynamic_tokens = ["顺序", "波浪", "依次", "张开", "并拢", "开掌", "握拳", "捏合", "滑动", "不断变换"]
    dynamic_tasks = {"seq_flex", "seq_extend", "seq_pinch", "finger_wave", "fist_open", "abduct_adduct", "thumb_swipe", "thumb_touchpad", "free_style", "wrist"}
    return task_id in dynamic_tasks or any(token in task_desc for token in dynamic_tokens)


def temporal_segments(row: dict[str, Any], config: dict[str, Any], existing: dict[str, Any] | None) -> list[dict[str, Any]]:
    meta = existing.get("metadata", {}) if existing else {}
    inherited = meta.get("temporal_segments") if isinstance(meta, dict) else None
    if isinstance(inherited, list) and inherited:
        inherit_verified_only = bool(config.get("temporal", {}).get("inherit_verified_segments_only", True))
        has_verified_boundary = any(
            str(seg.get("confidence", "")) == "verified"
            or str(seg.get("boundary_source", "")) in {"human_review", "frame_verified"}
            for seg in inherited
            if isinstance(seg, dict)
        )
        if has_verified_boundary or not inherit_verified_only:
            return inherited

    task_id = str(row["task_id"])
    task_name = str(row["task_name"])
    task_desc = str(row.get("task_desc") or task_name)
    end = duration_sec(row, config)
    hint = motion_hint(task_desc)
    mode = str(config.get("temporal", {}).get("mode", "phase_seed"))
    max_segments = int(config.get("temporal", {}).get("max_segments", 3))

    if mode == "phase_seed" and max_segments >= 3 and should_use_phases(task_desc, task_id):
        p1 = round(end * 0.25, 3)
        p2 = round(end * 0.75, 3)
        return [
            {
                "segment_id": "seg_000",
                "start_sec": 0.0,
                "end_sec": p1,
                "label": f"{task_name}-起始姿态",
                "description": f"进入{task_name}任务的起始姿态，开始出现{hint}。",
                "boundary_source": "uniform_phase_seed",
                "confidence": "medium",
                "notes": "阶段边界为规则生成，需要人工复核。",
            },
            {
                "segment_id": "seg_001",
                "start_sec": p1,
                "end_sec": p2,
                "label": f"{task_name}-主要动作",
                "description": f"主要执行{task_desc}，重点观察{hint}。",
                "boundary_source": "uniform_phase_seed",
                "confidence": "medium",
                "notes": "阶段边界为规则生成，需要人工复核。",
            },
            {
                "segment_id": "seg_002",
                "start_sec": p2,
                "end_sec": end,
                "label": f"{task_name}-结束或重复",
                "description": f"动作继续或回到稳定姿态，仍围绕{task_name}任务。",
                "boundary_source": "uniform_phase_seed",
                "confidence": "medium",
                "notes": "阶段边界为规则生成，需要人工复核。",
            },
        ]

    return [
        {
            "segment_id": "seg_000",
            "start_sec": 0.0,
            "end_sec": end,
            "label": task_name,
            "description": f"整段视频围绕{task_name}任务展开，主要观察点是{hint}。",
            "boundary_source": "metadata_duration",
            "confidence": "coarse",
            "notes": "粗粒度任务级时间段；尚未逐帧人工标注动作边界。",
        }
    ]


def segment_answer(segments: list[dict[str, Any]]) -> str:
    parts = []
    for seg in segments:
        parts.append(
            f"{as_float(seg.get('start_sec')):.2f}-{as_float(seg.get('end_sec')):.2f}s："
            f"{seg.get('description', seg.get('label', '手势动作'))}"
        )
    parts.append("以上时间段为初始标注，精确动作边界需要结合视频逐帧复核。")
    return " ".join(parts)


def seed_conversations(row: dict[str, Any], segments: list[dict[str, Any]]) -> list[dict[str, str]]:
    task_name = str(row["task_name"])
    task_desc = str(row.get("task_desc") or task_name)
    hint = motion_hint(task_desc)
    return [
        {
            "from": "user",
            "value": "<video>\n请描述视频中正在执行的手势任务。",
        },
        {
            "from": "assistant",
            "value": (
                f"视频展示的是 {task_name} 手势任务。根据元数据，任务要求为{task_desc}。"
                "该样本为 temporal caption 初始标注，需要后续人工复核视觉细节。"
            ),
        },
        {
            "from": "user",
            "value": "这个手势的关键手部动作是什么？",
        },
        {
            "from": "assistant",
            "value": f"关键手部动作包括：{task_desc}。可重点关注{hint}。",
        },
        {
            "from": "user",
            "value": "动作从头到尾有什么变化？请明确时间段。",
        },
        {
            "from": "assistant",
            "value": segment_answer(segments),
        },
    ]


def update_temporal_turn(conversations: list[dict[str, Any]], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = json.loads(json.dumps(conversations, ensure_ascii=False))
    for idx, turn in enumerate(out):
        if (
            isinstance(turn, dict)
            and turn.get("from") == "user"
            and ("动作从头到尾" in str(turn.get("value", "")) or "时间段" in str(turn.get("value", "")))
            and idx + 1 < len(out)
        ):
            turn["value"] = "动作从头到尾有什么变化？请明确时间段。"
            out[idx + 1]["value"] = segment_answer(segments)
            return out
    out.extend(
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
    return out


def make_sample(row: dict[str, Any], split: str, config: dict[str, Any]) -> dict[str, Any]:
    existing = row.get("_existing")
    segments = temporal_segments(row, config, existing)
    review_status = str(config.get("temporal", {}).get("review_status", "temporal_unreviewed"))
    meta = {
        "dataset": "ShoweeHandv2",
        "source_view": row["source_view"],
        "metadata_path": row["metadata_path"],
        "subject_id": row["subject_id"],
        "session_type": row["session_type"],
        "session_repeat": row["session_repeat"],
        "task_id": row["task_id"],
        "task_name": row["task_name"],
        "task_desc": row.get("task_desc", ""),
        "duration": row.get("duration"),
        "split": split,
        "annotation_source": "temporal_seeded_from_metadata",
        "review_status": review_status,
        "temporal_segments": segments,
    }
    if existing:
        old_meta = existing.get("metadata", {})
        if isinstance(old_meta, dict):
            meta["annotation_source"] = f"inherited:{old_meta.get('annotation_source', 'existing_annotation')}"
            meta["inherited_review_status"] = old_meta.get("review_status", "")
        conversations = update_temporal_turn(existing.get("conversations", []), segments)
    else:
        conversations = seed_conversations(row, segments)
    return {
        "id": row["id"],
        "video": row["video"],
        "metadata": meta,
        "conversations": conversations,
    }


def split_names(config: dict[str, Any]) -> list[str]:
    split = config.get("split", {})
    names: list[str] = []
    for key in ["train", "val", "test"]:
        names.extend([key] * int(split.get(key, 0)))
    if len(names) != int(config.get("sample_count", len(names))):
        raise ValueError("split counts must sum to sample_count")
    return names


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
            meta = sample["metadata"]
            writer.writerow(
                {
                    "sample_id": sample["id"],
                    "split": meta["split"],
                    "task_id": meta["task_id"],
                    "task_name": meta["task_name"],
                    "subject_id": meta["subject_id"],
                    "session_repeat": meta["session_repeat"],
                    "annotation_source": meta["annotation_source"],
                    "review_status": meta["review_status"],
                    "segment_count": len(meta.get("temporal_segments", [])),
                    "video": sample["video"],
                }
            )


def write_review(path: Path, samples: list[dict[str, Any]]) -> None:
    lines = [
        "# Temporal Caption v001 人工复核表",
        "",
        "状态可填：temporal_accepted / temporal_edited / temporal_rejected / frame_verified。",
        "重点检查：任务名是否正确、时间段边界是否合理、每段动作描述是否与视频一致。",
        "",
    ]
    for idx, sample in enumerate(samples, start=1):
        meta = sample["metadata"]
        lines.extend(
            [
                f"## {idx}. {sample['id']}",
                "",
                f"- status: {meta['review_status']}",
                f"- split: {meta['split']}",
                f"- video: {sample['video']}",
                f"- task: {meta['task_id']} / {meta['task_name']}",
                f"- annotation_source: {meta['annotation_source']}",
                "- temporal_segments:",
            ]
        )
        for seg in meta.get("temporal_segments", []):
            lines.append(
                f"  - {seg.get('segment_id')}: {seg.get('start_sec')}-{seg.get('end_sec')}s | "
                f"{seg.get('label')} | {seg.get('description')} | "
                f"{seg.get('boundary_source')} / {seg.get('confidence')}"
            )
        lines.extend(["", "review_notes:", "", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/temporal_caption_v001.yaml")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = read_yaml(config_path)
    version = str(config.get("version", "temporal_caption_v001"))
    source_view = str(config.get("source_view", "showee_head"))
    output = config.get("outputs", {})
    processed_dir = resolve_path(output.get("processed_dir", "data/processed"))
    splits_dir = resolve_path(output.get("splits_dir", "data/splits"))
    review_dir = resolve_path(output.get("review_dir", "eval/human_review"))
    run_dir = resolve_path(output.get("run_dir", f"runs/{version}"))

    existing = load_existing_annotations([str(x) for x in config.get("existing_annotations", [])])
    rows = scan_raw_samples(RAW_ROOT, source_view)
    selected = select_rows(rows, existing, config)
    names = split_names(config)
    if len(selected) < len(names):
        raise RuntimeError(f"selected only {len(selected)} samples, need {len(names)}")

    samples = [make_sample(row, split, config) for row, split in zip(selected, names)]
    by_split = {split: [s for s in samples if s["metadata"]["split"] == split] for split in ["train", "val", "test"]}

    write_json(processed_dir / f"{version}.json", samples)
    for split, split_samples in by_split.items():
        write_json(processed_dir / f"{version}_{split}.json", split_samples)
        write_json(splits_dir / f"{version}_{split}.json", split_samples)
    write_index(processed_dir / f"{version}_index.csv", samples)
    write_review(review_dir / f"{version}_review.md", samples)

    inherited_count = sum(1 for sample in samples if str(sample["metadata"]["annotation_source"]).startswith("inherited:"))
    report = {
        "version": version,
        "config": str(config_path),
        "raw_root": str(RAW_ROOT),
        "source_view": source_view,
        "indexed_rows": len(rows),
        "selected_rows": len(samples),
        "inherited_annotations": inherited_count,
        "metadata_seeded": len(samples) - inherited_count,
        "split_counts": {key: len(value) for key, value in by_split.items()},
        "outputs": {
            "all": str(processed_dir / f"{version}.json"),
            "index": str(processed_dir / f"{version}_index.csv"),
            "review": str(review_dir / f"{version}_review.md"),
        },
    }
    write_json(run_dir / "build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
