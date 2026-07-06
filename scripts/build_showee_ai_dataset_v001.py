#!/usr/bin/env python3
"""Build a small ShoweeHandv2 AI-labeled dataset for LoRA smoke training.

This is a member-B data pipeline:
  1. index readable ShoweeHandv2 showee_head videos,
  2. select a small diverse subset,
  3. ask Qwen3-VL for visual observations,
  4. write ShareGPT-style train data plus train/val/test splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(
    os.environ.get("SHOWEE_PIPELINE_ROOT", Path(__file__).resolve().parents[1])
)
RAW_ROOT = Path(
    os.environ.get("SHOWEE_RAW_ROOT", PROJECT_ROOT / "data/raw/ShoweeHandv2/raw")
)
DEFAULT_MODEL = Path(
    os.environ.get("QWEN3VL_MODEL", PROJECT_ROOT / "models/base/Qwen3-VL-8B-Instruct")
)

VERSION = "v001"
OUT_PROCESSED = PROJECT_ROOT / "data/processed"
OUT_SPLITS = PROJECT_ROOT / "data/splits"
OUT_EVAL_SETS = PROJECT_ROOT / "eval/sets"
OUT_HUMAN_REVIEW = PROJECT_ROOT / "eval/human_review"
OUT_RUN = PROJECT_ROOT / f"runs/liuyichen/showee_ai_label_{VERSION}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def find_head_video(sample_dir: Path) -> Path | None:
    candidates = sorted((sample_dir / "showee_head").glob("*.mkv"))
    return candidates[0] if candidates else None


def build_index(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(raw_root.glob("*/*/metadata.json")):
        sample_dir = metadata_path.parent
        video_path = find_head_video(sample_dir)
        if video_path is None:
            continue
        try:
            metadata = load_json(metadata_path)
        except Exception as exc:
            rows.append(
                {
                    "sample_id": safe_id(str(sample_dir.relative_to(raw_root))),
                    "sample_dir": str(sample_dir),
                    "video_path": "",
                    "metadata_path": str(metadata_path),
                    "readable": False,
                    "error": repr(exc),
                }
            )
            continue

        session = metadata.get("session", {})
        task = metadata.get("task", {})
        subject_id = str(session.get("subject_id", "unknown_subject"))
        session_type = str(session.get("session_type", "unknown_session"))
        session_repeat = str(session.get("session_repeat", "unknown_repeat"))
        task_id = str(task.get("id", sample_dir.name))
        task_name = str(task.get("name") or task.get("name_en") or task_id)
        task_desc = str(task.get("description") or task.get("desc_en") or task_name)
        sample_id = safe_id(
            f"showee_{subject_id}_{session_type}_{session_repeat}_{task_id}_head"
        )
        rows.append(
            {
                "sample_id": sample_id,
                "sample_dir": str(sample_dir),
                "video_path": str(video_path),
                "metadata_path": str(metadata_path),
                "subject_id": subject_id,
                "session_type": session_type,
                "session_repeat": session_repeat,
                "task_id": task_id,
                "task_name": task_name,
                "task_desc": task_desc,
                "duration": task.get("duration"),
                "source_view": "showee_head",
                "readable": video_path.exists(),
                "error": "",
            }
        )
    return rows


def select_diverse(rows: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for row in rows:
        if not row.get("readable"):
            continue
        task_id = str(row["task_id"])
        if task_id in seen_task_ids:
            continue
        selected.append(row)
        seen_task_ids.add(task_id)
        if len(selected) >= max_samples:
            return selected
    for row in rows:
        if row.get("readable") and row not in selected:
            selected.append(row)
            if len(selected) >= max_samples:
                return selected
    return selected


def planned_split_by_id(selected: list[dict[str, Any]]) -> dict[str, str]:
    splits = ["train"] * 14 + ["val"] * 3 + ["test"] * 3
    return {
        row["sample_id"]: split
        for row, split in zip(selected, splits)
    }


def write_csv(path: Path, rows: list[dict[str, Any]], selected_ids: set[str], split_by_id: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "selected_v001",
        "split_v001",
        "video_path",
        "metadata_path",
        "subject_id",
        "session_type",
        "session_repeat",
        "task_id",
        "task_name",
        "task_desc",
        "duration",
        "source_view",
        "readable",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in fieldnames}
            out["selected_v001"] = row.get("sample_id") in selected_ids
            out["split_v001"] = split_by_id.get(str(row.get("sample_id")), "")
            writer.writerow(out)


def load_video_frames(path: Path, num_frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0))
    total_frames = len(reader)
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {path}")
    fps = float(reader.get_avg_fps())
    sample_count = min(num_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, sample_count, dtype=np.int64)
    frame_times = [
        {
            "sample_index": idx,
            "frame_index": int(frame_idx),
            "time_sec": float(frame_idx / fps) if fps > 0 else None,
        }
        for idx, frame_idx in enumerate(indices.tolist())
    ]
    return reader.get_batch(indices).asnumpy(), {
        "total_frames": int(total_frames),
        "fps": fps,
        "duration_seconds_est": float(total_frames / fps) if fps > 0 else None,
        "sampled_frames": frame_times,
    }


def build_prompt(row: dict[str, Any], video_meta: dict[str, Any]) -> str:
    sampled = ", ".join(
        f"{f['sample_index']}={f['time_sec']:.2f}s"
        for f in video_meta["sampled_frames"]
        if f["time_sec"] is not None
    )
    return (
        "你是用于生成训练数据初稿的视觉标注助手。"
        "请观察第一视角手势视频，并结合给定元数据生成保守、可复核的标注。"
        "不要编造看不清的细节；看不清时明确写不确定。"
        "只输出 JSON，不要 Markdown。\n"
        f"任务 ID: {row['task_id']}\n"
        f"任务名称: {row['task_name']}\n"
        f"元数据动作说明: {row['task_desc']}\n"
        f"采集视角: {row['source_view']}\n"
        f"抽样帧时间点: {sampled}\n"
        "JSON 字段固定为："
        "{\"task_summary\":\"\","
        "\"visible_evidence\":\"\","
        "\"hand_motion\":\"\","
        "\"temporal_change\":\"\","
        "\"uncertainty\":\"\"}。"
        "每个字段用中文，单字段不超过 60 个汉字。"
    )


def parse_json_text(text: str) -> dict[str, str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {
            "task_summary": "",
            "visible_evidence": "",
            "hand_motion": "",
            "temporal_change": "",
            "uncertainty": f"model_output_not_json: {text[:180]}",
        }
    return {key: str(parsed.get(key, "")).strip() for key in [
        "task_summary",
        "visible_evidence",
        "hand_motion",
        "temporal_change",
        "uncertainty",
    ]}


def qwen_annotate(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    num_frames: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    frames, video_meta = load_video_frames(Path(row["video_path"]), num_frames)
    prompt = build_prompt(row, video_meta)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    parsed = parse_json_text(output_text)
    return {
        "raw_model_output": output_text,
        "parsed": parsed,
        "video_metadata": video_meta,
        "num_frames": int(frames.shape[0]),
    }


def answer_text(row: dict[str, Any], ann: dict[str, str], kind: str) -> str:
    task_name = row["task_name"]
    task_desc = row["task_desc"]
    if kind == "summary":
        visual = ann.get("task_summary") or ann.get("visible_evidence") or "画面显示手部正在执行该任务动作"
        return (
            f"视频展示的是 {task_name} 手势任务。"
            f"元数据说明该任务为：{task_desc}。"
            f"视觉初稿观察：{visual}"
        )
    if kind == "motion":
        visual = ann.get("hand_motion") or ann.get("visible_evidence") or "主要关注手指弯曲、伸展和手腕姿态变化"
        return f"关键手部动作包括：{task_desc}。视觉初稿补充：{visual}"
    temporal = ann.get("temporal_change") or "动作整体围绕同一手势任务展开，具体阶段变化需要人工复核。"
    uncertainty = ann.get("uncertainty")
    suffix = f" 不确定点：{uncertainty}" if uncertainty else ""
    return f"时间变化初稿：{temporal}{suffix}"


def make_sharegpt_sample(row: dict[str, Any], annotation: dict[str, Any], split: str) -> dict[str, Any]:
    ann = annotation["parsed"]
    return {
        "id": row["sample_id"],
        "video": row["video_path"],
        "metadata": {
            "dataset": "ShoweeHandv2",
            "source_view": "showee_head",
            "metadata_path": row["metadata_path"],
            "subject_id": row["subject_id"],
            "session_type": row["session_type"],
            "session_repeat": row["session_repeat"],
            "task_id": row["task_id"],
            "task_name": row["task_name"],
            "duration": row.get("duration"),
            "split": split,
            "annotation_source": "qwen3vl_ai_annotation_v001",
            "review_status": "unreviewed",
        },
        "conversations": [
            {
                "from": "user",
                "value": "<video>\n请描述视频中正在执行的手势任务。",
            },
            {
                "from": "assistant",
                "value": answer_text(row, ann, "summary"),
            },
            {
                "from": "user",
                "value": "这个手势的关键手部动作是什么？",
            },
            {
                "from": "assistant",
                "value": answer_text(row, ann, "motion"),
            },
            {
                "from": "user",
                "value": "动作从头到尾有什么变化？如果看不清也说明。",
            },
            {
                "from": "assistant",
                "value": answer_text(row, ann, "temporal"),
            },
        ],
    }


def split_samples(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": samples[:14],
        "val": samples[14:17],
        "test": samples[17:20],
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def make_eval_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        conversations = sample["conversations"]
        first_answer = conversations[1]["value"]
        rows.append(
            {
                "id": sample["id"],
                "video": sample["video"],
                "metadata": sample["metadata"],
                "question": "请描述视频中正在执行的手势任务，并说明关键手部动作。",
                "reference_answer": first_answer,
                "eval_focus": [
                    "是否匹配 metadata 中的 task_name/task_desc",
                    "是否提到手指、手腕或手部姿态等关键动作",
                    "是否避免编造无法从视频确认的细节",
                    "是否说明不确定或看不清的部分",
                ],
            }
        )
    return rows


def write_review_template(path: Path, samples: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ShoweeData v001 人工复核表",
        "",
        "状态可填：accepted / edited / rejected。",
        "重点检查：任务名是否正确、手指/手腕动作是否准确、是否有幻觉、是否需要改写。",
        "",
    ]
    ann_by_id = {ann["sample_id"]: ann for ann in annotations}
    for idx, sample in enumerate(samples, start=1):
        ann = ann_by_id.get(sample["id"], {})
        parsed = ann.get("annotation", {}).get("parsed", {})
        meta = sample["metadata"]
        lines.extend(
            [
                f"## {idx}. {sample['id']}",
                "",
                f"- status: unreviewed",
                f"- split: {meta['split']}",
                f"- video: {sample['video']}",
                f"- task: {meta['task_id']} / {meta['task_name']}",
                f"- model_task_summary: {parsed.get('task_summary', '')}",
                f"- model_visible_evidence: {parsed.get('visible_evidence', '')}",
                f"- model_hand_motion: {parsed.get('hand_motion', '')}",
                f"- model_temporal_change: {parsed.get('temporal_change', '')}",
                f"- model_uncertainty: {parsed.get('uncertainty', '')}",
                "",
                "review_notes:",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--index-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    rows = build_index(args.raw_root)
    selected = select_diverse(rows, args.max_samples)
    selected_ids = {row["sample_id"] for row in selected}
    split_by_id = planned_split_by_id(selected)
    write_csv(OUT_PROCESSED / f"video_index_showee_{VERSION}.csv", rows, selected_ids, split_by_id)
    if args.index_only:
        print(f"indexed {len(rows)} rows; selected {len(selected)}")
        return

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model),
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)

    raw_annotations: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        sample_start = time.time()
        annotation = qwen_annotate(
            model,
            processor,
            row,
            num_frames=args.num_frames,
            max_new_tokens=args.max_new_tokens,
        )
        item = {
            "sample_id": row["sample_id"],
            "video": row["video_path"],
            "metadata_path": row["metadata_path"],
            "task_id": row["task_id"],
            "task_name": row["task_name"],
            "task_desc": row["task_desc"],
            "annotation": annotation,
            "review_status": "unreviewed",
            "elapsed_seconds": round(time.time() - sample_start, 3),
        }
        raw_annotations.append(item)
        write_json(OUT_PROCESSED / f"showee_ai_annotations_raw_{VERSION}.json", raw_annotations)
        print(json.dumps({"done": idx, "sample_id": row["sample_id"], "elapsed": item["elapsed_seconds"]}, ensure_ascii=False), flush=True)

    split_names = ["train"] * 14 + ["val"] * 3 + ["test"] * 3
    samples = [
        make_sharegpt_sample(row, ann["annotation"], split)
        for row, ann, split in zip(selected, raw_annotations, split_names)
    ]
    splits = split_samples(samples)

    write_json(OUT_PROCESSED / f"showee_train_{VERSION}.json", splits["train"])
    write_json(OUT_PROCESSED / f"showee_val_{VERSION}.json", splits["val"])
    write_json(OUT_PROCESSED / f"showee_test_{VERSION}.json", splits["test"])
    write_json(OUT_SPLITS / f"showee_train_{VERSION}.json", splits["train"])
    write_json(OUT_SPLITS / f"showee_val_{VERSION}.json", splits["val"])
    write_json(OUT_SPLITS / f"showee_test_{VERSION}.json", splits["test"])
    write_json(OUT_EVAL_SETS / f"showee_val_{VERSION}.json", make_eval_rows(splits["val"]))
    write_json(OUT_EVAL_SETS / f"showee_test_{VERSION}.json", make_eval_rows(splits["test"]))
    write_review_template(
        OUT_HUMAN_REVIEW / f"showee_review_{VERSION}.md",
        samples,
        raw_annotations,
    )
    write_json(
        OUT_RUN / "run_report.json",
        {
            "version": VERSION,
            "indexed_rows": len(rows),
            "selected_rows": len(selected),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "num_frames": args.num_frames,
            "elapsed_seconds": round(time.time() - start, 3),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        },
    )


if __name__ == "__main__":
    main()
