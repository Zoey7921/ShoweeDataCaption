#!/usr/bin/env python3
"""Create one Qwen3-VL smoke-training sample from ShoweeHandv2.

The script selects one task recording, uses the showee_head RGB video, reads
metadata.json, and writes a ShareGPT-like multi-turn video QA sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/data1/shared_data/qwen3vl-showeeData")
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data/raw/ShoweeHandv2/raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/smoke_train.json"
DEFAULT_INDEX = PROJECT_ROOT / "data/processed/smoke_source_index.json"
DEFAULT_SAMPLE_DIR = Path("20260521_0006_midair_1/asl_1")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_head_video(sample_dir: Path) -> Path:
    head_dir = sample_dir / "showee_head"
    candidates = sorted(head_dir.glob("*.mkv"))
    if not candidates:
        raise FileNotFoundError(f"No showee_head mkv found under {head_dir}")
    if len(candidates) > 1:
        raise RuntimeError(f"Expected one showee_head mkv under {head_dir}, got {len(candidates)}")
    return candidates[0]


def make_sample(sample_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

    metadata = load_json(metadata_path)
    video_path = find_head_video(sample_dir)

    session = metadata.get("session", {})
    task = metadata.get("task", {})

    subject_id = session.get("subject_id", "unknown_subject")
    session_type = session.get("session_type", "unknown_session")
    session_repeat = session.get("session_repeat", "unknown_repeat")
    task_id = task.get("id", sample_dir.name)
    task_name = task.get("name") or task.get("name_en") or task_id
    task_desc = task.get("description") or task.get("desc_en") or task_name
    duration = task.get("duration")

    sample_id = f"showee_{subject_id}_{session_type}_{session_repeat}_{task_id}_head_smoke"

    duration_text = f"视频时长约 {duration} 秒。" if duration is not None else "视频时长见 metadata。"
    assistant_answer_1 = (
        f"视频展示的是 {task_name} 手势任务。根据元数据说明，动作内容是：{task_desc}"
    )
    assistant_answer_2 = (
        f"这是一次 {session_type} 采集，受试者编号为 {subject_id}，重复次数为 {session_repeat}。"
        f"当前样本使用头戴视角 showee_head 的 RGB 视频，{duration_text}"
    )

    sample = {
        "id": sample_id,
        "video": str(video_path),
        "metadata": {
            "dataset": "ShoweeHandv2",
            "source_view": "showee_head",
            "metadata_path": str(metadata_path),
            "subject_id": subject_id,
            "session_type": session_type,
            "session_repeat": session_repeat,
            "task_id": task_id,
            "task_name": task_name,
            "duration": duration,
        },
        "conversations": [
            {
                "from": "user",
                "value": "<video>\n请描述视频中正在执行的手势任务。",
            },
            {
                "from": "assistant",
                "value": assistant_answer_1,
            },
            {
                "from": "user",
                "value": "这个样本来自哪个采集视角？还有哪些关键信息？",
            },
            {
                "from": "assistant",
                "value": assistant_answer_2,
            },
        ],
    }

    source_index = {
        "sample_id": sample_id,
        "sample_dir": str(sample_dir),
        "video": str(video_path),
        "metadata": str(metadata_path),
        "task": task,
        "session": session,
    }
    return sample, source_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index-output", type=Path, default=DEFAULT_INDEX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = args.sample_dir
    if not sample_dir.is_absolute():
        sample_dir = args.raw_root / sample_dir

    sample, source_index = make_sample(sample_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump([sample], f, ensure_ascii=False, indent=2)
        f.write("\n")

    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    with args.index_output.open("w", encoding="utf-8") as f:
        json.dump(source_index, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"wrote sample: {args.output}")
    print(f"wrote source index: {args.index_output}")
    print(f"sample id: {sample['id']}")
    print(f"video: {sample['video']}")


if __name__ == "__main__":
    main()
