#!/usr/bin/env python3
"""Build ShoweeData v002 from reviewed v001 plus metadata-seeded expansion.

v002 goals:
  - expand the small Showee set to 50 samples,
  - make train cover the same or similar task types used in val/test,
  - split open-ended description evaluation from candidate task-name selection.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(
    os.environ.get("SHOWEE_PIPELINE_ROOT", Path(__file__).resolve().parents[1])
)
RAW_ROOT = Path(
    os.environ.get("SHOWEE_RAW_ROOT", PROJECT_ROOT / "data/raw/ShoweeHandv2/raw")
)
OUT_PROCESSED = PROJECT_ROOT / "data/processed"
OUT_SPLITS = PROJECT_ROOT / "data/splits"
OUT_EVAL_SETS = PROJECT_ROOT / "eval/sets"
OUT_REVIEW = PROJECT_ROOT / "eval/human_review"
VERSION = "v002"


V001_FILES = [
    OUT_PROCESSED / "showee_train_v001.json",
    OUT_PROCESSED / "showee_val_v001.json",
    OUT_PROCESSED / "showee_test_v001.json",
]


EXTRA_TRAIN_TASKS = [
    "asl_g",
    "asl_i",
    "asl_k",
    "asl_l",
    "asl_y",
    "finger_wave",
    "hand_shape",
    "middle_pinky",
    "seq_flex",
    "seq_pinch",
    "ring_bend",
    "index_bend",
    "thumb_swipe",
    "thumb",
    "pinky_bend",
    "middle_bend",
    "thumb_touchpad",
    "free_style",
    "fist_open",
    "thumb_middle",
    "ily",
    "wrist",
    "seq_extend",
    "middle",
    "index_pinky",
    "ring_pinky",
]

VAL_TASKS = ["asl_g", "asl_i", "asl_k", "asl_l", "asl_y"]
TEST_TASKS = ["finger_wave", "hand_shape", "seq_flex"]


ASL_TASKS = [
    "asl_0",
    "asl_1",
    "asl_2",
    "asl_3",
    "asl_4",
    "asl_5",
    "asl_6",
    "asl_7",
    "asl_8",
    "asl_9",
    "asl_a",
    "asl_c",
    "asl_e",
    "asl_g",
    "asl_i",
    "asl_k",
    "asl_l",
    "asl_y",
]

TASK_GROUPS = {
    "asl": ASL_TASKS,
    "finger_sequence": ["finger_wave", "seq_flex", "seq_extend", "seq_pinch"],
    "single_finger": [
        "thumb",
        "middle",
        "index_bend",
        "middle_bend",
        "ring_bend",
        "pinky_bend",
        "thumb_swipe",
        "thumb_touchpad",
    ],
    "finger_combo": [
        "index_pinky",
        "middle_pinky",
        "ring_pinky",
        "thumb_middle",
        "ily",
        "hand_shape",
        "fist_open",
        "abduct_adduct",
        "wrist",
        "free_style",
    ],
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def head_video(sample_dir: Path) -> Path:
    candidates = sorted((sample_dir / "showee_head").glob("*.mkv"))
    if not candidates:
        raise FileNotFoundError(f"no showee_head video under {sample_dir}")
    return candidates[0]


def load_metadata_sample(session_dir: str, task_id: str, split: str) -> dict[str, Any]:
    sample_dir = RAW_ROOT / session_dir / task_id
    metadata_path = sample_dir / "metadata.json"
    metadata = read_json(metadata_path)
    session = metadata["session"]
    task = metadata["task"]
    subject_id = str(session.get("subject_id", "unknown_subject"))
    session_type = str(session.get("session_type", "unknown_session"))
    session_repeat = str(session.get("session_repeat", "unknown_repeat"))
    task_name = str(task.get("name") or task.get("name_en") or task_id)
    task_desc = str(task.get("description") or task.get("desc_en") or task_name)
    sample_id = safe_id(f"showee_{subject_id}_{session_type}_{session_repeat}_{task_id}_head")
    video = str(head_video(sample_dir))
    return make_metadata_seeded_sample(
        sample_id=sample_id,
        video=video,
        metadata_path=str(metadata_path),
        subject_id=subject_id,
        session_type=session_type,
        session_repeat=session_repeat,
        task_id=task_id,
        task_name=task_name,
        task_desc=task_desc,
        duration=task.get("duration"),
        split=split,
    )


def motion_hint(task_desc: str) -> str:
    hints: list[str] = []
    if any(x in task_desc for x in ["转动手腕", "手腕"]):
        hints.append("手腕转动或朝向变化")
    if any(x in task_desc for x in ["弯曲", "屈"]):
        hints.append("手指弯曲程度变化")
    if any(x in task_desc for x in ["伸展", "伸出", "张开"]):
        hints.append("手指伸展或张开")
    if any(x in task_desc for x in ["并拢", "分合", "分和"]):
        hints.append("手指分合")
    if any(x in task_desc for x in ["捏合", "触", "touch"]):
        hints.append("指尖接触或捏合")
    if any(x in task_desc for x in ["顺序", "波浪"]):
        hints.append("按序连续变化")
    return "、".join(hints) if hints else "手部姿态保持与轻微变化"


def make_metadata_seeded_sample(
    *,
    sample_id: str,
    video: str,
    metadata_path: str,
    subject_id: str,
    session_type: str,
    session_repeat: str,
    task_id: str,
    task_name: str,
    task_desc: str,
    duration: Any,
    split: str,
) -> dict[str, Any]:
    hint = motion_hint(task_desc)
    return {
        "id": sample_id,
        "video": video,
        "metadata": {
            "dataset": "ShoweeHandv2",
            "source_view": "showee_head",
            "metadata_path": metadata_path,
            "subject_id": subject_id,
            "session_type": session_type,
            "session_repeat": session_repeat,
            "task_id": task_id,
            "task_name": task_name,
            "task_desc": task_desc,
            "duration": duration,
            "split": split,
            "annotation_source": "metadata_seeded_v002",
            "review_status": "metadata_seeded_unreviewed",
            "review_notes": "Metadata-derived v002 seed label; requires visual review before treating as final truth.",
        },
        "conversations": [
            {
                "from": "user",
                "value": "<video>\n请描述视频中正在执行的手势任务。",
            },
            {
                "from": "assistant",
                "value": (
                    f"视频展示的是 {task_name} 手势任务。根据元数据，任务要求为{task_desc}。"
                    "该样本来自 showee_head 头戴视角，当前标注以任务元数据为主，视觉细节需要后续人工复核。"
                ),
            },
            {
                "from": "user",
                "value": "这个手势的关键手部动作是什么？",
            },
            {
                "from": "assistant",
                "value": (
                    f"关键手部动作包括：{task_desc}。可重点关注{hint}。"
                    "如果头戴视角下看不清具体指尖接触关系，不应做过度细化。"
                ),
            },
            {
                "from": "user",
                "value": "动作从头到尾有什么变化？如果看不清也说明。",
            },
            {
                "from": "assistant",
                "value": (
                    f"整段视频应围绕 {task_name} 任务展开，主要观察点是{hint}。"
                    "当前为元数据种子标注，精确动作边界和逐帧顺序需要人工复核。"
                ),
            },
        ],
    }


def load_v001_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in V001_FILES:
        samples.extend(read_json(path))
    return samples


def retag_v001(sample: dict[str, Any], split: str) -> dict[str, Any]:
    out = json.loads(json.dumps(sample, ensure_ascii=False))
    out["metadata"]["split"] = split
    out["metadata"]["annotation_source"] = out["metadata"].get(
        "annotation_source", "human_review_v001_initial_from_ai"
    )
    out["metadata"]["v002_origin"] = "v001_reviewed"
    return out


def task_group(task_id: str) -> list[str]:
    for group in TASK_GROUPS.values():
        if task_id in group:
            return group
    return sorted({task_id, *ASL_TASKS[:5]})


def task_names_by_id(samples: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for sample in samples:
        meta = sample["metadata"]
        names[str(meta["task_id"])] = str(meta["task_name"])
    return names


def candidate_options(task_id: str, names: dict[str, str]) -> list[dict[str, str]]:
    if task_id in ASL_TASKS:
        group = [
            "asl_g",
            "asl_i",
            "asl_k",
            "asl_l",
            "asl_y",
            "asl_0",
            "asl_1",
            "asl_2",
            "asl_5",
            "asl_8",
        ]
    else:
        group = task_group(task_id)
    ordered = [task_id] + [x for x in group if x != task_id]
    if len(ordered) < 6:
        ordered.extend(x for x in ASL_TASKS if x not in ordered)
    task_ids = ordered[:6]
    offset = sum(ord(ch) for ch in task_id) % len(task_ids)
    task_ids = task_ids[offset:] + task_ids[:offset]
    return [{"task_id": x, "task_name": names.get(x, x)} for x in task_ids]


def first_answer(sample: dict[str, Any]) -> str:
    return str(sample["conversations"][1]["value"])


def make_open_eval(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        rows.append(
            {
                "id": sample["id"],
                "video": sample["video"],
                "metadata": sample["metadata"],
                "eval_type": "open_description",
                "question": "请描述视频中正在执行的手势任务，并说明关键手部动作。",
                "reference_answer": first_answer(sample),
                "eval_focus": [
                    "开放式回答是否提到正确 task_name 或等价任务名",
                    "是否描述关键手指/手腕动作",
                    "是否避免把 ASL 手势泛化成文化手势或设备交互",
                    "看不清时是否表达不确定而不是编造",
                ],
            }
        )
    return rows


def make_choice_eval(samples: list[dict[str, Any]], all_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = task_names_by_id(all_samples)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        meta = sample["metadata"]
        task_id = str(meta["task_id"])
        options = candidate_options(task_id, names)
        option_text = "\n".join(
            f"{idx + 1}. {opt['task_name']} ({opt['task_id']})"
            for idx, opt in enumerate(options)
        )
        rows.append(
            {
                "id": sample["id"],
                "video": sample["video"],
                "metadata": meta,
                "eval_type": "candidate_task_choice",
                "question": (
                    "请只从下面候选任务名中选择最符合视频的一个，并用 JSON 输出："
                    "{\"task_id\":\"...\",\"task_name\":\"...\",\"reason\":\"...\"}。\n"
                    f"候选任务：\n{option_text}"
                ),
                "candidate_tasks": options,
                "answer_task_id": task_id,
                "answer_task_name": str(meta["task_name"]),
                "reference_answer": {
                    "task_id": task_id,
                    "task_name": str(meta["task_name"]),
                },
                "eval_focus": [
                    "候选任务 task_id 是否选择正确",
                    "reason 是否引用可见手型或动作依据",
                    "是否严格从候选列表选择",
                ],
            }
        )
    return rows


def write_review_template(path: Path, samples: list[dict[str, Any]]) -> None:
    lines = [
        "# ShoweeData v002 人工复核表",
        "",
        "状态可填：accepted / edited / rejected。",
        "v001 继承项已人工初步复核；metadata_seeded_unreviewed 为 v002 新增元数据种子标注，需要后续视觉复核。",
        "",
    ]
    for idx, sample in enumerate(samples, start=1):
        meta = sample["metadata"]
        lines.extend(
            [
                f"## {idx}. {sample['id']}",
                "",
                f"- status: {meta.get('review_status', '')}",
                f"- split: {meta['split']}",
                f"- video: {sample['video']}",
                f"- task: {meta['task_id']} / {meta['task_name']}",
                f"- annotation_source: {meta.get('annotation_source', '')}",
                f"- task_desc: {meta.get('task_desc', '')}",
                "",
                "review_notes:",
                str(meta.get("review_notes", "")),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_index(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "split_v002",
        "task_id",
        "task_name",
        "subject_id",
        "session_type",
        "session_repeat",
        "annotation_source",
        "review_status",
        "video",
        "metadata_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            meta = sample["metadata"]
            writer.writerow(
                {
                    "sample_id": sample["id"],
                    "split_v002": meta["split"],
                    "task_id": meta["task_id"],
                    "task_name": meta["task_name"],
                    "subject_id": meta["subject_id"],
                    "session_type": meta["session_type"],
                    "session_repeat": meta["session_repeat"],
                    "annotation_source": meta.get("annotation_source", ""),
                    "review_status": meta.get("review_status", ""),
                    "video": sample["video"],
                    "metadata_path": meta.get("metadata_path", ""),
                }
            )


def main() -> None:
    v001_by_id = {sample["id"]: sample for sample in load_v001_samples()}
    v001_train = [retag_v001(sample, "train") for sample in read_json(OUT_PROCESSED / "showee_train_v001.json")]
    v001_val = [retag_v001(sample, "val") for sample in read_json(OUT_PROCESSED / "showee_val_v001.json")]
    v001_test = [retag_v001(sample, "test") for sample in read_json(OUT_PROCESSED / "showee_test_v001.json")]

    train = list(v001_train)
    train.extend(load_metadata_sample("20260521_0006_midair_2", task_id, "train") for task_id in EXTRA_TRAIN_TASKS)
    val = list(v001_val)
    val.extend(load_metadata_sample("20260521_0007_midair_1", task_id, "val") for task_id in VAL_TASKS if task_id not in {"asl_g", "asl_i", "asl_k"})
    test = list(v001_test)
    test.extend(load_metadata_sample("20260521_0007_midair_1", task_id, "test") for task_id in TEST_TASKS if task_id not in {"finger_wave"})

    assert len(train) == 40, len(train)
    assert len(val) == 5, len(val)
    assert len(test) == 5, len(test)
    samples = train + val + test
    ids = [sample["id"] for sample in samples]
    if len(ids) != len(set(ids)):
        dupes = sorted({x for x in ids if ids.count(x) > 1})
        raise RuntimeError(f"duplicate sample ids: {dupes}")

    train_task_ids = {sample["metadata"]["task_id"] for sample in train}
    eval_task_ids = {sample["metadata"]["task_id"] for sample in val + test}
    missing = sorted(eval_task_ids - train_task_ids)
    if missing:
        raise RuntimeError(f"eval task ids missing from train: {missing}")

    write_json(OUT_PROCESSED / f"showee_train_{VERSION}.json", train)
    write_json(OUT_PROCESSED / f"showee_val_{VERSION}.json", val)
    write_json(OUT_PROCESSED / f"showee_test_{VERSION}.json", test)
    write_json(OUT_SPLITS / f"showee_train_{VERSION}.json", train)
    write_json(OUT_SPLITS / f"showee_val_{VERSION}.json", val)
    write_json(OUT_SPLITS / f"showee_test_{VERSION}.json", test)

    write_json(OUT_EVAL_SETS / f"showee_val_{VERSION}_open.json", make_open_eval(val))
    write_json(OUT_EVAL_SETS / f"showee_test_{VERSION}_open.json", make_open_eval(test))
    write_json(OUT_EVAL_SETS / f"showee_val_{VERSION}_choice.json", make_choice_eval(val, samples))
    write_json(OUT_EVAL_SETS / f"showee_test_{VERSION}_choice.json", make_choice_eval(test, samples))
    write_review_template(OUT_REVIEW / f"showee_review_{VERSION}.md", samples)
    write_index(OUT_PROCESSED / f"video_index_showee_{VERSION}.csv", samples)

    report = {
        "version": VERSION,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "total": len(samples),
        "train_task_count": len(train_task_ids),
        "eval_task_ids": sorted(eval_task_ids),
        "eval_task_ids_missing_from_train": missing,
        "open_eval_sets": [
            f"eval/sets/showee_val_{VERSION}_open.json",
            f"eval/sets/showee_test_{VERSION}_open.json",
        ],
        "choice_eval_sets": [
            f"eval/sets/showee_val_{VERSION}_choice.json",
            f"eval/sets/showee_test_{VERSION}_choice.json",
        ],
    }
    write_json(PROJECT_ROOT / f"runs/puzuo/showee_dataset_{VERSION}/build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
