#!/usr/bin/env python3
"""Build candidate task-name choice training samples for ShoweeData v002."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(
    os.environ.get("SHOWEE_PIPELINE_ROOT", Path(__file__).resolve().parents[1])
)
TRAIN_V002 = PROJECT_ROOT / "data/processed/showee_train_v002.json"
OUT_CHOICE = PROJECT_ROOT / "data/processed/showee_train_v002_choice.json"
OUT_CHOICE_MESSAGES = PROJECT_ROOT / "data/processed/showee_train_v002_choice_messages.json"
OUT_MIXED = PROJECT_ROOT / "data/processed/showee_train_v002_with_choice.json"
OUT_REPORT = PROJECT_ROOT / "runs/puzuo/showee_choice_train_v002/build_report.json"


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


def task_group(task_id: str) -> list[str]:
    for group in TASK_GROUPS.values():
        if task_id in group:
            return group
    return [task_id]


def task_names_by_id(samples: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(sample["metadata"]["task_id"]): str(sample["metadata"]["task_name"])
        for sample in samples
    }


def rotate(items: list[str], key: str) -> list[str]:
    if not items:
        return items
    offset = sum(ord(ch) for ch in key) % len(items)
    return items[offset:] + items[:offset]


def hard_candidates(task_id: str, names: dict[str, str]) -> list[dict[str, str]]:
    if task_id in ASL_TASKS:
        pool = ["asl_g", "asl_i", "asl_k", "asl_l", "asl_y", "asl_0", "asl_1", "asl_2", "asl_5", "asl_8"]
    else:
        pool = task_group(task_id)
    ordered = [task_id] + [x for x in pool if x != task_id]
    if len(ordered) < 6:
        ordered.extend(x for x in names if x not in ordered)
    task_ids = rotate(ordered[:6], f"hard:{task_id}")
    return [{"task_id": x, "task_name": names.get(x, x)} for x in task_ids]


def mixed_candidates(task_id: str, names: dict[str, str]) -> list[dict[str, str]]:
    group = task_group(task_id)
    distractors: list[str] = []
    for key in sorted(names):
        if key != task_id and key not in group:
            distractors.append(key)
    ordered = [task_id]
    ordered.extend(x for x in group if x != task_id)
    ordered.extend(distractors)
    task_ids = rotate(ordered[:8], f"mixed:{task_id}")[:6]
    if task_id not in task_ids:
        task_ids[-1] = task_id
        task_ids = rotate(task_ids, f"mixed-answer:{task_id}")
    return [{"task_id": x, "task_name": names.get(x, x)} for x in task_ids]


def option_text(candidates: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{idx + 1}. {item['task_name']} ({item['task_id']})"
        for idx, item in enumerate(candidates)
    )


def user_prompt(candidates: list[dict[str, str]]) -> str:
    return (
        "<video>\n"
        "候选任务如下：\n"
        f"{option_text(candidates)}\n"
        "请判断视频属于哪一个，并只输出 JSON。"
        "JSON 格式固定为：{\"task_id\":\"...\",\"task_name\":\"...\"}。"
    )


def answer_json(sample: dict[str, Any]) -> str:
    meta = sample["metadata"]
    return json.dumps(
        {
            "task_id": str(meta["task_id"]),
            "task_name": str(meta["task_name"]),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def safe_suffix(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").lower()


def make_choice_sample(sample: dict[str, Any], candidates: list[dict[str, str]], variant: str) -> dict[str, Any]:
    meta = json.loads(json.dumps(sample["metadata"], ensure_ascii=False))
    meta["choice_variant"] = variant
    meta["training_objective"] = "candidate_task_choice"
    meta["candidate_tasks"] = candidates
    return {
        "id": f"{sample['id']}_choice_{safe_suffix(variant)}",
        "video": sample["video"],
        "metadata": meta,
        "conversations": [
            {
                "from": "user",
                "value": user_prompt(candidates),
            },
            {
                "from": "assistant",
                "value": answer_json(sample),
            },
        ],
    }


def to_messages_sample(choice_sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": choice_sample["id"],
        "video": choice_sample["video"],
        "metadata": choice_sample["metadata"],
        "messages": [
            {
                "role": "user",
                "content": choice_sample["conversations"][0]["value"].replace("<video>\n", ""),
            },
            {
                "role": "assistant",
                "content": choice_sample["conversations"][1]["value"],
            },
        ],
    }


def main() -> None:
    train = read_json(TRAIN_V002)
    names = task_names_by_id(train)
    choice_samples: list[dict[str, Any]] = []
    for sample in train:
        task_id = str(sample["metadata"]["task_id"])
        choice_samples.append(make_choice_sample(sample, hard_candidates(task_id, names), "hard_same_family"))
        choice_samples.append(make_choice_sample(sample, mixed_candidates(task_id, names), "mixed_family"))

    messages_samples = [to_messages_sample(sample) for sample in choice_samples]
    mixed_train = train + choice_samples

    write_json(OUT_CHOICE, choice_samples)
    write_json(OUT_CHOICE_MESSAGES, messages_samples)
    write_json(OUT_MIXED, mixed_train)
    report = {
        "source_train": str(TRAIN_V002),
        "source_train_count": len(train),
        "choice_samples": len(choice_samples),
        "choice_samples_per_video": 2,
        "mixed_train_count": len(mixed_train),
        "outputs": {
            "choice_sharegpt": str(OUT_CHOICE),
            "choice_messages": str(OUT_CHOICE_MESSAGES),
            "mixed_train": str(OUT_MIXED),
        },
        "variants": ["hard_same_family", "mixed_family"],
    }
    write_json(OUT_REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
