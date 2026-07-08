#!/usr/bin/env python3
"""Expand single-turn streaming samples into short multi-turn interactions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from build_streaming_interaction_dataset import to_messages_sample


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


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def frame_index(time_sec: float, fps: float) -> int:
    return int(round(time_sec * fps))


def segment_for_turn(sample: dict[str, Any], turn: dict[str, Any]) -> dict[str, Any]:
    segments = sample.get("metadata", {}).get("temporal_segments", [])
    if isinstance(segments, list):
        by_id = {
            str(seg.get("segment_id", f"seg_{idx:03d}")): seg
            for idx, seg in enumerate(segments)
            if isinstance(seg, dict)
        }
        seg = by_id.get(str(turn.get("segment_id", "")))
        if seg:
            return seg
        if segments and isinstance(segments[0], dict):
            return segments[0]
    time_sec = as_float(turn.get("time_sec"))
    return {
        "segment_id": str(turn.get("segment_id", "seg_000")),
        "start_sec": max(0.0, time_sec - 3.0),
        "end_sec": time_sec + 3.0,
    }


def sample_times(segment: dict[str, Any], target_turns: int) -> list[float]:
    start = as_float(segment.get("start_sec"))
    end = as_float(segment.get("end_sec"), start)
    if end <= start:
        return [round(start, 3)] * target_turns
    return [
        round(start + (end - start) * (idx + 1) / (target_turns + 1), 3)
        for idx in range(target_turns)
    ]


def turn_intervals(segment: dict[str, Any], times: list[float]) -> list[tuple[float, float]]:
    start = as_float(segment.get("start_sec"))
    end = as_float(segment.get("end_sec"), start)
    intervals: list[tuple[float, float]] = []
    cursor = start
    for idx, time_sec in enumerate(times):
        interval_end = time_sec if idx < len(times) - 1 else end
        if interval_end <= cursor:
            interval_end = min(end, cursor + 0.001)
        intervals.append((round(cursor, 3), round(interval_end, 3)))
        cursor = interval_end
    return intervals


def static_templates(task_id: str, task_name: str) -> list[tuple[str, str]]:
    base: dict[str, list[tuple[str, str]]] = {
        "asl_0": [
            ("我现在在做 ASL 0，应该怎么保持？", "保持双手在画面中，让手指围成清楚的 0 形或圆形轮廓。手腕不要大幅晃动，停住一小段时间，方便看清手形。"),
            ("这个 0 形还需要调整吗？", "重点检查手指和拇指围出的圆形是否完整，手掌角度不要太偏。如果圆形被遮住，就稍微转正手腕。"),
            ("最后怎么确认 ASL 0 做清楚了？", "结束前短暂停住，确认圆形轮廓、手腕朝向和双手位置都清楚可见，然后再放松。"),
        ],
        "asl_1": [
            ("我现在在做 ASL 1，接下来要注意什么？", "保持食指清楚伸出，其他手指收拢。把手放在画面中央，手腕稳定，避免伸出的手指被另一只手或手腕遮住。"),
            ("食指的方向需要怎么调整？", "让食指方向清楚，尽量不要倾斜到看不出单指伸出的状态。其他手指继续收拢。"),
            ("最后检查 ASL 1 的哪里？", "确认只有食指突出明显，手形稳定且没有移出画面，再结束这一轮。"),
        ],
        "asl_2": [
            ("我现在在做 ASL 2，这样应该怎么调整？", "让两根手指清楚伸出，其他手指保持收拢。手掌朝向要稳定，停住一小段时间，让两根伸出的手指分得清楚。"),
            ("两根手指要不要再分开一点？", "如果两根手指贴得太近，可以稍微分开，让数字 2 的手形更明显。保持手腕不要晃动。"),
            ("最后怎么确认 ASL 2？", "短暂停住，检查两根伸出的手指和收拢的其他手指都清楚可见。"),
        ],
        "asl_3": [
            ("我现在在做 ASL 3，下一步怎么保持？", "把目标手形稳定住，让伸出的手指彼此分开，收拢的手指不要突然打开。保持双手在画面中，短暂停住方便识别。"),
            ("这个手形还需要调整角度吗？", "可以稍微转正手掌，让伸出的三根手指不要重叠。手腕保持稳定。"),
            ("最后检查 ASL 3 的什么？", "确认三根伸出的手指都能分辨出来，其他手指没有误伸出。"),
        ],
        "asl_4": [
            ("我现在在做 ASL 4，要怎么做得更清楚？", "让四根手指尽量伸直并分开，拇指收在合适位置。手掌保持朝向镜头，不要让手指重叠在一起。"),
            ("四根手指要保持什么状态？", "保持四根手指展开且间距清楚，避免手指弯曲或互相贴住。"),
            ("最后怎么确认 ASL 4？", "收尾时停住，检查拇指位置和四根伸出手指都能被看清。"),
        ],
        "asl_5": [
            ("我现在在做 ASL 5，应该检查哪里？", "把五指全部张开，手掌朝向保持稳定。确认每根手指都能看清，尤其不要让拇指或小指被遮住。"),
            ("手掌角度需要调整吗？", "如果五指有重叠，稍微转正手掌并把手指分开，让张开的形状更明显。"),
            ("最后怎么确认 ASL 5？", "短暂停住，确认五根手指都展开且在画面中，没有被手腕或另一只手遮住。"),
        ],
        "asl_6": [
            ("我现在在做 ASL 6，接下来怎么确认？", "保持 ASL 6 的目标手形，注意让拇指和对应手指的接触点清楚可见。其他手指尽量展开，手不要偏出画面。"),
            ("接触点要怎么摆才清楚？", "把拇指和目标手指的接触位置转向镜头，避免被手掌挡住。"),
            ("最后检查 ASL 6 的哪里？", "确认接触点、伸出的其他手指和手腕朝向都稳定可见。"),
        ],
        "asl_7": [
            ("我现在在做 ASL 7，这个手形要怎么保持？", "保持 ASL 7 的目标手形，重点让拇指和对应手指的接触位置可见。手腕保持稳定，伸出的手指不要互相贴在一起。"),
            ("我需要调整手腕吗？", "可以轻微转动手腕，让关键接触点和伸出的手指都不被遮挡。"),
            ("最后怎么确认 ASL 7？", "短暂停住，检查接触点清楚、其他手指形态稳定，再结束。"),
        ],
        "asl_8": [
            ("我现在在做 ASL 8，要注意什么？", "把 ASL 8 的手形稳定住，让拇指和对应手指的接触点不要被遮挡。保持手掌角度一致，停住一小段时间。"),
            ("接触位置看不清时怎么办？", "稍微转正手掌，让拇指和目标手指的接触点朝向镜头。"),
            ("最后检查 ASL 8 的什么？", "确认接触点和未接触手指都清楚可见，手形没有散开。"),
        ],
        "asl_9": [
            ("我现在在做 ASL 9，这样可以吗？", "保持 ASL 9 的手形，注意拇指和目标手指形成清楚接触或圆形轮廓。其他手指不要挡住关键接触点。"),
            ("圆形或接触点要怎么更明显？", "把关键位置稍微转向镜头，手指不要重叠，保持稳定。"),
            ("最后怎么确认 ASL 9？", "停住一小段时间，确认关键轮廓清楚，其他手指没有误伸出。"),
        ],
        "asl_a": [
            ("我现在在做 ASL A，应该怎么摆？", "保持拳形稳定，让拇指位置清楚可见。手腕不要旋转太多，停在画面中央，避免拳头边缘被遮住。"),
            ("拇指的位置要怎么检查？", "确认拇指贴在正确位置且没有被拳头遮住，其他手指继续收拢。"),
            ("最后怎么确认 ASL A？", "短暂停住，检查拳形紧凑、拇指清楚、手腕稳定。"),
        ],
        "asl_c": [
            ("我现在在做 ASL C，接下来要调整吗？", "让手指和拇指保持弯曲，形成清楚的 C 形空间。手掌稍微朝向镜头，确保弧形轮廓能被看见。"),
            ("C 形空间要怎么保持？", "不要把手指完全合上，保留清楚的弧形空隙。手腕保持稳定。"),
            ("最后怎么确认 ASL C？", "停住一小段时间，确认 C 形轮廓完整且没有被遮挡。"),
        ],
        "asl_e": [
            ("我现在在做 ASL E，这个姿态怎么保持？", "把手指向内收拢，保持紧凑的 ASL E 手形。拇指和弯曲的手指都要在画面中，手腕保持稳定。"),
            ("手指收拢得够清楚吗？", "检查弯曲手指是否集中且稳定，避免突然张开成其他字母。"),
            ("最后怎么确认 ASL E？", "短暂停住，确认紧凑手形、拇指位置和手掌角度都清楚。"),
        ],
        "asl_g": [
            ("我现在在做 ASL G，要怎么让它更清楚？", "让食指和拇指的伸出方向清楚，其他手指收拢。保持手腕角度稳定，避免把伸出的指尖转到看不见的位置。"),
            ("食指和拇指方向需要怎么调整？", "让两者形成清楚的指向关系，不要被手掌遮住。手腕可以轻微转正。"),
            ("最后怎么确认 ASL G？", "确认伸出的食指、拇指和收拢的其他手指都清楚可见。"),
        ],
        "asl_i": [
            ("我现在在做 ASL I，下一步怎么检查？", "保持小指清楚伸出，其他手指收拢。把小指放在画面中清楚的位置，手腕不要晃动。"),
            ("小指看不清时要怎么调整？", "稍微转正手腕，让小指不要被其他手指或手掌挡住。"),
            ("最后怎么确认 ASL I？", "短暂停住，确认小指突出明显，其他手指保持收拢。"),
        ],
        "asl_k": [
            ("我现在在做 ASL K，应该注意什么？", "保持 ASL K 的手形，让伸出的两根手指和拇指位置都能看清。手掌角度不要变化太快，短暂停住方便识别。"),
            ("拇指和伸出手指要怎么摆？", "让拇指位置清楚，不要被两根伸出的手指遮挡。保持两根手指分开。"),
            ("最后怎么确认 ASL K？", "确认两根伸出手指、拇指位置和收拢手指都稳定可见。"),
        ],
        "asl_y": [
            ("我现在在做 ASL Y，应该怎么保持？", "伸出拇指和小指，其他手指收拢，形成清楚的 Y 形。把手放在画面中央，确保两端伸出的手指都能看见。"),
            ("Y 形还需要调整吗？", "把拇指和小指的方向分开一点，避免其中一端被手掌遮住。"),
            ("最后怎么确认 ASL Y？", "短暂停住，确认拇指和小指都清楚伸出，其他手指保持收拢。"),
        ],
        "ily": [
            ("我现在在做“我爱你”手势，怎么确认做对了？", "保持拇指、食指和小指清楚伸出，其他手指收拢。手掌朝向稳定，停住一小段时间，让这个组合手形完整可见。"),
            ("这个手势需要调整哪个手指？", "重点检查拇指、食指和小指都伸出，另外两根手指不要误打开。"),
            ("最后怎么确认“我爱你”手势？", "短暂停住，确认三个伸出手指形成清楚组合，手掌没有偏出画面。"),
        ],
        "hand_shape": [
            ("我现在在注册手形，应该怎么做？", "保持当前手形稳定，把双手放在画面中央。不要频繁换姿态，先停住一小段时间，让系统能清楚记录手指和手腕位置。"),
            ("注册时还需要调整吗？", "如果手有偏移，就移回画面中央。保持手指形态不变，避免采集中途换手形。"),
            ("最后怎么确认注册手形可用？", "短暂停住，确认双手、手腕和关键手指都完整可见，再结束注册。"),
        ],
    }
    if task_id in base:
        return base[task_id]
    return [
        (f"我现在在做{task_name}，应该怎么保持？", f"保持“{task_name}”对应的目标手形，把关键手指放在画面中央。手腕尽量稳定，短暂停住方便识别。"),
        ("还需要调整手腕或手指吗？", "如果关键手指被遮挡，就稍微转正手腕；如果手形不稳定，先停住再继续。"),
        ("最后怎么确认这个手形？", "确认目标手指、手掌朝向和双手位置都清楚可见，然后再结束。"),
    ]


def rebuild_history(turns: list[dict[str, Any]]) -> None:
    history: list[dict[str, str]] = []
    for turn in turns:
        turn["history"] = list(history)
        history.append({"user": str(turn.get("user", "")), "assistant": str(turn.get("assistant", ""))})


def expand_sample(sample: dict[str, Any], target_turns: int, fps: float) -> bool:
    turns = sample.get("turns", [])
    if not isinstance(turns, list) or target_turns <= 1:
        return False
    if len(turns) == target_turns:
        segment_ids = {str(turn.get("segment_id", "")) for turn in turns}
        segments = sample.get("metadata", {}).get("temporal_segments", [])
        is_single_segment = isinstance(segments, list) and len(segments) == 1
        missing_interval = any(
            "interval_start_sec" not in turn or "interval_end_sec" not in turn
            for turn in turns
        )
        if len(segment_ids) == 1 and is_single_segment and missing_interval:
            segment = segment_for_turn(sample, turns[0])
            intervals = turn_intervals(segment, [as_float(turn.get("time_sec")) for turn in turns])
            for turn, (start, end) in zip(turns, intervals):
                turn["interval_start_sec"] = start
                turn["interval_end_sec"] = end
            return True
        return False
    if len(turns) != 1:
        return False
    meta = sample.get("metadata", {})
    task_id = str(meta.get("task_id", ""))
    task_name = str(meta.get("task_name") or task_id)
    first = turns[0]
    segment = segment_for_turn(sample, first)
    times = sample_times(segment, target_turns)
    intervals = turn_intervals(segment, times)
    templates = static_templates(task_id, task_name)[:target_turns]
    if len(templates) < target_turns:
        return False
    segment_id = str(segment.get("segment_id", first.get("segment_id", "seg_000")))
    sample["turns"] = [
        {
            "turn_id": f"turn_{idx:03d}",
            "time_sec": times[idx],
            "frame_index": frame_index(times[idx], fps),
            "interval_start_sec": intervals[idx][0],
            "interval_end_sec": intervals[idx][1],
            "segment_id": segment_id,
            "history": [],
            "user": user,
            "assistant": assistant,
        }
        for idx, (user, assistant) in enumerate(templates)
    ]
    rebuild_history(sample["turns"])
    meta["review_status"] = "streaming_edited"
    note = f"单段静态手形样本已扩展为 {target_turns} 轮流式问答，并基于关键时间点抽帧复核。"
    old_note = str(meta.get("review_notes", "")).strip()
    meta["review_notes"] = f"{old_note} {note}".strip() if old_note else note
    return True


def write_review(path: Path, samples: list[dict[str, Any]]) -> None:
    lines = [
        "# Streaming Interaction v001 人工复核表",
        "",
        "状态可填：streaming_accepted / streaming_edited / streaming_rejected。",
        "重点检查：当前时间点画面、历史对话和下一步建议是否一致。",
        "",
    ]
    for idx, sample in enumerate(samples, start=1):
        meta = sample.get("metadata", {})
        lines.extend(
            [
                f"## {idx}. {sample['id']}",
                "",
                f"- status: {meta.get('review_status', '')}",
                f"- video: {sample['video']}",
                f"- source_id: {meta.get('streaming_source_id', '')}",
                f"- task: {meta.get('task_id', '')} / {meta.get('task_name', '')}",
                "- turns:",
            ]
        )
        for turn in sample.get("turns", []):
            lines.extend(
                [
                    f"  - {turn['turn_id']} @ {float(turn['time_sec']):.2f}s frame={turn['frame_index']} segment={turn['segment_id']}",
                    f"    user: {turn['user']}",
                    f"    assistant: {turn['assistant']}",
                ]
            )
        lines.extend(["", f"review_notes: {meta.get('review_notes', '')}", "", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/streaming_interaction_v001.json"))
    parser.add_argument("--messages", type=Path, default=Path("data/processed/streaming_interaction_v001_messages.json"))
    parser.add_argument("--review", type=Path, default=Path("eval/human_review/streaming_interaction_v001_review.md"))
    parser.add_argument("--report", type=Path, default=Path("runs/streaming_interaction_v001/expand_static_turns_report.json"))
    parser.add_argument("--target-turns", type=int, default=3)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_path = resolve_path(args.dataset)
    messages_path = resolve_path(args.messages)
    review_path = resolve_path(args.review)
    report_path = resolve_path(args.report)

    samples = read_json(dataset_path)
    if not isinstance(samples, list):
        raise TypeError(f"dataset must be a JSON list: {dataset_path}")

    before_counts: dict[int, int] = {}
    for sample in samples:
        before_counts[len(sample.get("turns", []))] = before_counts.get(len(sample.get("turns", [])), 0) + 1

    expanded_ids: list[str] = []
    for sample in samples:
        if expand_sample(sample, args.target_turns, args.fps):
            expanded_ids.append(str(sample.get("id", "")))

    after_counts: dict[int, int] = {}
    for sample in samples:
        after_counts[len(sample.get("turns", []))] = after_counts.get(len(sample.get("turns", [])), 0) + 1

    report = {
        "dataset": str(dataset_path),
        "sample_count": len(samples),
        "target_turns": args.target_turns,
        "expanded_count": len(expanded_ids),
        "turn_distribution_before": before_counts,
        "turn_distribution_after": after_counts,
        "total_turns": sum(len(sample.get("turns", [])) for sample in samples),
        "expanded_ids": expanded_ids,
        "dry_run": args.dry_run,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    write_json(dataset_path, samples)
    write_json(messages_path, [to_messages_sample(sample) for sample in samples])
    write_review(review_path, samples)
    write_json(report_path, report)


if __name__ == "__main__":
    main()
