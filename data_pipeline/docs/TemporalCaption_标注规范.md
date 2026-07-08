# Temporal Caption 标注规范

## 目标

Temporal Caption 数据用于训练和评估模型在描述手势动作时显式给出时间范围的能力。

该 pipeline 不绑定 `v001` 或 `v002` 实验批次。历史标注文件只作为可选继承来源；没有历史标注的样本从 ShoweeHandv2 `raw` 的 `metadata.json` 生成初始标注。

默认只继承历史样本中的人工确认时间段。如果历史样本的 `temporal_segments` 只是 `metadata_duration` 或 `uniform_phase_seed` 这类初始边界，pipeline 会重新按当前配置生成时间段；已有的任务描述和多轮对话仍可继承。时间段终点优先使用真实视频时长和 metadata 计划时长中的较小值，避免标注超出视频尾帧。

## 样本格式

保留原有 ShareGPT-like 多轮格式：

```json
{
  "id": "...",
  "video": "...",
  "metadata": {
    "task_id": "seq_flex",
    "task_name": "序列收回",
    "task_desc": "从张开姿态开始，依次收回并伸出手指，转动手腕",
    "review_status": "temporal_unreviewed",
    "temporal_segments": [
      {
        "segment_id": "seg_000",
        "start_sec": 0.0,
        "end_sec": 7.5,
        "label": "序列收回-起始姿态",
        "description": "进入序列收回任务的起始姿态，开始出现手指伸展或张开、按序连续变化。",
        "boundary_source": "uniform_phase_seed",
        "confidence": "medium",
        "notes": "阶段边界为规则生成，需要人工复核。"
      }
    ]
  },
  "conversations": [
    {
      "from": "user",
      "value": "<video>\n请描述视频中正在执行的手势任务。"
    },
    {
      "from": "assistant",
      "value": "视频展示的是序列收回手势任务..."
    },
    {
      "from": "user",
      "value": "动作从头到尾有什么变化？请明确时间段。"
    },
    {
      "from": "assistant",
      "value": "0.00-7.50s：... 7.50-22.50s：..."
    }
  ]
}
```

## 时间段字段

- `start_sec`：动作段起始时间，单位秒。
- `end_sec`：动作段结束时间，单位秒。
- `label`：该时间段的短标签。
- `description`：该时间段的自然语言描述，用于训练回答。
- `boundary_source`：边界来源。
- `confidence`：边界置信等级。
- `notes`：复核说明。

## 边界来源

- `metadata_duration`：只用 metadata 的任务时长生成整段级标注。
- `uniform_phase_seed`：按规则把视频粗分为起始、主要动作、结束/重复三个阶段。
- `model_proposed`：模型提出的时间边界，尚未人工确认。
- `human_review`：人工看视频后修正过的边界。
- `frame_verified`：逐帧或工具辅助确认过的精确边界。

## 置信等级

- `coarse`：整段级或非常粗的时间范围，只能说明该视频总体在做什么。
- `medium`：阶段级时间范围，可以用于训练模型输出时间段，但不能当作精确动作边界。
- `verified`：人工复核后的可信时间边界。

## 复核要求

人工复核时优先检查：

1. `task_id/task_name` 是否与视频一致。
2. 每个时间段的起止是否明显错误。
3. `description` 是否描述了该段可见动作，而不是只重复任务名。
4. 对头戴视角下不可见的指尖接触、左右手细节，不要过度细化。
5. 如果需要精确动作边界，应把 `boundary_source` 更新为 `human_review` 或 `frame_verified`，并把 `confidence` 更新为 `verified`。

## 回写格式

人工复核时优先只修改 `eval/human_review/temporal_caption_v001_review.md`。需要更新 JSON 时，在对应样本段落下添加 `review_update:` YAML 块：

```yaml
review_update:
  status: frame_verified
  review_notes: 人工逐帧复核通过。
  temporal_segments:
    - segment_id: seg_000
      start_sec: 0.0
      end_sec: 5.2
      label: 起始张开
      description: 手掌张开，五指基本伸展。
      boundary_source: human_review
      confidence: verified
      notes: 人工看视频确认。
    - segment_id: seg_001
      start_sec: 5.2
      end_sec: 12.4
      label: 主要动作
      description: 手指依次收回并再次伸展。
      boundary_source: human_review
      confidence: verified
      notes: 人工看视频确认。
```

先试跑：

```bash
python scripts/apply_temporal_review.py --dry-run
```

确认 `applied_updates` 数量正确后正式回写：

```bash
python scripts/apply_temporal_review.py
```

脚本会同步更新：

- `data/processed/temporal_caption_v001.json`
- `data/processed/temporal_caption_v001_train.json`
- `data/processed/temporal_caption_v001_val.json`
- `data/processed/temporal_caption_v001_test.json`
- `data/splits/temporal_caption_v001_train.json`
- `data/splits/temporal_caption_v001_val.json`
- `data/splits/temporal_caption_v001_test.json`
- `data/processed/temporal_caption_v001_index.csv`

## 第一版建议

`temporal_caption_v001` 建议先生成 100 条：

- train：80
- val：10
- test：10

优先覆盖动作变化明显的任务：

- `seq_flex`
- `seq_extend`
- `seq_pinch`
- `finger_wave`
- `fist_open`
- `abduct_adduct`
- `thumb_swipe`
- `thumb_touchpad`
- `wrist`
- `free_style`

第一版可以使用 `coarse` 和 `uniform_phase_seed`，但正式报告前应抽样人工复核，并把关键验证集提升为 `verified`。
