# ShoweeData v001 标注说明

更新时间：2026-07-03 15:00 CST

## 目标

本批数据用于 Challenge 2 的第一轮 ShoweeData LoRA smoke 训练闭环。目标是先提供一批可读、可训练、可复核的小规模样本，而不是最终高质量人工真值。

## 数据来源

- 数据集：ShoweeHandv2
- 视角：`showee_head`
- 原始目录：`<ShoweeHandv2>/raw`
- 项目软链接：`<pipeline_root>/data/raw/ShoweeHandv2`
- 可见候选样本数：506
- 本批选中样本数：20

## 生成方式

使用本地 Qwen3-VL-8B-Instruct 对每条视频均匀抽取 8 帧，结合 `metadata.json` 中的任务名和动作说明，生成视觉标注初稿。

生成脚本：

```bash
<pipeline_root>/scripts/build_showee_ai_dataset_v001.py
```

运行记录：

```bash
<pipeline_root>/runs/liuyichen/showee_ai_label_v001/run.log
<pipeline_root>/runs/liuyichen/showee_ai_label_v001/run_report.json
```

## 输出文件

视频索引：

```bash
<pipeline_root>/data/processed/video_index_showee_v001.csv
```

AI 原始标注：

```bash
<pipeline_root>/data/processed/showee_ai_annotations_raw_v001.json
```

LoRA 训练用数据：

```bash
<pipeline_root>/data/processed/showee_train_v001.json
<pipeline_root>/data/processed/showee_val_v001.json
<pipeline_root>/data/processed/showee_test_v001.json
```

固定划分：

```bash
<pipeline_root>/data/splits/showee_train_v001.json
<pipeline_root>/data/splits/showee_val_v001.json
<pipeline_root>/data/splits/showee_test_v001.json
```

评估集：

```bash
<pipeline_root>/eval/sets/showee_val_v001.json
<pipeline_root>/eval/sets/showee_test_v001.json
```

人工复核表：

```bash
<pipeline_root>/eval/human_review/showee_review_v001.md
```

val/test 快速复核 contact sheets：

```bash
<pipeline_root>/eval/human_review/contact_sheets/
```

## 划分

- train：14 条
- val：3 条
- test：3 条

截至 2026-07-03 14:40 CST，v001 全部 20 条样本已完成初步人工复核并标为 `edited`。当前 v001 可用于训练链路 smoke 和第一轮 LoRA 小实验，但不应作为最终高质量人工真值集；正式报告前如需要更高质量指标，应继续做逐帧或双人交叉复核。

## 数据格式

训练数据是 ShareGPT-like 多轮视频问答格式：

```json
{
  "id": "showee_0006_midair_1_asl_1_head",
  "video": "/path/to/video.mkv",
  "metadata": {
    "dataset": "ShoweeHandv2",
    "source_view": "showee_head",
    "split": "train",
    "annotation_source": "human_review_v001_initial_from_ai",
    "review_status": "edited"
  },
  "conversations": [
    {"from": "user", "value": "<video>\n请描述视频中正在执行的手势任务。"},
    {"from": "assistant", "value": "..."}
  ]
}
```

## 注意事项

1. 这是 AI 初稿，不是人工真值。
2. Qwen3-VL 可能会在手指数量、左右手、细粒度 ASL 姿态上出错。
3. val/test 不应参与训练。
4. 成员 A 可以先用 `showee_train_v001.json` 跑 LoRA 最小闭环；成员 B 后续继续提高 v001 复核质量或扩展 v002。
5. 如果 LoRA 入口要求单一训练文件，优先使用：

```bash
<pipeline_root>/data/processed/showee_train_v001.json
```

## 当前使用建议

- 训练链路 smoke：可以使用 `data/processed/smoke_train.json` 或 `data/processed/showee_train_v001.json`。
- 第一轮 LoRA 小实验：只使用 `showee_train_v001.json` 训练，不使用 val/test。
- baseline/LoRA 对比：使用 `eval/sets/showee_val_v001.json` 和 `eval/sets/showee_test_v001.json`，当前二者已完成初步复核。
- 更高质量复核：优先查看 `eval/human_review/contact_sheets/` 或 `tmp/review_hand_crops_v001/` 中的 contact sheet，对低置信的 ASL 指尖接触关系做逐帧确认。
- 数据扩展：下一版建议从 20 条扩展到 50 条左右，覆盖更多 subject、session type、task 和视角条件。
