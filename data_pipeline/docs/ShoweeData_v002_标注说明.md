# ShoweeData v002 标注说明

更新时间：2026-07-03 17:24 CST

## 目标

v002 用于解决 v001 训练集覆盖不足的问题：v001 的 val/test 包含 `asl_g/asl_i/asl_k/asl_l/asl_y/finger_wave`，但这些 task_id 不在 train 中，导致 LoRA 训练后仍无法命中未见任务名。

v002 将数据扩展到 50 条，并显式保证 val/test 中的任务类型在 train 中有同类覆盖。同时将开放式描述评估和候选任务名选择评估拆开，避免把“生成式描述能力”和“给定候选分类能力”混成一个指标。

## 数据规模

- train：40 条
- val：5 条
- test：5 条
- total：50 条

train 覆盖 40 个不同 task_id。val/test 涉及的任务：

```text
asl_g, asl_i, asl_k, asl_l, asl_y, finger_wave, hand_shape, seq_flex
```

这些 task_id 均已在 train 中出现。

## 标注来源

v002 由两部分组成：

- 20 条 v001 已初步人工复核样本，`review_status=edited`。
- 30 条 metadata-seeded 新增样本，已基于 contact sheet 完成初步视觉复核并更新为 `review_status=edited`。

新增样本的任务名、任务说明和视频路径来自 ShoweeHandv2 的 `metadata.json`。文本答案已结合 contact sheet 做初步改写，强调抽帧可见动作和不确定边界。细粒度指尖接触关系、顺序动作的逐帧先后仍需后续高质量复核。

## 输出文件

训练/划分：

```bash
data/processed/showee_train_v002.json
data/processed/showee_val_v002.json
data/processed/showee_test_v002.json
data/splits/showee_train_v002.json
data/splits/showee_val_v002.json
data/splits/showee_test_v002.json
```

候选式 choice 训练扩展：

```bash
data/processed/showee_train_v002_choice.json
data/processed/showee_train_v002_choice_messages.json
data/processed/showee_train_v002_with_choice.json
runs/puzuo/showee_choice_train_v002/build_report.json
```

其中 `showee_train_v002_choice.json` 是训练脚本可直接读取的 ShareGPT-like 格式；`showee_train_v002_choice_messages.json` 保留 `messages` 格式，便于人工查看和后续适配其他训练入口；`showee_train_v002_with_choice.json` 将原 40 条开放式训练样本和 80 条 choice 训练样本合并，共 120 条。

评估集分为两类：

```bash
eval/sets/showee_val_v002_open.json
eval/sets/showee_test_v002_open.json
eval/sets/showee_val_v002_choice.json
eval/sets/showee_test_v002_choice.json
```

复核表和索引：

```bash
eval/human_review/showee_review_v002.md
eval/human_review/contact_sheets_v002/
data/processed/video_index_showee_v002.csv
runs/puzuo/showee_dataset_v002/build_report.json
```

## 评估方式

开放式描述评估：

```bash
python scripts/infer_qwen3vl_lora_showee.py \
  --eval-set eval/sets/showee_val_v002_open.json \
  --output eval/results/baseline_qwen3vl_showee_val_v002_open_predictions.jsonl
```

候选任务名选择评估：

```bash
python scripts/infer_qwen3vl_lora_showee.py \
  --eval-set eval/sets/showee_val_v002_choice.json \
  --output eval/results/baseline_qwen3vl_showee_val_v002_choice_predictions.jsonl \
  --max-new-tokens 128
```

评估输出汇总：

```bash
python scripts/summarize_showee_eval.py eval/results/baseline_qwen3vl_showee_val_v002_open_predictions.jsonl
python scripts/summarize_showee_eval.py eval/results/baseline_qwen3vl_showee_val_v002_choice_predictions.jsonl
```

## 训练入口

v002 LoRA 配置：

```bash
configs/lora_showee_v002.yaml
```

加入候选式 choice 监督后的 LoRA 配置：

```bash
configs/lora_showee_v002_with_choice.yaml
```

启动命令：

```bash
python scripts/train_qwen3vl_lora.py --config configs/lora_showee_v002.yaml
python scripts/train_qwen3vl_lora.py --config configs/lora_showee_v002_with_choice.yaml
```

## 注意事项

1. v002 是训练覆盖修正版，不是最终高质量人工真值集。
2. 新增 30 条样本已完成 contact-sheet 初步视觉复核；如需最终真值，应继续做逐帧或双人交叉复核，尤其是 ASL 指尖接触关系、顺序手指动作和自由手势。
3. val/test 不参与训练；v002 train 已包含相同 task_id 的其他样本，目的是验证同类泛化而不是未见类别泛化。
4. 开放式描述指标和候选式选择指标必须分开报告。
5. choice 训练样本只从 train 生成，不引入 val/test 视频；每条训练视频生成 2 条候选选择监督，分别覆盖同族 hard candidates 和跨族 mixed candidates。
