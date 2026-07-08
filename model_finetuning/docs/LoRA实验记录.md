# Qwen3-VL ShoweeData LoRA 实验记录

更新时间：2026-07-08 11:47 CST

## 1. 环境变更

已在项目 `.venv` 中安装：

```text
peft==0.19.1
```

仍未安装：

```text
bitsandbytes
trl
deepspeed
flash_attn
```

本轮实验使用 bf16 LoRA，没有使用 QLoRA。

## 2. 已新增文件

训练入口：

```bash
scripts/train_qwen3vl_lora.py
```

推理入口：

```bash
scripts/infer_qwen3vl_lora_showee.py
```

配置：

```bash
configs/lora_showee_smoke.yaml
configs/lora_showee_v001.yaml
configs/lora_showee_v001_8epoch.yaml
configs/lora_showee_v002.yaml
configs/lora_showee_v002_with_choice.yaml
configs/lora_temporal_caption_v001.yaml
configs/lora_wearable_egoconv_v001_warmup.yaml
configs/lora_wearable_egoconv_v001_reviewed.yaml
configs/lora_wearable_egoconv_v001_multiturn.yaml
```

## 3. LoRA target_modules

已用空权重方式确认 Qwen3-VL 语言模型中存在以下模块：

```text
model.language_model.layers.*.self_attn.q_proj
model.language_model.layers.*.self_attn.k_proj
model.language_model.layers.*.self_attn.v_proj
model.language_model.layers.*.self_attn.o_proj
model.language_model.layers.*.mlp.gate_proj
model.language_model.layers.*.mlp.up_proj
model.language_model.layers.*.mlp.down_proj
```

本轮 LoRA 只挂语言侧这些模块，不挂视觉塔。

## 4. smoke LoRA 结果

训练命令：

```bash
cd /data1/shared_data/qwen3vl-showeeData
source .venv/bin/activate
python scripts/train_qwen3vl_lora.py --config configs/lora_showee_smoke.yaml
```

训练配置摘要：

```text
train_data: data/processed/smoke_train.json
num_frames: 4
max_steps: 20
lora_r: 8
lora_alpha: 16
gradient_accumulation_steps: 1
```

结果：

```text
first_loss: 4.2179
final_loss: 0.2697
elapsed_seconds: 84.059
peak_memory_gb: 26.15
trainable_params: 21,823,488
trainable_percent: 0.2483%
```

产物：

```bash
models/lora/showee_smoke_lora_v001/
runs/puzuo/showee_smoke_lora_v001/run_report.json
runs/puzuo/showee_smoke_lora_v001/train_log.json
eval/results/showee_smoke_lora_v001_predictions.jsonl
```

结论：smoke 过拟合链路通过，adapter 可保存和加载。但该 adapter 只见过 1 条样本，推理中仍有幻觉，只能作为链路验证。

## 5. Showee train v001 LoRA 结果

训练命令：

```bash
python scripts/train_qwen3vl_lora.py --config configs/lora_showee_v001.yaml --max-steps 28
```

训练配置摘要：

```text
train_data: data/processed/showee_train_v001.json
sample_count: 14
num_frames: 8
max_steps: 28
optimizer_updates: 7
lora_r: 16
lora_alpha: 32
gradient_accumulation_steps: 4
```

结果：

```text
first_loss: 3.7725
final_loss: 2.9615
elapsed_seconds: 92.881
peak_memory_gb: 26.727
trainable_params: 43,646,976
trainable_percent: 0.4954%
```

产物：

```bash
models/lora/showee_train_v001_lora/
runs/puzuo/showee_train_v001_lora/run_report.json
runs/puzuo/showee_train_v001_lora/train_log.json
```

## 6. val/test 对比

已保存 baseline 与 LoRA 输出：

```bash
eval/results/baseline_qwen3vl_showee_val_v001_predictions.jsonl
eval/results/lora_showee_train_v001_val_predictions.jsonl
eval/results/baseline_qwen3vl_showee_test_v001_predictions.jsonl
eval/results/lora_showee_train_v001_test_predictions.jsonl
eval/results/lora_showee_train_v001_8epoch_val_predictions.jsonl
eval/results/lora_showee_train_v001_8epoch_test_predictions.jsonl
eval/results/showee_v001_lora_comparison_summary.json
```

轻量规则指标：预测文本中是否直接包含 `metadata.task_name`。

```text
baseline val: 0/3
LoRA 28-step val: 0/3
LoRA 112-step val: 0/3
baseline test: 0/3
LoRA 28-step test: 0/3
LoRA 112-step test: 0/3
```

人工观察：

- baseline 常把 ASL G/I/L/Y/K 识别为“指向”“V 字”“摇滚”等通用手势。
- 28 micro-step LoRA 尚未学会在 val/test 上输出正确任务名。
- LoRA test 的部分回答减少了文化解释和设备交互等无关扩展，但仍未解决核心分类错误。
- 112 micro-step LoRA 明显学到了训练答案模板，会输出“视频展示的是 ASL X 手势任务。根据元数据...”这类格式，但在未见 val/test 类别上把任务错套成训练集中见过的 ASL 1/2/8。

## 7. Showee train v001 8 epoch LoRA 结果

训练命令：

```bash
python scripts/train_qwen3vl_lora.py --config configs/lora_showee_v001_8epoch.yaml
```

训练配置摘要：

```text
train_data: data/processed/showee_train_v001.json
sample_count: 14
num_frames: 8
epochs: 8
micro_steps: 112
optimizer_updates: 28
lora_r: 16
lora_alpha: 32
gradient_accumulation_steps: 4
```

结果：

```text
first_loss: 3.7725
final_loss: 1.0118
elapsed_seconds: 368.299
peak_memory_gb: 26.727
trainable_params: 43,646,976
trainable_percent: 0.4954%
```

产物：

```bash
models/lora/showee_train_v001_lora_8epoch/
runs/puzuo/showee_train_v001_lora_8epoch/run_report.json
runs/puzuo/showee_train_v001_lora_8epoch/train_log.json
eval/results/lora_showee_train_v001_8epoch_val_predictions.jsonl
eval/results/lora_showee_train_v001_8epoch_test_predictions.jsonl
```

结论：完整 8 epoch 可以稳定跑完，训练集 loss 明显下降，但 val/test task_name 命中仍为 0/6。当前数据规模下继续加 epoch 会更强地记住训练模板和训练类名，不一定提升未见手势类别识别。

## 8. Showee train v002 LoRA 结果

训练命令：

```bash
python scripts/train_qwen3vl_lora.py --config configs/lora_showee_v002.yaml
```

训练配置摘要：

```text
train_data: data/processed/showee_train_v002.json
sample_count: 40
num_frames: 8
epochs: 4
micro_steps: 160
optimizer_updates: 40
lora_r: 16
lora_alpha: 32
gradient_accumulation_steps: 4
```

结果：

```text
first_loss: 3.7725
final_loss: 0.9514
elapsed_seconds: 518.902
peak_memory_gb: 26.726
trainable_params: 43,646,976
trainable_percent: 0.4954%
```

产物：

```bash
models/lora/showee_train_v002_lora/
runs/puzuo/showee_train_v002_lora/run_report.json
runs/puzuo/showee_train_v002_lora/train_log.json
```

## 9. v002 baseline / LoRA 评估结果

已保存 baseline 与 LoRA 输出：

```bash
eval/results/baseline_qwen3vl_showee_val_v002_open_predictions.jsonl
eval/results/baseline_qwen3vl_showee_test_v002_open_predictions.jsonl
eval/results/baseline_qwen3vl_showee_val_v002_choice_predictions.jsonl
eval/results/baseline_qwen3vl_showee_test_v002_choice_predictions.jsonl
eval/results/lora_showee_train_v002_val_open_predictions.jsonl
eval/results/lora_showee_train_v002_test_open_predictions.jsonl
eval/results/lora_showee_train_v002_val_choice_predictions.jsonl
eval/results/lora_showee_train_v002_test_choice_predictions.jsonl
eval/results/showee_v002_lora_comparison_summary.json
```

指标摘要：

```text
baseline open val: 0/5
baseline open test: 0/5
baseline choice val: 1/5
baseline choice test: 0/5

LoRA v002 open val: 0/5
LoRA v002 open test: 0/5
LoRA v002 choice val: 1/5
LoRA v002 choice test: 0/5
```

人工观察：

- v002 LoRA 明显学到 Showee 标注风格，回答更像“视频展示的是 X 手势任务，关键动作是...”。
- 但任务类别仍严重漂移，ASL G/I/L/Y 之间常互相误判。
- 开放式生成仍会产生训练集中没有的任务名，例如“ASL 10”“手势任务 001”“手指转圈”。
- 候选式 choice 没有明显提升，说明当前 LoRA 只学到描述模板，不足以解决细粒度视觉区分。

## 10. v002_with_choice LoRA 结果

训练命令：

```bash
python scripts/train_qwen3vl_lora.py --config configs/lora_showee_v002_with_choice.yaml
```

训练配置摘要：

```text
train_data: data/processed/showee_train_v002_with_choice.json
sample_count: 120
open_description_samples: 40
choice_samples: 80
num_frames: 8
epochs: 2
micro_steps: 240
optimizer_updates: 60
lora_r: 16
lora_alpha: 32
gradient_accumulation_steps: 4
```

结果：

```text
first_loss: 3.7725
final_loss: 0.0551
elapsed_seconds: 771.904
peak_memory_gb: 26.726
trainable_params: 43,646,976
trainable_percent: 0.4954%
```

产物：

```bash
models/lora/showee_train_v002_with_choice_lora/
runs/puzuo/showee_train_v002_with_choice_lora/run_report.json
runs/puzuo/showee_train_v002_with_choice_lora/train_log.json
```

choice 评估产物：

```bash
eval/results/lora_showee_train_v002_with_choice_val_choice_predictions.jsonl
eval/results/lora_showee_train_v002_with_choice_test_choice_predictions.jsonl
eval/results/lora_showee_train_v002_with_choice_val_choice_predictions.jsonl.metrics.json
eval/results/lora_showee_train_v002_with_choice_test_choice_predictions.jsonl.metrics.json
eval/results/showee_v002_with_choice_lora_comparison_summary.json
```

指标摘要：

```text
baseline choice val: 1/5
baseline choice test: 0/5
LoRA v002 choice val: 1/5
LoRA v002 choice test: 0/5
LoRA v002_with_choice choice val: 0/5
LoRA v002_with_choice choice test: 2/5
```

人工观察：

- v002_with_choice 能稳定输出合法 JSON，`parse_failures=0`。
- test choice 有提升，命中 `finger_wave` 和 `hand_shape`。
- val choice 反而降到 0/5，ASL G/I/K/L/Y 均被预测为 `asl_0`。
- test 中 ASL L/Y 也被预测为 `asl_0`，`seq_flex` 被预测为 `seq_extend`。

结论：choice 监督确实改变了输出格式并改善了部分非 ASL 类候选选择，但没有解决头戴视角下 ASL 细粒度区分。后续优先排查 ASL contact sheets、增加手部局部视觉输入和改进候选负例设计。

## 11. temporal_caption_v001 LoRA 结果

本轮不再使用候选式 choice 问答，改用 annotation pipeline 生成的动作 + 时间段标注数据。

数据：

```text
data/processed/temporal_caption_v001.json: 100 条
data/processed/temporal_caption_v001_train.json: 80 条
data/processed/temporal_caption_v001_val.json: 10 条
data/processed/temporal_caption_v001_test.json: 10 条
```

每条样本为 3 轮 ShareGPT-like 对话：

```text
1. 描述视频中正在执行的手势任务
2. 描述关键手部动作
3. 描述动作从头到尾的变化，并明确时间段
```

注意：本批数据的 `review_status` 为 `temporal_unreviewed`，时间段多为 pipeline 初始标注，不是逐帧人工精标。

训练命令：

```bash
python scripts/train_qwen3vl_lora.py --config configs/lora_temporal_caption_v001.yaml
```

训练配置摘要：

```text
train_data: data/processed/temporal_caption_v001_train.json
sample_count: 80
num_frames: 16
epochs: 3
micro_steps: 240
optimizer_updates: 60
lora_r: 16
lora_alpha: 32
gradient_accumulation_steps: 4
```

训练结果：

```text
first_loss: 3.0971
final_loss: 0.2588
elapsed_seconds: 1034.064
peak_memory_gb: 26.970
trainable_params: 43,646,976
trainable_percent: 0.4954%
```

产物：

```bash
models/lora/temporal_caption_v001_lora/
runs/puzuo/temporal_caption_v001_lora/run_report.json
runs/puzuo/temporal_caption_v001_lora/train_log.json
```

新增评估脚本与评估集：

```bash
scripts/build_temporal_caption_eval_sets.py
scripts/summarize_temporal_caption_eval.py
eval/sets/temporal_caption_v001_val_temporal.json
eval/sets/temporal_caption_v001_test_temporal.json
```

评估产物：

```bash
eval/results/lora_temporal_caption_v001_val_temporal_predictions.jsonl
eval/results/lora_temporal_caption_v001_test_temporal_predictions.jsonl
eval/results/lora_temporal_caption_v001_val_temporal_predictions.jsonl.metrics.json
eval/results/lora_temporal_caption_v001_test_temporal_predictions.jsonl.metrics.json
eval/results/temporal_caption_v001_lora_summary.json
```

轻量指标摘要：

```text
val temporal:
  task_name/task_id hit: 0/10
  all expected time spans hit: 10/10
  all segment labels hit: 0/10

test temporal:
  task_name/task_id hit: 0/10
  all expected time spans hit: 0/10
  all segment labels hit: 0/10
```

人工观察：

- 模型学会了 temporal caption 的回答风格，通常会输出时间段和“精确动作边界需要结合视频逐帧复核”这类标注提示。
- val 集全是单段 `0.00-20.00s` 粗粒度样本，因此时间字符串命中 10/10，但动作名和段标签没有命中。
- test 集全是 3 段样本，模型大多退化成单段 `0.00-20.00s` 描述，没有泛化出 `0.00-5.00s / 5.00-15.00s / 15.00-20.00s` 或 `0.00-7.50s / 7.50-22.50s / 22.50-30.00s` 这类三段边界。
- 有个别输出出现重复短语，推理侧后续应加 `repetition_penalty` 或降低 `max_new_tokens`。

结论：训练链路已跑通，LoRA 能拟合动作+时间段标注格式；但当前数据分布和标注方式导致模型更容易学习“单段模板”，没有稳定学会根据视频/时长生成多段动作边界。下一步应调整训练数据构成，让 train/val/test 都覆盖单段和三段样本，并把时间段输出改成更结构化的 JSON 或固定表格格式。

## 12. wearable_egoconv_v001 warm-up LoRA 结果

本轮使用新的标准范式数据：

```bash
data/processed/wearable_egoconv_v001.jsonl
```

该文件为 JSONL，共 100 条。字段包含相对视频路径、视频时长、阶段区间、问题、答案、任务名和 `dialog` 多轮对话。

转换脚本：

```bash
scripts/convert_wearable_egoconv_v001.py
```

转换产物：

```bash
data/processed/wearable_egoconv_v001_train.json
runs/puzuo/wearable_egoconv_v001_build/build_report.json
```

转换结果：

```text
sample_count: 100
single_turn_samples: 64
multi_turn_samples: 36
task_count: 40
review_status: coarse_warmup
```

训练命令：

```bash
python scripts/train_qwen3vl_lora.py --config configs/lora_wearable_egoconv_v001_warmup.yaml
```

训练配置摘要：

```text
train_data: data/processed/wearable_egoconv_v001_train.json
sample_count: 100
num_frames: 8
max_steps: 100
optimizer_updates: 25
lora_r: 16
lora_alpha: 32
gradient_accumulation_steps: 4
```

训练结果：

```text
first_loss: 2.2497
final_loss: 0.7418
elapsed_seconds: 392.341
peak_memory_gb: 27.056
trainable_params: 43,646,976
trainable_percent: 0.4954%
```

产物：

```bash
models/lora/wearable_egoconv_v001_warmup_lora/
runs/puzuo/wearable_egoconv_v001_warmup_lora/run_report.json
runs/puzuo/wearable_egoconv_v001_warmup_lora/train_log.json
eval/results/wearable_egoconv_v001_warmup_summary.json
```

人工观察：

- 训练过程稳定，100 step warm-up 完整跑完。
- 前段三轮样本 loss 从约 2.2 降到后段约 0.8-1.0。
- 单轮 ASL 样本前段 loss 较高，后段也降到约 1.0-1.8。
- 这轮只用于适配新的 wearable EgoConv 对话范式，不作为充分收敛或最终效果评估。

结论：新标准范式数据已经能被当前 LoRA 训练链路读取并完成 warm-up。由于标注仍是粗标注，本轮不建议继续加 epoch；下一步应先做少量人工复核或固定验证集，再决定是否扩大训练。

## 13. wearable_egoconv_v001 reviewed LoRA 结果

本轮继续使用同一个标准范式 JSONL，但文件已经过复核调整：

```bash
data/processed/wearable_egoconv_v001.jsonl
```

输入文件修改时间为 2026-07-08 09:42 CST。复核后答案更短、更直接，目标是 warm-up 模型回答“动作 + 对应时间段”，不再使用候选式 choice 问答。

转换命令：

```bash
python scripts/convert_wearable_egoconv_v001.py \
  --output data/processed/wearable_egoconv_v001_reviewed_train.json \
  --report runs/puzuo/wearable_egoconv_v001_reviewed_build/build_report.json \
  --review-status reviewed_warmup
```

转换产物：

```bash
data/processed/wearable_egoconv_v001_reviewed_train.json
runs/puzuo/wearable_egoconv_v001_reviewed_build/build_report.json
```

转换结果：

```text
sample_count: 100
single_turn_samples: 64
multi_turn_samples: 36
task_count: 40
review_status: reviewed_warmup
```

训练命令：

```bash
python scripts/train_qwen3vl_lora.py --config configs/lora_wearable_egoconv_v001_reviewed.yaml
```

训练配置摘要：

```text
train_data: data/processed/wearable_egoconv_v001_reviewed_train.json
sample_count: 100
num_frames: 8
max_steps: 200
epochs: 2
optimizer_updates: 50
lora_r: 16
lora_alpha: 32
gradient_accumulation_steps: 4
save_every: 100
```

训练结果：

```text
first_loss: 3.3761
final_loss: 1.3290
first_20_avg: 4.2561
steps_81_100_avg: 1.9747
steps_101_120_avg: 1.8957
last_20_avg: 0.9931
elapsed_seconds: 913.228
peak_memory_gb: 26.611
trainable_params: 43,646,976
trainable_percent: 0.4954%
```

产物：

```bash
models/lora/wearable_egoconv_v001_reviewed_lora/
runs/puzuo/wearable_egoconv_v001_reviewed_lora/run_report.json
runs/puzuo/wearable_egoconv_v001_reviewed_lora/train_log.json
eval/results/wearable_egoconv_v001_reviewed_lora_summary.json
```

人工观察：

- 训练过程稳定，step-100 和 step-200 checkpoint 均已保存，最终 adapter 已落盘。
- 第二个 epoch 仍有明显同集拟合收益：前 20 step 平均 loss 4.2561，最后 20 step 平均 loss 0.9931。
- 相比粗标注 warm-up，复核版回答更短，监督 token 和答案分布不同，因此 first loss 不能直接按数值高低比较。
- 当前 100 条全部用于 warm-up，没有固定 held-out eval，因此这轮只能说明模型已适配复核后的动作 + 时间段回答范式，不能作为泛化效果结论。

结论：复核版 wearable_egoconv_v001 LoRA 已完成 2 epoch warm-up，模型能拟合新标注范式。下一步应固定 10-20 条复核验证集并跑推理评估，再决定是否继续扩大数据或增加 epoch。

## 14. wearable_egoconv_v001 reviewed val20 评估

用户继续补充并复核了 20 条样本后，`data/processed/wearable_egoconv_v001.jsonl` 从 100 条扩展到 120 条。本轮将新增的 line 101-120 作为 held-out 验证集，未参与上一轮 reviewed LoRA 训练。

新增脚本：

```bash
scripts/build_wearable_egoconv_eval_set.py
scripts/infer_qwen3vl_lora_wearable.py
scripts/summarize_wearable_egoconv_eval.py
```

构建验证集：

```bash
python scripts/build_wearable_egoconv_eval_set.py \
  --input data/processed/wearable_egoconv_v001.jsonl \
  --output eval/sets/wearable_egoconv_v001_reviewed_val20.json \
  --start-line 101 \
  --end-line 120
```

验证集构成：

```text
source_rows: 20
eval_turns: 22
line_range: 101-120
```

其中 line 117 是三轮多轮样本，评估脚本会带上前序用户/assistant 历史分别生成 turn1/turn2/turn3。

推理命令：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/infer_qwen3vl_lora_wearable.py \
  --eval-set eval/sets/wearable_egoconv_v001_reviewed_val20.json \
  --adapter models/lora/wearable_egoconv_v001_reviewed_lora \
  --output eval/results/lora_wearable_egoconv_v001_reviewed_val20_predictions.jsonl \
  --num-frames 8 \
  --max-new-tokens 160
```

产物：

```bash
eval/sets/wearable_egoconv_v001_reviewed_val20.json
eval/results/lora_wearable_egoconv_v001_reviewed_val20_predictions.jsonl
eval/results/lora_wearable_egoconv_v001_reviewed_val20_predictions.jsonl.summary.json
eval/results/lora_wearable_egoconv_v001_reviewed_val20_predictions.jsonl.metrics.json
```

自动指标：

```text
count: 22
non_empty_rate: 1.0000
refusal_like_rate: 0.0000
task_hit_rate: 0.5000
avg_rouge_l_f1: 0.4201
avg_rouge_l_recall: 0.4738
interval_hit_rate: 0.0000
elapsed_seconds: 96.745
```

人工观察：

- 模型能稳定输出中文动作指导，没有空答或拒答。
- 输出风格已经贴近 reviewed warm-up 数据，通常包含“保持手形稳定、手腕不要晃动、停住一小段时间、方便识别”等指导。
- 多轮 ASL L 的第 2/3 轮能利用历史继续回答，说明多轮上下文链路可用。
- ASL 细粒度动作仍明显模板化：ASL 1/2/4/5/A/C/G/I/Y 等多次被泛化成“拇指和对应手指接触点清楚可见”，没有命中参考答案中的食指、四指、五指、小指、C 形、Y 形等关键动作细节。
- ASL L 第 1 轮出现明显错误，把 L 形说成“伸出拇指和小指”，更接近 Y 形。
- `interval_hit_rate=0` 的直接原因是当前训练监督的 assistant 答案没有显式包含时间戳，时间段只在 `video_intervals` / `target_interval` 元数据中。若目标是让模型答出精确“动作 + 对应时间段”，下一版训练样本需要把时间段写进 assistant 输出，例如固定为 JSON 或表格。

结论：reviewed LoRA 已学到回答格式和通用动作指导，但还没有学会可靠的细粒度 ASL 手形区分，也没有被监督输出显式时间段。下一步不应继续直接加 epoch，应先把训练目标改为结构化动作 + 时间段，并把新增 20 条作为固定验证集保留。

## 15. wearable_egoconv_v001 multiturn LoRA 结果

随后 `data/processed/wearable_egoconv_v001.jsonl` 被统一调整为全多轮对话格式：

```text
total_rows: 120
dialog_len: 6 for all rows
questions_per_row: 3
answers_per_row: 3
```

本轮继续保留 line 101-120 作为 held-out 验证集，只用 line 1-100 训练。

转换训练集：

```bash
python scripts/convert_wearable_egoconv_v001.py \
  --input data/processed/wearable_egoconv_v001.jsonl \
  --output data/processed/wearable_egoconv_v001_multiturn_train.json \
  --report runs/puzuo/wearable_egoconv_v001_multiturn_build/build_report.json \
  --review-status reviewed_multiturn_train \
  --start-line 1 \
  --end-line 100
```

训练数据：

```text
sample_count: 100
source_line_range: 1-100
single_turn_samples: 0
multi_turn_samples: 100
```

验证集：

```bash
python scripts/build_wearable_egoconv_eval_set.py \
  --input data/processed/wearable_egoconv_v001.jsonl \
  --output eval/sets/wearable_egoconv_v001_multiturn_val20.json \
  --start-line 101 \
  --end-line 120
```

验证集构成：

```text
source_rows: 20
eval_turns: 60
line_range: 101-120
```

训练命令：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qwen3vl_lora.py \
  --config configs/lora_wearable_egoconv_v001_multiturn.yaml
```

训练配置摘要：

```text
train_data: data/processed/wearable_egoconv_v001_multiturn_train.json
sample_count: 100
num_frames: 8
max_steps: 200
epochs: 2
optimizer_updates: 50
lora_r: 16
lora_alpha: 32
gradient_accumulation_steps: 4
```

训练结果：

```text
first_loss: 3.3761
final_loss: 1.2607
first_20_avg: 4.3235
steps_81_100_avg: 2.0641
steps_101_120_avg: 1.8497
last_20_avg: 0.8215
elapsed_seconds: 648.570
peak_memory_gb: 26.611
trainable_params: 43,646,976
trainable_percent: 0.4954%
```

产物：

```bash
models/lora/wearable_egoconv_v001_multiturn_lora/
runs/puzuo/wearable_egoconv_v001_multiturn_lora/run_report.json
runs/puzuo/wearable_egoconv_v001_multiturn_lora/train_log.json
eval/results/wearable_egoconv_v001_multiturn_lora_summary.json
```

验证命令：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_qwen3vl_lora_wearable.py \
  --eval-set eval/sets/wearable_egoconv_v001_multiturn_val20.json \
  --adapter models/lora/wearable_egoconv_v001_multiturn_lora \
  --output eval/results/lora_wearable_egoconv_v001_multiturn_val20_predictions.jsonl \
  --num-frames 8 \
  --max-new-tokens 160
```

验证产物：

```bash
eval/sets/wearable_egoconv_v001_multiturn_val20.json
eval/results/lora_wearable_egoconv_v001_multiturn_val20_predictions.jsonl
eval/results/lora_wearable_egoconv_v001_multiturn_val20_predictions.jsonl.summary.json
eval/results/lora_wearable_egoconv_v001_multiturn_val20_predictions.jsonl.metrics.json
```

自动指标：

```text
count: 60
non_empty_rate: 1.0000
refusal_like_rate: 0.0000
task_hits: 18
task_hit_rate: 0.3000
first_turn_task_hits: 18/20
avg_rouge_l_f1: 0.4608
avg_rouge_l_recall: 0.5121
turn1_avg_rouge_l_f1: 0.3998
turn2_avg_rouge_l_f1: 0.4200
turn3_avg_rouge_l_f1: 0.5627
interval_hit_rate: 0.0000
elapsed_seconds: 221.740
```

人工观察：

- 全多轮训练显著改善了第 2/3 轮的上下文跟随，模型会根据前序问答继续补充“稍微转正”“短暂停住”“确认某个关键手指”等细节。
- ASL L 上一轮首问误说成“拇指和小指”，本轮已修正为“伸出食指和拇指形成 L 形”。
- ASL 2/3/4/5/Y/我爱你等样本在第 2/3 轮能更稳定命中“两根/三根/四根/五根手指”“拇指和小指”“拇指、食指和小指”等细节。
- 首轮仍有少量模板残留，例如 ASL 1/I 仍会出现“拇指和对应手指”的泛化说法，但后续轮次能部分纠正到食指或小指。
- `task_hit_rate=0.3` 不宜直接理解为变差，因为第 2/3 轮参考答案和模型输出通常不重复任务名；只看首轮是 18/20，高于上一轮首轮 11/20。
- `interval_hit_rate=0` 仍未改善，原因不变：assistant 监督答案没有显式时间戳。

结论：全多轮重训是有效的，主要收益在多轮上下文和 ASL 细节纠偏；但若目标仍是“答出动作 + 对应时间段”，下一步必须把时间段显式写入 assistant 监督，而不能只放在 metadata。

## 16. 当前结论

本轮已完成 LoRA 训练闭环：依赖安装、target_modules 确认、训练脚本、adapter 保存、adapter 加载推理、固定 val/test baseline 与 LoRA 对比均已跑通。

效果层面，v002 扩展到 50 条并让 train 覆盖 val/test 同类 task_id 后，开放式 LoRA 训练 loss 可以降到 0.9514；加入 choice 监督后，混合训练 loss 进一步降到 0.0551，choice test 从 0/5 提升到 2/5，但 choice val 降到 0/5，ASL 细粒度任务仍集中误判为 `asl_0`。temporal_caption_v001 改为动作 + 时间段监督后，训练 loss 降到 0.2588，但评估显示模型主要学到了回答格式，尚未稳定学会多段时间边界和具体动作名。wearable_egoconv_v001 标准范式数据已完成粗标注 100 step warm-up、复核版 200 step warm-up、全多轮 200 step 重训和 val20 验证；全多轮版 loss 3.3761 -> 1.2607，60-turn val20 平均 ROUGE-L F1 为 0.4608，首轮任务命中 18/20，多轮上下文明显改善。下一步不要继续简单加 epoch，应先把输出目标改成结构化动作 + 时间段，并固定验证集指标。

## 17. v002 choice 训练扩展准备

已为每条 v002 train 视频额外生成 2 条候选任务名选择训练样本：

```text
source_train_count: 40
choice_samples: 80
choice_samples_per_video: 2
mixed_train_count: 120
variants: hard_same_family, mixed_family
```

产物：

```bash
scripts/build_showee_choice_train_v002.py
data/processed/showee_train_v002_choice.json
data/processed/showee_train_v002_choice_messages.json
data/processed/showee_train_v002_with_choice.json
configs/lora_showee_v002_with_choice.yaml
runs/puzuo/showee_choice_train_v002/build_report.json
```

`showee_train_v002_choice.json` 使用训练脚本当前支持的 ShareGPT-like `conversations` 格式；`showee_train_v002_choice_messages.json` 使用 `messages` 格式，便于按候选任务名选择任务人工查看。本轮训练命令：

```bash
python scripts/train_qwen3vl_lora.py --config configs/lora_showee_v002_with_choice.yaml
```

生成后已做轻量校验：80 条 choice 样本均包含正确 task_id，assistant 输出 JSON 与样本元数据一致；抽样通过 `scripts/train_qwen3vl_lora.py` 的 assistant token 对齐检查。
