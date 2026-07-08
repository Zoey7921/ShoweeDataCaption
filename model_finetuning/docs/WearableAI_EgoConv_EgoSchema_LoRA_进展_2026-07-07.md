# Wearable AI Challenge 2: EgoConv + EgoSchema LoRA Progress

日期：2026-07-07

## 目标

当前任务从 ShoweeHandv2 标注线切到 Wearable AI Challenge 2 / EgoConv：

- 使用官方 700 个 EgoConv val 视频调试 Qwen3-VL 推理链路。
- 引入外部 egocentric video QA 数据 EgoSchema 做 LoRA 微调。
- 在 EgoConv 50 条上对比 baseline 与 LoRA，判断是否值得跑 700。

## 官方 EgoConv Baseline

已完成 50 条完整多轮 baseline：

- 预测文件：`eval/results/baseline_qwen3vl_challenge2_egoconv_50_predictions.jsonl`
- Golden：`runs/liuyichen/challenge2_egoconv_50/golden.jsonl`
- 评估结果：`eval/results/baseline_qwen3vl_challenge2_egoconv_50_results_summary.json`

结果：

```text
total_conversations: 50
total_turns: 344
BLEU: 0.062
```

错误分析文件：

```text
runs/liuyichen/challenge2_egoconv_50/error_analysis.md
```

主要问题：

- 原始 Qwen3-VL 经常回答“视频里看不出来”，拒答偏多。
- 细粒度场景识别不稳定，例如 Sudoku 场景被看成 surfing。
- 多轮指代如 `this / here / that` 不稳定。
- 商品、地点、运动指导类问题需要结合视觉和常识，baseline 策略偏保守。

## EgoSchema 数据处理

EgoSchema 官方公开问题数为 5031 条，其中 `subset_answers.json` 有 500 条公开答案，因此本轮只使用这 500 条做监督微调。

路径：

```text
/data1/shared_data/egocentric_datasets/egoschema
```

下载后发现 Kaggle CLI 单文件下载得到的外层文件名是 `.mp4`，但实际内容是 ZIP 容器。已新增并运行解包脚本：

```text
scripts/unpack_kaggle_single_file_mp4s.py
```

解包结果：

```text
total_mp4_files: 501
unpacked: 501
```

随后重新生成 EgoSchema 训练数据：

```text
scripts/build_egoschema_dataset_v001.py
```

输出：

```text
data/processed/egoschema_v001_train.json
data/processed/egoschema_v001_val.json
eval/sets/egoschema_v001_val_choice.json
data/processed/egoschema_v001_build_report.json
```

完整可用样本：

```text
total_public_answers: 500
usable_samples: 500
train_count: 450
val_count: 50
missing_samples: 0
```

已进行 decord 可解码清洗：

```text
scripts/filter_egoschema_decodable.py
```

clean 输出：

```text
data/processed/egoschema_v001_train_clean.json
data/processed/egoschema_v001_val_clean.json
eval/sets/egoschema_v001_val_choice_clean.json
data/processed/egoschema_v001_decodable_report.json
```

清洗结果：

```text
train_clean: 450
val_clean: 50
bad_count: 0
```

## EgoSchema LoRA 训练

配置：

```text
configs/lora_egoschema_v001.yaml
```

训练数据：

```text
data/processed/egoschema_v001_train_clean.json
```

训练结果：

```text
sample_count: 450
target_steps: 450
updates: 113
elapsed_seconds: 477.49
peak_memory_gb: 19.496
trainable_params: 43,646,976
trainable_percent: 0.4954
first_loss: 0.5593655109
final_loss: 0.0005085656
```

Adapter 输出：

```text
models/lora/egoschema_v001_lora
```

50-step smoke 也已成功：

```text
models/lora/egoschema_v001_lora_smoke
runs/liuyichen/egoschema_v001_lora_smoke/run_report.json
```

## EgoConv LoRA 回测

已修改 EgoConv 推理脚本，新增 `--adapter` 参数：

```text
scripts/qwen3vl_egoconv_smoke.py
```

用 EgoSchema LoRA 回测 EgoConv 50 条：

```text
tmux: egoconv_50_egoschema_lora
adapter: models/lora/egoschema_v001_lora
output: eval/results/egoschema_lora_v001_challenge2_egoconv_50_predictions.jsonl
summary: eval/results/egoschema_lora_v001_challenge2_egoconv_50_results_summary.json
```

结果：

```text
total_conversations: 50
total_turns: 344
BLEU: 0.0686
```

对比：

```text
Baseline BLEU: 0.0620
EgoSchema LoRA BLEU: 0.0686
Absolute gain: +0.0066
Relative gain: about +10.6%
```

对比分析文件：

```text
runs/liuyichen/challenge2_egoconv_50_egoschema_lora/baseline_vs_lora_error_analysis.md
runs/liuyichen/challenge2_egoconv_50_egoschema_lora/baseline_vs_lora_error_analysis.json
```

观察：

- LoRA 对运动指导、常识问答、减少拒答有一定帮助。
- 仍然不是大幅提升；BLEU 只小幅增长。
- 部分样本有退化，说明 EgoSchema 选择题训练可能增强了回答意愿，但没有完全解决 EgoConv 多轮指代和细粒度视觉问题。

## Prompt 调整实验

已给 EgoConv 推理脚本增加可选参数：

```text
scripts/qwen3vl_egoconv_smoke.py --prompt-style {default,less_refusal}
```

`default` 保持原始行为；`less_refusal` 的目标是减少无意义拒答，并要求模型在有合理视觉证据时给出最可能答案，同时更明确地处理多轮指代。

为了快速判断方向，已截取同样前 10 条样本重评：

```text
Baseline, 10 samples:               BLEU 0.0625
EgoSchema LoRA default, 10 samples:  BLEU 0.0807
EgoSchema LoRA less_refusal, 10 samples: BLEU 0.0872
```

对应文件：

```text
eval/results/baseline_qwen3vl_challenge2_egoconv_10_results_summary.json
eval/results/egoschema_lora_v001_challenge2_egoconv_10_results_summary.json
eval/results/egoschema_lora_v001_challenge2_egoconv_10_less_refusal_results_summary.json
```

初步结论：

- 前 10 条里，LoRA 本身明显优于 baseline。
- `less_refusal` prompt 在前 10 条里继续小幅提升。
- 这个结论还需要 50 条完整回测确认，不能只看 10 条就直接跑 700。

50 条完整回测已完成：

```text
EgoSchema LoRA default, 50 samples:      BLEU 0.0686
EgoSchema LoRA less_refusal, 50 samples: BLEU 0.0743
```

`less_refusal` 比默认 prompt 更好，但类别上有明显取舍：

- 提升明显：Events、FashionStyle、Tourism、DanceExercise。
- 退化明显：Cooking、DailyActivities、HomeDIY、Shopping、Sightseeing。

原因判断：`less_refusal` 减少了拒答，但有时回答过短，或者为了结合视频画面而忽略常识型问题。例如 `how/when/why/teach` 类问题不应被无关画面带偏。

因此新增第三个 prompt：

```text
--prompt-style balanced_detail
```

设计目标：

- 继续减少无意义拒答。
- 回答保持 1-2 句，保留关键可见细节，如名字、数字、颜色、标签、地点、动作。
- 常识题、教学题、原因题直接回答问题。
- 不编造精确地点、型号、原因。

前 10 条快速验证：

```text
EgoSchema LoRA balanced_detail, 10 samples: BLEU 0.1373
```

这个结果明显高于同样前 10 条的 default 和 less_refusal，因此已启动 50 条完整回测。

50 条完整回测已完成：

```text
EgoSchema LoRA balanced_detail, 50 samples: BLEU 0.1100
```

这是目前所有 50 条实验里最高的配置。

## 当前正在运行

正在测试更多帧配置：

```text
tmux: egoconv_50_egoschema_lora_frames16
adapter: models/lora/egoschema_v001_lora
frames_per_interval: 3
max_frames: 16
output: eval/results/egoschema_lora_v001_challenge2_egoconv_50_frames16_predictions.jsonl
```

跑完后自动评估：

```text
eval/results/egoschema_lora_v001_challenge2_egoconv_50_frames16_results_summary.json
```

`frames16` 50 条已完成：

```text
EgoSchema LoRA frames16, 50 samples: BLEU 0.0755
```

虽然比 `less_refusal` 略高，但明显低于 `balanced_detail`，因此不作为当前 700 条候选配置。

同时正在跑 50 条 prompt 完整回测：

```text
tmux: egoconv_50_egoschema_lora_balanced_detail
adapter: models/lora/egoschema_v001_lora
frames_per_interval: 2
max_frames: 12
prompt_style: balanced_detail
output: eval/results/egoschema_lora_v001_challenge2_egoconv_50_balanced_detail_predictions.jsonl
summary: eval/results/egoschema_lora_v001_challenge2_egoconv_50_balanced_detail_results_summary.json
```

## 当前 700 条运行

已启动当前最佳配置跑完整官方 700 条：

```text
tmux: egoconv_700_balanced_detail_wait
adapter: models/lora/egoschema_v001_lora
frames_per_interval: 2
max_frames: 12
prompt_style: balanced_detail
```

运行脚本：

```text
scripts/run_egoconv_700_balanced_detail.sh
```

输出位置：

```text
predictions: eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_predictions.jsonl
golden: runs/liuyichen/challenge2_egoconv_700_egoschema_lora_balanced_detail/golden.jsonl
run_report: runs/liuyichen/challenge2_egoconv_700_egoschema_lora_balanced_detail/run_report.json
run_log: runs/liuyichen/challenge2_egoconv_700_egoschema_lora_balanced_detail/run.log
eval_result: eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_results.json
eval_summary: eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_results_summary.json
eval_log: runs/liuyichen/challenge2_egoconv_700_egoschema_lora_balanced_detail/eval.log
```

启动时服务器 GPU 全部被占用，因此脚本包含等待空闲 GPU 的逻辑。当前已选中 GPU2 并开始推理。

## 评测指标说明

当前文档中的 BLEU 分数来自 starter kit：

```text
/data1/wearable_ai_challenge_data/starter_kit/run_evaluation.py
```

运行方式是：

```text
--task convqa --eval-only --no-llm-judge
```

因此这些 BLEU 分数只作为开发阶段快速 ablation 指标，用于比较 LoRA、prompt、frames 等设置。

官方 Challenge 2 / EgoConv leaderboard 排名指标是 LLM-as-Judge / Response Accuracy。starter kit README 明确说明 EgoConv 需要使用官方 judge：

```text
meta-llama/Llama-4-Maverick-17B-128E-Instruct
vLLM backend
tensor parallel size 8
online quantization fp8
```

已经准备官方 LLM judge 脚本：

```text
scripts/run_egoconv_700_llm_judge.sh
```

对应 tmux：

```text
egoconv_700_llm_judge_wait
```

该脚本会等待 700 条 predictions 完成，然后检查 vLLM 和 8 张 GPU 是否可用，再运行官方 judge。当前项目 `.venv` 中尚未安装 `vllm`，因此如果环境不补齐，脚本会记录：

```text
MISSING_VLLM
```

LLM judge 输出位置：

```text
eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_llmjudge_results.json
eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_llmjudge_results_summary.json
runs/liuyichen/challenge2_egoconv_700_egoschema_lora_balanced_detail/llm_judge.log
```

如果官方 judge 成功，脚本会生成附带 `llm_judge` 分数的提交候选：

```text
eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_predictions_with_llm_judge.jsonl
```

## 下一步建议

1. 等 700 条 `balanced_detail` 跑完。
2. 读取 700 条 BLEU summary，作为快速 sanity check。
3. 补齐 vLLM + Llama-4-Maverick judge 环境，运行官方 LLM-as-Judge。
4. 如果 `llm_judge` 分数正常，把附带 `{"llm_judge": score}` 的 jsonl 作为 leaderboard submission 候选。
5. 保留 run log、summary、配置记录，方便复现实验。
