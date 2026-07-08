# Qwen3-VL ShoweeData LoRA 微调方案

更新时间：2026-07-03 15:45 CST

## 1. 当前可用基础

当前项目已经具备 LoRA smoke 训练的前置条件：

- 基础模型：`models/base/Qwen3-VL-8B-Instruct -> /data1/Qwen3-VL-8B-Instruct`
- Python 环境：`.venv`，已验证 `torch==2.12.1+cu126`、`transformers==5.12.1`、`accelerate==1.14.0`
- GPU：当前环境可见 8 张 A800 80GB
- 视频读取：已用 `decord` 预抽帧绕开 `torchcodec`/`torchvision.io.read_video` 问题
- 数据：
  - smoke：`data/processed/smoke_train.json`，1 条，2 轮问答
  - v001 train：`data/processed/showee_train_v001.json`，14 条，每条 3 轮问答
  - v001 val/test：`eval/sets/showee_val_v001.json`、`eval/sets/showee_test_v001.json`，各 3 条
- 复核状态：v001 共 20 条均为 `review_status: edited`

当前状态：

- `.venv` 已安装 `peft==0.19.1`
- 已新增 LoRA 训练与推理入口
- 已生成 `models/lora/showee_smoke_lora_v001/` 和 `models/lora/showee_train_v001_lora/`
- 已生成完整 8 epoch adapter：`models/lora/showee_train_v001_lora_8epoch/`
- 已完成 Showee v001 baseline/LoRA 同集评估，详见 `docs/LoRA实验记录.md`
- `trl`、`bitsandbytes`、`deepspeed` 仍未安装

## 2. 微调目标

第一阶段目标不是追求最终指标，而是验证链路：

1. 能读取 ShareGPT-like 视频多轮问答数据。
2. 能把视频均匀抽帧后喂给 Qwen3-VL processor。
3. 能只训练 LoRA adapter 并保存 checkpoint。
4. 能加载 base model + LoRA adapter 做推理。
5. 能在固定 val/test 上保存 baseline 与 LoRA 输出，便于人工对比。

## 3. 推荐路线

### 3.1 先做 bf16 LoRA，不急于 QLoRA

当前机器是 A800 80GB，Qwen3-VL-8B 使用 bf16 LoRA 更稳。第一版建议：

- base model bf16 加载
- 冻结视觉塔和大部分基础权重
- 对语言模型 attention/MLP 线性层挂 LoRA
- 每条视频固定抽 8 或 16 帧
- batch size 从 1 开始，gradient accumulation 做到有效 batch 4

QLoRA 放到第二步做。原因是当前环境还没有 `bitsandbytes`，而 4bit 量化对多模态模型和新 transformers 版本更容易遇到兼容性问题。

### 3.2 数据使用顺序

按风险从低到高推进：

1. `smoke_train.json`：1 条样本，目标是过拟合，确认 loss 下降和 adapter 可保存。
2. `showee_train_v001.json`：14 条样本，目标是小规模有效训练。
3. 固定 `showee_val_v001.json`/`showee_test_v001.json`：只做推理评估，不进入训练。

### 3.3 训练样本构造

训练时保留完整多轮对话，但 loss 只计算 assistant token：

```text
user: <video>\n请描述视频中正在执行的手势任务。
assistant: ...
user: 这个手势的关键手部动作是什么？
assistant: ...
user: 动作从头到尾有什么变化？如果看不清也说明。
assistant: ...
```

实现上应通过 processor 的 chat template 构造 input，再把 user/system 部分 label 置为 `-100`。如果第一版 masking 成本过高，可以先做保守版本：整段文本建模跑通链路，但正式实验前必须改成只监督 assistant。

## 4. 第一版训练脚本设计

建议新增：

```text
scripts/train_qwen3vl_lora.py
configs/lora_showee_smoke.yaml
configs/lora_showee_v001.yaml
scripts/infer_qwen3vl_lora_showee.py
```

训练脚本核心逻辑：

1. 读取 JSON list。
2. 用 `decord` 从 `video` 路径均匀抽 `num_frames` 帧。
3. 将每条 conversations 转成 Qwen3-VL messages，其中第一轮 user content 包含 `video`，后续轮次只包含文本。
4. 调用 `processor.apply_chat_template(..., tokenize=True, return_tensors="pt")`。
5. 构造 labels，只保留 assistant 答案 token。
6. 加载 `Qwen3VLForConditionalGeneration.from_pretrained(..., dtype=torch.bfloat16)`。
7. 用 PEFT `LoraConfig` 包装模型。
8. 使用 `transformers.Trainer` 或自定义最小训练循环保存 adapter。

## 5. LoRA 参数建议

smoke 过拟合：

```text
num_frames: 8
epochs: 20
learning_rate: 1e-4
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
batch_size: 1
gradient_accumulation_steps: 1
max_seq_length: 2048
output_dir: models/lora/showee_smoke_lora_v001
```

v001 小规模训练：

```text
num_frames: 8 或 16
epochs: 5-10
learning_rate: 5e-5
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
batch_size: 1
gradient_accumulation_steps: 4
max_seq_length: 3072
output_dir: models/lora/showee_train_v001_lora
```

如果 16 帧显存压力过大，优先降 `num_frames` 到 8，再降 `max_seq_length`，不要先改数据格式。

## 6. target_modules 选择

第一版优先只训语言模型线性层，避免视觉编码器过拟合小数据：

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

如果 PEFT 自动匹配失败，应先打印 `model.named_modules()`，确认 Qwen3-VL 在当前 transformers 版本中的真实模块名，再配置 target_modules。不要用 `all-linear` 盲挂到视觉塔，除非已经确认显存和效果。

## 7. 依赖安装建议

最小 bf16 LoRA：

```bash
cd /data1/shared_data/qwen3vl-showeeData
source .venv/bin/activate
pip install -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com peft
```

可选 QLoRA：

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com bitsandbytes
```

暂不建议第一版引入 `deepspeed` 或 `flash-attn`。如果训练能跑但速度慢，再单独评估加速依赖，避免把环境问题和训练逻辑问题混在一起。

## 8. 验收标准

smoke 训练通过标准：

- 训练能完成并生成 `adapter_config.json`、`adapter_model.safetensors`
- 训练日志里 loss 有明显下降
- 加载 base + adapter 后，对 smoke 样本能复现任务名和关键动作
- 记录峰值显存、训练耗时、帧数、seq length

v001 小实验通过标准：

- 只用 14 条 train 训练
- 不读取 val/test 作为训练样本
- 保存 baseline 输出和 LoRA 输出到 `eval/results/`
- 对每个 val/test 样本保留 prompt、prediction、reference、model_path、adapter_path、num_frames
- 人工检查是否减少任务名错误、手指动作幻觉和拒答

## 9. 评估建议

自动指标只做辅助。Showee v001 是开放式中文手势描述，BLEU 不稳定，第一轮更适合记录：

- task hit：是否说对 `task_name`
- motion hit：是否提到关键手指/手腕动作
- hallucination：是否编造物体、场景或无法确认的指尖关系
- uncertainty：是否在看不清时保守说明

后续可以写一个半自动评估脚本，先用规则检查任务名和关键词，再用人工复核表确认质量。

## 10. 下一步执行顺序

1. 扩展 v002 到约 50 条高质量样本，覆盖 val/test 相近手势类型。
2. 增强评估脚本，加入 motion hit、幻觉、不确定性表达统计。
3. 增加候选式任务名选择评估，和开放式描述分开看。
4. 如需 QLoRA，再安装并单独验证 `bitsandbytes`。
