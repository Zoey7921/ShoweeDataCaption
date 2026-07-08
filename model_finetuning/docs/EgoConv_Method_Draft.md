# EgoConv Challenge 2 Method Draft

Date: 2026-07-07

## Task

Challenge 2 / EgoConv is a multi-turn egocentric video conversation task. For each conversation, the model receives a long first-person video, time-aligned question turns, and the dialogue history up to the current turn. The model must answer each turn without using future video segments.

## Model

Base model:

```text
Qwen3-VL-8B-Instruct
```

Adapter:

```text
models/lora/egoschema_v001_lora
```

The final inference model is:

```text
Qwen3-VL-8B-Instruct + EgoSchema LoRA adapter
```

## Training

We used EgoSchema as an external egocentric video QA dataset for lightweight LoRA adaptation.

Processed data:

```text
data/processed/egoschema_v001_train_clean.json
data/processed/egoschema_v001_val_clean.json
```

Training configuration:

```text
configs/lora_egoschema_v001.yaml
```

Training output:

```text
models/lora/egoschema_v001_lora
runs/liuyichen/egoschema_v001_lora/run_report.json
```

Training summary:

```text
train samples: 450
updates: 113
trainable parameters: 43,646,976
trainable percent: 0.4954%
peak memory: about 19.5 GB
```

## Inference

Main inference script:

```text
scripts/qwen3vl_egoconv_smoke.py
```

For each conversation:

1. Read `video_path`, `video_intervals`, and `questions` from the official EgoConv JSONL.
2. Process turns sequentially.
3. At turn `t`, sample frames from video intervals `0..t`, so the model only sees the current and past video.
4. Provide the current question and previous model answers as conversation context.
5. Generate one answer per turn.

Final 700-run configuration:

```text
adapter: models/lora/egoschema_v001_lora
prompt_style: balanced_detail
frames_per_interval: 2
max_frames: 12
max_new_tokens: 160
```

Run script:

```text
scripts/run_egoconv_700_balanced_detail.sh
```

Output:

```text
eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_predictions.jsonl
```

## Prompt Strategy

We compared several answer strategies. The best local BLEU configuration used `balanced_detail`.

Intent of `balanced_detail`:

- Reduce unnecessary refusal.
- Answer in one or two concise sentences.
- Preserve visible details such as object names, numbers, colors, places, and actions.
- Use general knowledge for factual, how-to, why, and when questions.
- Avoid inventing exact names, locations, models, or causes when not visible or strongly implied.

## Development Ablations

Local BLEU is not the official metric; it was used only for fast development.

50-conversation ablation results:

```text
Qwen3-VL baseline:                  BLEU 0.0620
Qwen3-VL + EgoSchema LoRA:          BLEU 0.0686
LoRA + less_refusal prompt:         BLEU 0.0743
LoRA + frames16:                    BLEU 0.0755
LoRA + balanced_detail prompt:      BLEU 0.1100
```

Partial 700-run sanity check:

```text
228 / 700 conversations
BLEU: 0.1037
```

Partial analysis report:

```text
runs/liuyichen/challenge2_egoconv_700_egoschema_lora_balanced_detail/partial_228_analysis.md
```

## Official Metric

The official EgoConv leaderboard metric is LLM-as-Judge / response accuracy, not BLEU. BLEU is only used as a lightweight sanity check.

The official judge environment has been prepared:

```text
.venv-vllm-judge
vllm 0.19.1
torch 2.10.0+cu128
```

Judge script:

```text
scripts/run_egoconv_700_llm_judge.sh
```

Judge tmux:

```text
egoconv_700_llm_judge_wait
```

Expected official judge outputs:

```text
eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_llmjudge_results.json
eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_llmjudge_results_summary.json
eval/results/egoschema_lora_v001_challenge2_egoconv_700_balanced_detail_predictions_with_llm_judge.jsonl
```

## Current Known Issues

- Fine-grained museum, exhibit, artist, and location questions remain difficult.
- The model sometimes hallucinates plausible names when text is visually ambiguous.
- Longer multi-turn conversations can propagate earlier mistakes into later answers.
- BLEU can under-reward short correct answers and over-reward wording overlap, so final conclusions should use LLM-as-Judge.

## Next Steps

1. Wait for all 700 predictions to complete.
2. Run full BLEU sanity check with starter kit.
3. Run official LLM-as-Judge once 8 GPUs are free and Hugging Face access is available.
4. Submit the prediction JSONL with appended `{"llm_judge": score}`.
5. If time remains, build a closer EgoConv-style LoRA dataset from multi-turn egocentric data instead of only EgoSchema multiple-choice QA.
