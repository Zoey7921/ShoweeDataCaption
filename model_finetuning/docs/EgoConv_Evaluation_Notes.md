# EgoConv Evaluation Notes

Date: 2026-07-07

## Key Point

BLEU is not the official ranking metric for Challenge 2 / EgoConv.

We only use BLEU as a lightweight local development signal because it is fast and does not require running a large judge model. The official metric is LLM-as-Judge response accuracy.

## Why BLEU Is Limited

EgoConv is open-ended, multi-turn video QA. A prediction can be semantically correct while sharing few exact words with the reference answer.

Example pattern:

```text
Reference: The correct spelling is "naan", N-A-A-N, a type of flatbread.
Prediction: N-A-A-N.
```

This can be correct for the question, but BLEU may be low because the prediction is shorter and has fewer n-gram overlaps.

Therefore, BLEU should not be interpreted as a percent score. A BLEU of 0.10 does not mean the model is "10% correct".

## Current Use Of BLEU

We use BLEU for quick ablations:

```text
baseline_50:                    BLEU 0.0620
Qwen3-VL + EgoSchema LoRA:      BLEU 0.0686
LoRA + less_refusal:            BLEU 0.0743
LoRA + frames16:                BLEU 0.0755
LoRA + balanced_detail:         BLEU 0.1100
partial 700 balanced_detail:    BLEU 0.1037 on 228 conversations
```

Interpretation:

```text
balanced_detail is a stronger local development configuration than baseline under BLEU,
but final quality must be judged by official LLM-as-Judge.
```

## Official Metric

The official EgoConv leaderboard uses LLM-as-Judge / response accuracy. The judge assigns:

```text
1.0 = correct
0.5 = partially correct
0.0 = wrong or irrelevant
```

The required judge setup is:

```text
model: meta-llama/Llama-4-Maverick-17B-128E-Instruct
backend: vLLM
tensor parallel size: 8
online quantization: fp8
```

## Our Prepared Judge Environment

Environment:

```text
/data1/shared_data/qwen3vl-showeeData/.venv-vllm-judge
```

Verified:

```text
vllm 0.19.1
torch 2.10.0+cu128
cuda_available True
gpu_count 8
```

Judge script:

```text
/data1/shared_data/qwen3vl-showeeData/scripts/run_egoconv_700_llm_judge.sh
```

Waiting tmux:

```text
egoconv_700_llm_judge_wait
```

The script waits for the 700 prediction rows, then waits for 8 free GPUs, then runs the official judge.

## Reporting Wording

Use this wording in reports:

```text
We use BLEU only as a lightweight development metric for rapid ablations. The official EgoConv metric is LLM-as-Judge response accuracy. After selecting the best configuration with BLEU-based sanity checks, we evaluate the full 700 validation predictions using the official Llama-4-Maverick judge.
```

Do not write:

```text
Our official score is BLEU 0.1100.
```

Write:

```text
Our best local BLEU sanity-check configuration reached 0.1100 on a 50-conversation subset. The official LLM-as-Judge evaluation is pending completion of the 700-run predictions.
```
