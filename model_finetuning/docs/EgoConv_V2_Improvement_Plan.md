# EgoConv V2 Improvement Plan

Date: 2026-07-07

## Motivation

The current best configuration is:

```text
Qwen3-VL-8B-Instruct + EgoSchema LoRA
prompt_style: balanced_detail
frames_per_interval: 2
max_frames: 12
```

Partial 700 analysis indicates that refusals are no longer the main failure mode. The dominant issues are:

```text
missing_number: 189
entity_or_place_question: 182
number_or_date_question: 136
instruction_or_reason_question: 85
missing_year: 51
uncertainty_or_refusal: 23
```

This suggests that the next gains should come from better handling of:

```text
text in video
numbers and dates
entity/place/artifact names
long multi-turn context
```

## V2 Direction 1: Dynamic Prompt For Text/Number Questions

Add a prompt variant focused on visual text and numeric facts.

Trigger questions:

```text
how many
what number
what year
when
what does it say
read
spell
name of
artist
where
which color
```

Prompt behavior:

```text
1. Look for visible text, labels, numbers, signs, app screens, museum plaques, and product packaging.
2. Do not invent exact years, names, quantities, or locations if the text is unclear.
3. If a concise answer is sufficient, answer concisely but include the exact value.
4. For spelling questions, return the spelling directly.
5. For how-to/why questions, answer using general knowledge when video is not essential.
```

Risk:

```text
More conservative answers may lower BLEU if the reference contains specific facts and the model avoids guessing.
```

## V2 Direction 2: Dynamic Frame Allocation

Current frame sampling is fixed:

```text
frames_per_interval: 2
max_frames: 12
```

Proposed dynamic setting:

```text
normal questions:
  frames_per_interval = 2
  max_frames = 12

text/number/entity questions:
  current interval frames = 4 to 6
  keep previous intervals compressed
  max_frames = 16
```

Reason:

```text
Text and numbers are often visible for a short moment. Uniform low-density sampling may miss the key frame.
```

Implementation idea:

```text
For turn t:
  sample fewer frames from previous intervals
  sample more frames from interval t
  uniformly downsample to max_frames
```

This is more targeted than globally using frames16, which was slower and only reached BLEU 0.0755 on the 50-sample ablation.

## V2 Direction 3: OCR-Assisted Prompt

Use OCR on sampled frames and append the extracted text to the user prompt:

```text
Visible text candidates:
- ...
```

Potential OCR options:

```text
PaddleOCR
EasyOCR
Tesseract
Qwen visual OCR through an auxiliary pass
```

Start with a small smoke test on 10-20 samples containing text/number questions.

Risks:

```text
OCR setup may be heavy.
OCR errors can mislead the model.
Additional OCR pass increases runtime.
```

## V2 Direction 4: More Task-Matched LoRA Data

The current LoRA was trained on EgoSchema:

```text
single-turn multiple-choice video QA
```

EgoConv requires:

```text
multi-turn open-ended video QA
```

Next LoRA should use data closer to EgoConv:

```text
video + multi-turn user questions + open-ended assistant answers
```

Candidate sources:

```text
ShoweeHandv2 annotated data
AI-generated multi-turn QA from external egocentric videos
public egocentric caption/QA datasets converted into conversation format
```

Avoid data leakage:

```text
Do not train on the official EgoConv val answers and evaluate on the same val set.
Use external data or clearly separated held-out splits.
```

## Suggested Next Experiments After Current 700 Run

1. Wait for current 700 predictions and official LLM judge.
2. Select 50-100 hard cases from:

```text
partial_228_worst_turns.csv
partial_228_error_summary.json
```

3. Run a 50-sample V2 prompt test:

```text
prompt_style: text_numeric_careful
frames_per_interval: 2
max_frames: 12
```

4. Run a dynamic-frame 50-sample test:

```text
prompt_style: text_numeric_careful
dynamic current interval frames for text/number questions
max_frames: 16
```

5. Compare using:

```text
BLEU as a quick sanity signal
manual review on worst-turn CSV
official/smaller LLM judge if available
```

6. If V2 is better, run a second 700 submission candidate.

## Current Do-Not-Do

Do not interrupt the current 700 run unless it crashes. It is already more than a quarter complete and is producing valid predictions.
