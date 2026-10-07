---
base_model: unsloth/Qwen3.5-4B
library_name: peft
license: apache-2.0
language:
- vi
tags:
- lora
- education
- ticket-classification
---

# Lab21 Vietnamese ticket classification LoRA

Educational adapter by Nguyễn Nhân Sâm, student ID 2A202602672.
Base: `unsloth/Qwen3.5-4B`. This directory contains a PEFT adapter, requiring the base model separately.

## Training and data

Original course dataset: 250 synthetic customer-support tickets, split 225/25 with seed42.
LoRA text-linear placement, rank16, alpha32, learning rate1e-4, 30steps,
assistant-only pretokenized supervision, TeslaT4 fp16 base. Saved adapter weights are F32.
The custom education-support draft is a separate dataset; this adapter was not trained on it.

## Evaluation and limitations

On the course's fixed 50-ticket target evaluation, mean four-field accuracy increased
from0.765 with the optimized baseline prompt to0.970 with the adapter. Full-ticket
accuracy was44/50. JSON format score was1.0. On15 general regression questions,
keyword recall declined from0.791111 to0.522222. The original gate therefore returned
**FAILED**. This adapter is an experimental teaching artifact; these measurements do
not support a deployment recommendation.

The target pairs contain33wins,17ties,0losses. Raw original FT regression completions
were not retained. Model/tokenizer revisions were null in the original manifest,
so exact historical remote revision reproduction is not established. Training logs
contain some `grad_norm=nan`; effective skipped fp16 updates were not recorded.
All saved adapter tensor values were checked finite.

## Reproduction

Code and report: https://github.com/Nguyen-Sam-sheep-zzz/Day21-Track3-Finetuning-Lab/tree/feature/lab21-finetuning

Original GPU code commit: `90679f91842e9f42c9192b9c5eab316935705e9c`.
See `src/labkit/generate.py` and `notebooks/05_evaluate_and_verdict.py` for the exact
template, prompt and deterministic generation protocol. Full evidence is in `results/`.
Loading and running requires compatible torch/transformers/peft and adequate GPU memory.

## License and provenance

The base model's Hugging Face metadata was checked on2026-10-07 and reports Apache-2.0.
The current base revision at that check was `3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636`;
it is not asserted to be the historical training revision. Original source data and
evaluation files remain unchanged. AI assisted evidence tooling, audit and report editing;
the student executed the original GPU experiment and supplied the checkpoints.
