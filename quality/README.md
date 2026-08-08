# Gemma 4 quality and factuality campaign

This tree is independent of the synthetic performance campaign in
`benchmarks/`. It sends the same 100 curated prompts to all three checkpoints,
uses deterministic decoding, and applies reference-based scoring. Quality
scores do not measure latency or throughput and cannot be compared with those
metrics.

## Fixed campaign contract

- Dataset: `gemma-quality-v1.0.0`, 100 prompts, byte-level SHA-256 recorded in
  `dataset_manifest.json`.
- Models: E4B, 12B, and 26B A4B NVFP4 checkpoints listed in `runner.py`.
- Decoding: temperature 0, top-p 1, top-k -1, seed 0, and 256 generated tokens.
- Execution: one vLLM server and one model at a time; completed prompt IDs are
  never requested again on resume.
- Default quality budget: two runner hours and USD 1.50 of compute at the
  supplied hourly rate. This is separate from the five-hour performance budget.
  Stopping the process does not stop Pod billing; the operator must stop the Pod
  after verified artifact collection.
- Reporting: the final report is withheld until all three raw result sets are
  complete. `--allow-partial` is an explicit exception for smoke runs or a
  deliberately partial campaign.

Factual references were checked on the snapshot dates in each JSONL record.
The date is part of the reference contract: a time-sensitive answer should not
silently be treated as timeless if its source later changes.

## Prompt list

The exact, human-readable list is in [`prompts.md`](prompts.md). The JSONL is
the scoring source of truth and additionally contains answer variants,
normalization rules, key facts, known contradictions, reference URLs, source
dates, and abstention/refusal expectations.

## Validate and test locally

```bash
python3 quality/evaluator.py --validate-only
python3 -m unittest discover -s quality/tests -v
```

## Smoke test

Upload the `quality/` tree to the Pod, activate the existing benchmark virtual
environment, and share only the Hugging Face model cache:

```bash
export HF_HOME=/workspace/gemma4-benchmark/cache/huggingface
python /workspace/gemma4-quality/runner.py \
  --results-dir /workspace/gemma4-quality/results-smoke \
  --models gemma-4-E4B-it-NVFP4 --limit 5 \
  --max-campaign-seconds 1800 --max-cost-usd 0.35
python /workspace/gemma4-quality/evaluator.py \
  --results-dir /workspace/gemma4-quality/results-smoke \
  --models gemma-4-E4B-it-NVFP4 --allow-partial \
  --report /workspace/gemma4-quality/summary/smoke-report.md
```

## Full campaign

```bash
export HF_HOME=/workspace/gemma4-benchmark/cache/huggingface
python /workspace/gemma4-quality/runner.py \
  --results-dir /workspace/gemma4-quality/results \
  --hourly-rate 0.69 --max-campaign-seconds 7200 --max-cost-usd 1.50
python /workspace/gemma4-quality/evaluator.py \
  --results-dir /workspace/gemma4-quality/results \
  --report /workspace/gemma4-quality/summary/quality-report.md
```

Each model directory contains `raw.jsonl`, `run.json`, the vLLM server log, and
the generated `scores.json`. `raw.jsonl` keeps the prompt, final output, timing,
token usage, request error, hashes, and every attempt. Environment files contain
an allow-listed hardware/software inventory rather than a complete environment
dump, preventing credential capture.

## Score interpretation

- Closed factual QA and arithmetic use normalized exact or numeric match.
- Multi-hop answers use required-fact precision, recall, and F1. Precision is
  reference-bound: it penalizes known contradictions in the dataset and is not
  a general-purpose hallucination detector.
- Instruction prompts use explicit JSON, regex, line, word, ordering, and term
  checks.
- Uncertainty and safety prompts require deterministic abstention or refusal
  language.
- The aggregate is the unweighted mean of all 100 per-prompt scores. Errors,
  missing outputs, and empty answers score zero and remain classified.

## Recorded campaign

The full run completed on the RTX 5090 Pod on 2026-08-08 UTC. Every model
received all 100 IDs with zero request errors or unanswered responses:

| Model | Aggregate | Exact match | Fact precision / recall | Abstention/refusal | Cost/prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| E4B | 85.5% | 90.9% | 100.0% / 95.1% | 40.0% | $0.000294 |
| 12B | 88.3% | 94.5% | 100.0% / 95.1% | 40.0% | $0.000393 |
| 26B A4B | 88.3% | 98.2% | 100.0% / 90.2% | 30.0% | $0.001354 |

The complete score and provenance tables are in
[`summary/quality-report.md`](summary/quality-report.md); each raw output and
the per-model server log are linked from the report.

The separate MoE routing/efficiency follow-up is scoped in
[`MOE_EVALUATION_PLAN.md`](MOE_EVALUATION_PLAN.md); it is intentionally not a
quality score or part of this campaign.
