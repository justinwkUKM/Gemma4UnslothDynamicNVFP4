# Model quality, MoE efficiency, and security reasoning campaigns

This tree is independent of the synthetic performance campaign in
`benchmarks/`. It sends the same 100 curated prompts to all three checkpoints,
uses deterministic decoding, and applies reference-based scoring. Quality
scores do not measure latency or throughput and cannot be compared with those
metrics.

Before any GPU is started, run `../scripts/test_all.sh`. The command includes
the original evaluator/runner tests plus Qwen backend gating, common campaign
provenance, MoE normalization, and security parser/replay/contract tests.

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

## Active model coverage

Qwen3.6 35B is explicitly skipped. Its compatibility runner remains available
for auditability but has no default model target. The active MoE and security
matrix uses the three Gemma 4 checkpoints: E4B, 12B, and 26B A4B. The generic
quality runner also defaults only to those Gemma models, preventing accidental
Qwen inclusion.

## MoE efficiency campaign

`moe_runner.py` implements `MOE_EVALUATION_PLAN.md` as a separately budgeted
matrix. The committed configuration fixes a 4,096-token context, seed 0,
matched 512-input/256-output workloads, three repetitions, and concurrency
levels 1/4/16. The MoE arm additionally restarts the same checkpoint with
routing capture enabled. Run-time-unavailable routing/dispatch fields are
reported as `null`, never estimated from throughput.

```bash
python3 quality/moe_runner.py --root /workspace/moe/UTC
```

Each campaign contains model metadata, detailed raw request results, server
logs, GPU samples, Prometheus snapshots, `summary/moe-summary.json`, CSV, and
`summary/moe-report.md`. Existing quality scores are read only for
output-equivalence normalization and are not folded into speed.

## Security reasoning campaign

The incremental harness is under [`security/`](security/). It supports the
required raw, fixed-window, triggered, stateful-memory, and bounded tool-using
modes. Public inputs must be anonymized and label-free; fresh cyber-range runs
use the separate `unseen` track. Ground truth is supplied only to the evaluator
after inference. The pinned OTRF source and OpTC documentation are downloaded;
ten OTRF attack archives have been CPU-normalized, while LANL and corrected
OpTC remain explicitly acquisition/storage gated. See
[`security/README.md`](security/README.md) for exact status and commands.

The broader future security telemetry track is described in
[`SECURITY_LLM_REASONING_BENCHMARK_PLAN.md`](SECURITY_LLM_REASONING_BENCHMARK_PLAN.md).
It will evaluate stateful real-time reasoning over OTRF, LANL, and DARPA OpTC
data, separately measuring intelligence and operational performance.
