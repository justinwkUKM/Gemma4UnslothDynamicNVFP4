# Gemma 4 RunPod NVFP4 Benchmark

This repository contains the reproducible benchmark specification and captured
results for Unsloth Gemma 4 NVFP4 text-generation checkpoints on RunPod.

It now has two deliberately separate tracks: the existing synthetic performance
campaign measures serving speed, while [`quality/`](quality/) measures factual
and instruction quality with 100 fixed prompts. A quality score is not a
throughput score and the two must not be combined into a single metric.

Copy [`.env.example`](.env.example) to `.env` for local CLI workflows. The
example contains placeholders only; `.env` is ignored and must never be
committed. SSH private keys and model/Hugging Face tokens follow the same rule.

The detailed, operational procedure is in
[`RUNPOD_GEMMA_BENCHMARK_SPEC.md`](RUNPOD_GEMMA_BENCHMARK_SPEC.md). It fixes the
Pod shape, software versions, kernel policy, workloads, retry rules, artifact
layout, and budget limits.

## How the benchmark works

1. **Prepare one Pod.** We use an RTX 5090 Community Cloud Pod with CUDA 12.9+
   or newer, a 40 GB container disk, and an 80 GB persistent volume. Python,
   vLLM, FlashInfer, and NVIDIA CUTLASS are installed under `/workspace`.
2. **Validate the machine.** The runner records the GPU identity (including
   SM120), CUDA/PyTorch versions, free disk, package versions, and available
   NVFP4 kernels before downloading models. Incompatible-kernel, CPU-offload,
   emulation, and OOM conditions fail clearly rather than producing misleading
   speed numbers.
3. **Load one model at a time.** vLLM downloads each checkpoint into the
   persistent Hugging Face cache and serves it on `localhost:8000`. Every model
   uses the same 8K context limit, 90% GPU-memory target, Gemma 4 reasoning
   parser, and accelerated linear/MoE backend policy. The server is cleaned up
   before the next checkpoint is loaded.
4. **Warm up.** Five preliminary batches allow CUDA kernels, memory allocation,
   and vLLM scheduling to settle. Warm-ups are not used for ranking.
5. **Measure two workloads.** Each workload has three measured repetitions:
   - *Interactive:* 512 input tokens, 512 output tokens, 10 prompts,
     concurrency 1.
   - *Throughput:* 512 input tokens, 512 output tokens, 100 prompts,
     concurrency 16.
6. **Save everything.** Raw vLLM JSON, per-run metadata, server logs, GPU
   samples, environment reports, and the generated summary remain under
   `/workspace` and are copied into timestamped `benchmarks/` snapshots here.

## Results so far

All three checkpoints completed all requests without failures. Values below
are medians across the three measured repetitions. “TTFT” is time to first
token; “TPOT” is time per generated token.

| Model | Workload | Output tok/s | Total tok/s | TTFT | TPOT |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma 4 E4B | Interactive | 208.4 | 420.6 | 19.9 ms | 4.77 ms |
| Gemma 4 E4B | Throughput | 2,718.5 | 5,487.2 | 47.3 ms | 5.23 ms |
| Gemma 4 12B | Interactive | 108.5 | 219.8 | 31.3 ms | 9.17 ms |
| Gemma 4 12B | Throughput | 1,315.6 | 2,665.6 | 67.1 ms | 10.94 ms |
| Gemma 4 26B A4B | Interactive | 215.4 | 436.3 | 18.0 ms | 4.62 ms |
| Gemma 4 26B A4B | Throughput | 1,762.9 | 3,572.1 | 130.1 ms | 8.18 ms |

In practical terms, E4B is the aggregate-throughput leader. 26B A4B slightly
leads E4B on this interactive sample, while 12B is substantially slower. At the
common GPU price of $0.69/hour, the output-tokens-per-dollar ranking follows
throughput: E4B, 26B A4B, then 12B. The first repetition can have higher TTFT
because of one-time compilation and startup effects, which is why the report
uses medians.

The quality campaign has its own report at
[`quality/summary/quality-report.md`](quality/summary/quality-report.md). It uses
the exact prompts listed in [`quality/prompts.md`](quality/prompts.md), fixed
decoding, reference snapshots, and deterministic scoring. Source snapshot dates
matter for any fact that can change over time.

The completed 2026-08-08 UTC run evaluated all 100 prompts on each model with
zero request errors. Aggregate quality was 85.5% for E4B, 88.3% for 12B, and
88.3% for 26B A4B. Exact-match accuracy was 90.9%, 94.5%, and 98.2%
respectively. These values are quality-only results and are not throughput or
latency rankings.

## Repository contents

- [`RUNPOD_GEMMA_BENCHMARK_SPEC.md`](RUNPOD_GEMMA_BENCHMARK_SPEC.md): complete
  provisioning, execution, collection, and shutdown specification.
- [`benchmarks/`](benchmarks/): timestamped raw-artifact snapshots, including
  the completed campaign at `20260808T173800Z`; secrets,
  private keys, model caches, and virtual environments are intentionally excluded.
- [`scripts/remote_preflight.sh`](scripts/remote_preflight.sh): read-only remote
  identity, GPU, disk, and environment check.
- [`scripts/remote_setup.sh`](scripts/remote_setup.sh): idempotent Python 3.13,
  vLLM, FlashInfer, and CUTLASS environment setup.
- [`scripts/collect_results.sh`](scripts/collect_results.sh): copies config,
  environment, logs, results, and summaries without stopping the Pod.
- [`quality/`](quality/): versioned quality dataset, manifest, resumable runner,
  evaluator, tests, audit outputs, and a separate quality report.
- [`scripts/collect_quality_results.sh`](scripts/collect_quality_results.sh):
  collects the independent quality artifacts without mixing them into a
  timestamped performance campaign.

The helper scripts deliberately do not create or terminate Pods. For example:

```bash
ssh root@HOST -p PORT -i ~/.ssh/id_ed25519 < scripts/remote_preflight.sh
ssh root@HOST -p PORT -i ~/.ssh/id_ed25519 < scripts/remote_setup.sh
scripts/collect_results.sh root@HOST PORT results/20260808T-final ~/.ssh/id_ed25519
```
