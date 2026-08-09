# MoE efficiency evaluation plan

This is a separately budgeted performance study. Its executable implementation
is [`moe_runner.py`](moe_runner.py), with fixed defaults in
[`moe_config.json`](moe_config.json). It must not alter the
100-prompt quality campaign or the existing synthetic benchmark rankings.

Qwen3.6 35B is skipped. The active matrix uses the three Gemma 4 checkpoints:
E4B and 12B as dense baselines, plus 26B A4B for MoE routing/concurrency
characterization.

## Questions

1. How much quality is obtained per active parameter and per dollar?
2. Does expert routing reduce arithmetic enough to offset routing, dispatch, and
   memory overhead on the RTX 5090?
3. How sensitive are latency and throughput to batch size, prompt length, and
   expert-load imbalance?
4. Which valid kernel/backend configuration gives the best efficiency without
   changing model outputs?

## Controlled comparisons

Use the same Pod shape, software image, tokenizer, context limit, prompt IDs,
seed, temperature, top-p, output cap, and request schedule for every arm. Run
models serially and reserve a fresh budget. At minimum include:

- E4B, 12B, and 26B A4B as the current checkpoints;
- the published total parameter count and active-parameter count per token;
- interactive and concurrent throughput workloads at 128, 512, and 2,048
  input tokens;
- a fixed quality subset and the full quality suite only as an output-equivalence
  check, never as a throughput workload;
- the selected `linear-backend=auto, moe-backend=flashinfer_cutlass` arm and a
  separately approved alternative backend only when it is compatible with the
  checkpoint.

Do not compare different quantization, context, offload, speculative-decoding,
or parallelism settings in one table.

## Telemetry

Capture raw vLLM results and GPU samples plus, when the runtime exposes them:

- experts selected per token and tokens per expert;
- expert-load mean, p50/p95, coefficient of variation, and max/min ratio;
- dispatched and returned token counts, overflow/dropped-token count, and
  all-to-all/dispatch time;
- prefill/decode time, TTFT, TPOT, end-to-end latency, output/total TPS;
- peak memory, KV-cache occupancy, GPU utilization, power, and temperature;
- kernel-selection lines and any compilation/JIT duration.

If a metric is unavailable, record `null` and the exact runtime/version rather
than inferring it from total throughput.

## Derived metrics

Report both raw and normalized values:

- quality per active parameter and quality per total parameter;
- output tokens per GPU-hour and per dollar;
- tokens per joule and dispatch overhead fraction;
- routing imbalance and its correlation with p95 latency;
- active/total parameter ratio and throughput relative to the dense baseline;
- output equivalence rate on the fixed subset under identical decoding.

Confidence intervals should be bootstrap intervals over prompts or repetitions;
do not treat concurrent requests as independent model repetitions.

## Artifacts and acceptance

Use a separate tree such as `moe/20260809T.../` with immutable config, raw
telemetry, normalized records, environment, and `summary/moe-report.md`. The
report must show arm/model/backend tables, routing distributions, failure
classes, costs, and links to raw output. Acceptance requires identical prompt
IDs/settings across arms, no hidden fallback kernel, complete raw telemetry,
and an explicit partial status when an arm fails. A follow-up campaign can add
expert-targeted prompts or an analytical dense baseline after this controlled
study is complete.
