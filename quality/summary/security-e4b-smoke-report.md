# Gemma 4 E4B OTRF security smoke report

Status: **partial, unscored smoke**

This run exercised the triggered-event security reasoning path against an anonymized OTRF attack scenario. It is not an accuracy benchmark: the prepared scenario does not yet include benign controls or sealed event-level ground truth. Model classifications and risk scores are recorded as outputs, not treated as correct answers.

## Configuration

| Field | Value |
|---|---|
| Model | `unsloth/gemma-4-E4B-it-NVFP4` |
| Revision | `1c363766fcfe575ac17a6d544963201a0d9b24c3` |
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB |
| vLLM | 0.26.0 |
| Quantized linear backend | FlashInfer CUTLASS NVFP4 |
| Attention backend | Triton attention |
| Context limit | 4,096 tokens |
| Maximum sequences | 16 |
| Seed | 0 |
| Mode | triggered, 30-second windows |
| Input bounds | 20 events and 8,000 serialized prompt bytes |
| Hourly rate | $0.69 |

Prefix caching was enabled by the vLLM 0.26.0 default. This does not affect the unscored smoke status, but the run must not be used as a ranked performance result.

## Final campaign metrics

| Metric | Result |
|---|---:|
| Investigations | 9 |
| Contract-valid successes | 8 |
| Preserved contract failures | 1 |
| Prompt tokens | 24,105 |
| Completion tokens | 6,877 |
| Total tokens | 30,982 |
| Successful-request tokens | 27,772 |
| Median request latency, successful requests | 3.715 s |
| Median TTFT, successful requests | 38.9 ms |
| Median TPOT, successful requests | 4.975 ms/token |
| Aggregate decode throughput, successful requests | 201.3 output tokens/s |
| Output throughput including TTFT | 198.8 output tokens/s |
| Sequential request throughput | 0.272 requests/s (16.3/min) |

The eight valid responses all returned `suspicious`; their median risk score was 70. These are model outputs only and are not accuracy results.

## Full smoke-session operations

- Pod rental duration: 22 minutes 35 seconds, from 2026-08-09 17:37:06 UTC to 17:59:41 UTC.
- Estimated rental cost: $0.260 at $0.69/hour.
- Inference activity window: 7 minutes 11 seconds, including bounded diagnosis pauses.
- Unique inference attempts: 30 (8 successful, 22 failed during contract and output-cap hardening).
- Known metered usage across 17 attempts: 59,421 tokens. Thirteen early failures occurred before error responses retained usage, so this is a lower bound rather than a full-session token total.
- Summed attempt execution time: 123.6 seconds.
- GPU samples: 68; peak memory 29,626 MiB, peak utilization 100%, peak power 345 W, and peak temperature 68 C.
- The authenticated shutdown path verified campaign hashes, stopped pod `04mfzkiacvw5lr`, and confirmed zero running pods. Its persistent volume was preserved.

## Failure findings and harness changes

The first dry-run context was 234 KB, so inference was blocked before GPU use. Triggered contexts now retain semantic evidence fields, discard duplicated raw message text, and split at hard event/byte bounds.

The initial model responses exceeded 512 and then 1,024 output tokens. Concise response caps, strict JSON schema metadata, raw error-output retention, Gemma code-fence/sentinel handling, null-safe campaign finalization, and explicit unscored-smoke manifests were added. The remaining failed investigation still echoed an unrequested `investigation` object and exhausted its output cap; the failure was retained rather than converted into a score.

## Artifact integrity

Raw artifacts are retained in the ignored local collection at `security-data/runs/e4b-otrf-attack-01-smoke/remote/`. The collected tree passed its complete SHA-256 manifest. Key hashes:

| Artifact | SHA-256 |
|---|---|
| Final `raw.jsonl` | `85612e34f9277548a740c947fc51f0d20de9d87c60ba6a4854630744dfe57492` |
| Final `contexts.jsonl` | `2f77bbe63f7c5903a6f56fb467e60913ade2f6193995526a66825d6aa98c1731` |
| Final `campaign.json` | `1998f18343c56f26a96a7e6f7b918671693dad478a13c9609bccaabd512cc24b` |
| Execution summary | `8448c6ae5530afa4eda99bdcfb4fafe7dac8fdbd32972e3523ef37f06ec6243a` |
| vLLM server/backend log | `28922a62ffcd7fef5c4391e3f04fa9d71c860926b3459840b59956da3d8aca55` |

The tracked machine-readable summary is in `quality/results-security/e4b-otrf-attack-01-smoke/execution-summary.json`.
