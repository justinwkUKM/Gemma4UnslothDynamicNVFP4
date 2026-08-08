# Gemma 4 quality and factuality report

Generated: 2026-08-08T18:28:41.139649+00:00  
Dataset: `gemma-quality-v1.0.0` (`1da22586dbe86d5b9d04ff25a3fc6706cbb08e1eac6ba3412b7060e24662a469`)  
Campaign status: **complete**

> Quality scores come from deterministic, reference-based checks and are not comparable to latency or throughput metrics. Factual references are snapshots; time-sensitive claims must be interpreted using their source dates.

## Model-by-category scores

| Model | closed_book_factual_qa | multi_hop_factual | arithmetic_reasoning | instruction_following | abstention_uncertainty | Aggregate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gemma-4-E4B-it-NVFP4 | 16.7% | 0.0% | 0.0% | 0.0% | 0.0% | 5.0% |

## Audit metrics

| Model | Exact match | Fact precision | Fact recall | Abstention/refusal | Error or unanswered | Cost/prompt | Raw outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gemma-4-E4B-it-NVFP4 | 100.0% | n/a | n/a | n/a | 95.0% | $0.016650 | [raw.jsonl](../results/gemma-4-E4B-it-NVFP4/raw.jsonl) |

## Failure classifications

| Model | Classification | Count |
| --- | --- | ---: |
| gemma-4-E4B-it-NVFP4 | `missing_result` | 95 |
