# Gemma 4 quality and factuality report

Generated: 2026-08-08T18:54:19.490015+00:00  
Dataset: `gemma-quality-v1.0.0` (`1da22586dbe86d5b9d04ff25a3fc6706cbb08e1eac6ba3412b7060e24662a469`)  
Campaign status: **complete**

> Quality scores come from deterministic, reference-based checks and are not comparable to latency or throughput metrics. Factual references are snapshots; time-sensitive claims must be interpreted using their source dates.

## Model-by-category scores

| Model | closed_book_factual_qa | multi_hop_factual | arithmetic_reasoning | instruction_following | abstention_uncertainty | Aggregate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gemma-4-E4B-it-NVFP4 | 96.7% | 96.7% | 84.0% | 81.1% | 40.0% | 85.5% |
| gemma-4-12b-it-NVFP4 | 100.0% | 96.7% | 88.0% | 86.7% | 40.0% | 88.3% |
| gemma-4-26B-A4B-it-NVFP4 | 100.0% | 91.7% | 96.0% | 86.7% | 30.0% | 88.3% |

## Audit metrics

| Model | Exact match | Fact precision | Fact recall | Abstention/refusal | Error or unanswered | Cost/prompt | Raw outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gemma-4-E4B-it-NVFP4 | 90.9% | 100.0% | 95.1% | 40.0% | 0.0% | $0.000294 | [raw.jsonl](../results/gemma-4-E4B-it-NVFP4/raw.jsonl) |
| gemma-4-12b-it-NVFP4 | 94.5% | 100.0% | 95.1% | 40.0% | 0.0% | $0.000393 | [raw.jsonl](../results/gemma-4-12b-it-NVFP4/raw.jsonl) |
| gemma-4-26B-A4B-it-NVFP4 | 98.2% | 100.0% | 90.2% | 30.0% | 0.0% | $0.001354 | [raw.jsonl](../results/gemma-4-26B-A4B-it-NVFP4/raw.jsonl) |

## Execution provenance

| Model | Model revision | Startup seconds | Active wall seconds | Total cost | Settings hash |
| --- | --- | ---: | ---: | ---: | --- |
| gemma-4-E4B-it-NVFP4 | `1c363766fcfe575ac17a6d544963201a0d9b24c3` | 138.03267543500988 | 153.63089279498672 | $0.029446 | `0dc735fcec2d39715161739da70c894fc6fb52d47cfbf63bd06060b9665d05e0` |
| gemma-4-12b-it-NVFP4 | `b1f649734b34aa5575b03d186abd1b9be3d0d5c4` | 186.0394308739924 | 205.14120442804415 | $0.039319 | `0dc735fcec2d39715161739da70c894fc6fb52d47cfbf63bd06060b9665d05e0` |
| gemma-4-26B-A4B-it-NVFP4 | `20df0542b1a86ce19f495ac2eca2c7c12bce82f9` | 696.1778464799863 | 706.1884113569977 | $0.135353 | `0dc735fcec2d39715161739da70c894fc6fb52d47cfbf63bd06060b9665d05e0` |

## Failure classifications

| Model | Classification | Count |
| --- | --- | ---: |
| All models | none | 0 |
