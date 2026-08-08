# Gemma 4 NVFP4 benchmark report

Generated: 2026-08-08T17:37:27Z  Rate: $0.69/GPU-hour

This final report combines the original E4B/12B campaign with the corrected 26B-only rerun. The 26B rerun reused cached weights and kernels; no model was reinstalled. All measured requests completed successfully.

## Results
| Model | Workload | Median output TPS | Median total TPS | TTFT ms | TPOT ms | Request TPS | Failures |
|---|---|---:|---:|---:|---:|---:|---:|
| gemma-4-E4B-it-NVFP4 | interactive | 208.425 | 420.596 | 19.926 | 4.770 | 0.407 | 0 |
| gemma-4-E4B-it-NVFP4 | throughput | 2718.524 | 5487.171 | 47.303 | 5.229 | 5.310 | 0 |
| gemma-4-12b-it-NVFP4 | interactive | 108.499 | 219.795 | 31.316 | 9.169 | 0.212 | 0 |
| gemma-4-12b-it-NVFP4 | throughput | 1315.551 | 2665.635 | 67.150 | 10.944 | 2.569 | 0 |
| gemma-4-26B-A4B-it-NVFP4 | interactive | 215.365 | 436.283 | 18.005 | 4.616 | 0.421 | 0 |
| gemma-4-26B-A4B-it-NVFP4 | throughput | 1762.931 | 3572.139 | 130.088 | 8.180 | 3.443 | 0 |

## Failures

```json
[]
```
