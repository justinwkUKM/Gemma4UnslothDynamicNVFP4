# RunPod Gemma 4 NVFP4 benchmark specification

Status: operational source of truth. Revision: 2026-08-09.

This document defines one reproducible, text-generation-only benchmark campaign for
the following Hugging Face checkpoints:

| ID | Checkpoint |
| --- | --- |
| `gemma-4-E4B-it-NVFP4` | `unsloth/gemma-4-E4B-it-NVFP4` |
| `gemma-4-12b-it-NVFP4` | `unsloth/gemma-4-12b-it-NVFP4` |
| `gemma-4-26B-A4B-it-NVFP4` | `unsloth/gemma-4-26B-A4B-it-NVFP4` |

The runner must not make undocumented choices. Values below are fixed unless a
run is explicitly marked as a deviation in its result manifest.

This specification covers the synthetic performance campaign. The independent
quality campaign is defined in section 12 and under `quality/`; its scores must
never be presented as latency, throughput, or a component of those metrics.

## 1. Provisioning and budget

Create one Community Cloud Pod with one RTX 5090, CUDA 12.9 or newer, a 40 GB
container disk, and an 80 GB volume disk mounted at `/workspace`. The RTX 5090
must report compute capability `12.0` (SM120). A different GPU is a provisioning
failure, except for the separately documented RTX PRO 6000 fallback.

The campaign has both hard limits:

* maximum wall-clock runtime: five hours from the recorded Pod start time;
* compute spend: USD 4.00, excluding storage.

Before setup, record `pod_started_at_utc`, the Pod GPU hourly rate, and the
currency/rate source. The runner computes `deadline = min(start + 5h,
start + $4 / hourly_rate)` and reserves 15 minutes for artifact collection. It
must not start a model if the projected remaining work cannot finish before the
reserved deadline. It kills the benchmark/server at the deadline, preserves all
partial results, and reports a budget-threatening event immediately. The cost
estimate is `hourly_rate * active_wall_seconds / 3600`; storage charges are
reported separately when known.

A billing ceiling cannot be guaranteed by killing a process: a running Pod may
continue to accrue GPU charges. Automatic Pod stop/terminate therefore requires
the operator’s explicit authorization and a usable RunPod API/console path. If
that authorization is absent, the operator must stop the Pod manually as soon as
the verified download completes; the runner never calls a destructive RunPod
operation on its own.

Do not provision a fallback merely because a model is slow. An RTX PRO 6000
fallback is permitted only after the 5090 compatibility failure is preserved,
the user approves the new Pod, the remaining compute budget covers the quoted
PRO 6000 rate for at least 90 minutes plus the collection reserve, and the
five-hour campaign deadline still permits it. A fallback is a new campaign
segment with its own GPU identity and rate in the manifest.

## 2. Persistence and remote layout

The volume, not the container disk, is durable when a Pod stops. Verify that
`/workspace` is the 80 GB volume before writing anything. All durable state is
under this tree:

```
/workspace/gemma4-benchmark/
  env/                         # virtual environment and Python 3.13 metadata
  cache/huggingface/            # HF_HOME, TRANSFORMERS_CACHE, HF_HUB_CACHE
  cache/uv/                     # uv downloads/cache
  config/                       # immutable generated config and command records
  logs/<model-id>/              # server, setup, watchdog and GPU samples
  results/<model-id>/           # raw vLLM JSON and parsed per-run records
  environment/                 # preflight, package and kernel reports
  summary/                      # generated Markdown and machine-readable summary
```

Never put tokens, private keys, or complete environment dumps in this tree. A
Hugging Face token, if the public checkpoints unexpectedly require one, is read
from the operator’s remote environment and is never written to logs or config.
The local destination is `results/<UTC timestamp>/` in this repository.

## 3. Remote setup (Python 3.13)

For local automation, copy `.env.example` to `.env` and fill in the RunPod and
GitHub credentials. These values are used only by local CLI/publishing tools;
they are never uploaded to the Pod or written into benchmark artifacts.

Run these commands as root or the Pod’s configured user, with the volume mounted:

```bash
set -Eeuo pipefail
export ROOT=/workspace/gemma4-benchmark
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export UV_CACHE_DIR="$ROOT/cache/uv"
mkdir -p "$ROOT"/{env,cache/huggingface,cache/uv,config,logs,results,environment,summary}

command -v curl >/dev/null
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/workspace/bin sh
  export PATH=/workspace/bin:$PATH
fi
export UV_PYTHON_INSTALL_DIR=/workspace/python
uv python install 3.13
uv venv --python 3.13 "$ROOT/env"
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0
export PATH="/usr/local/cuda-13.0/bin:$ROOT/env/bin:$PATH"
source "$ROOT/env/bin/activate"
uv pip install 'vllm[bench]>=0.25.0' 'flashinfer-python>=0.6.13' 'nvidia-cutlass-dsl>=4.5.2'
python -m pip check
```

Record the resolved versions (`python --version`, `vllm --version`,
`importlib.metadata.version` for `vllm`, `flashinfer-python`, and
`nvidia-cutlass-dsl`) and the full `pip freeze` in
`environment/packages.txt`. Installation failure is classified as
`dependency_installation`; no model download may begin.

The equivalent idempotent helper is `scripts/remote_setup.sh`. It writes
`environment/pip-check.txt`, `environment/versions.txt`, and
`environment/packages.txt`; it does not start a server or download a model.

## 4. Compatibility gate

Before downloading a checkpoint, save these reports:

```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version \
  --format=csv,noheader
df -h / /workspace
findmnt -T /workspace
python - <<'PY'
import json, platform, sys, torch, vllm
print(json.dumps({
  "python": sys.version, "platform": platform.platform(),
  "torch": torch.__version__, "torch_cuda": torch.version.cuda,
  "cuda_available": torch.cuda.is_available(),
  "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
  "vllm": getattr(vllm, "__version__", "unknown"),
}, indent=2))
PY
```

The gate passes only when there is one visible RTX 5090, capability `(12, 0)`,
CUDA is available, the driver/runtime is compatible with CUDA 12.9+, `/workspace`
is the persistent volume, and at least 65 GiB is free there (and 20 GiB on the
container filesystem). `torch.version.cuda` and the driver’s reported CUDA
version are both recorded; a missing `nvcc` is not itself a failure.

The server command must use the same mixed-kernel policy for every model. The
default selection for these mixed NVFP4/FP8 Gemma checkpoints is:

```bash
--linear-backend auto --moe-backend flashinfer_cutlass
```

FlashInfer CUTLASS provides the accelerated NVFP4 MoE path while vLLM’s
automatic linear dispatch handles checkpoint-specific FP8 scale layouts. The
server log and vLLM runtime configuration must show
`moe_backend='flashinfer_cutlass'` and an accelerated NVFP4 kernel. A startup
error or log indicating Marlin, emulation/dequantization, CPU offload, an
incompatible compute capability, or an NVFP4 kernel other than the selected
accelerated path fails the `kernel_selection` gate. Do not silently retry with
a slower backend. If the installed vLLM build does not expose the selected flag,
classify the run as `dependency_installation`/`kernel_selection` and stop.

## 5. Identical server settings

Models run serially in the table order above. Before each model, assert that
port 8000 is unused; start exactly one localhost server and wait for both
`GET /health` and `GET /v1/models` to succeed. Use:

```bash
vllm serve "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --served-model-name "$MODEL" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser gemma4 \
  --linear-backend auto \
  --moe-backend flashinfer_cutlass \
  --seed 0
```

The exact command, vLLM version, startup monotonic duration, peak GPU memory,
and server log are saved. Do not use tensor/pipeline parallelism, CPU offload,
quantization overrides, speculative decoding, prefix caching, or model-specific
flags. The model is unloaded before the next model. On every exit path, send
TERM to the recorded process group, wait up to 30 seconds, then KILL only that
recorded group if needed; never use an unscoped `pkill`.

## 6. Workloads and repetitions

Use `vllm bench serve` against `http://127.0.0.1:8000/v1/chat/completions` with
`backend=openai-chat`, dataset `random`, seed `0`, `random-range-ratio=0`, and
`ignore-eos` so lengths are fixed. Every model receives the same generated
workload parameters. Save detailed JSON for every invocation.

For each model, run five warm-up invocations first. A warm-up is five requests
of input length 512 and output length 512 at concurrency 1; warm-up output is
saved but excluded from rankings. Then run three measured repetitions of each:

| Workload | Input | Output | Requests | Max concurrency |
| --- | ---: | ---: | ---: | ---: |
| `interactive` | 512 tokens | 512 tokens | 10 | 1 |
| `throughput` | 512 tokens | 512 tokens | 100 | 16 |

The generated invocation is equivalent to:

```bash
vllm bench serve --backend openai-chat \
  --base-url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
  --model "$MODEL" --dataset-name random --random-input-len 512 \
  --random-output-len 512 --random-range-ratio 0 --num-prompts "$N" \
  --max-concurrency "$C" --seed 0 --ignore-eos --save-result --save-detailed \
  --result-dir "$ROOT/results/$MODEL_ID" --result-filename "$RUN_ID.json" \
  --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,90,99
```

The runner checks the installed `vllm bench serve --help` before the first
workload and records the help text. If a required flag is absent, classify as
`dependency_installation` and preserve the partial campaign.

Each measured invocation has a 20-minute timeout; startup has a 45-minute
timeout; no more than one retry is allowed for a transient HTTP/client failure.
Do not retry OOM, kernel, model-load, authentication, or deterministic CLI
errors. A retry gets a new run ID and remains visible in raw results. A timeout
terminates the server, records the last log/GPU sample, and proceeds to the next
model only if the budget guard permits it.

## 7. Metrics and result contract

For each raw vLLM JSON, retain the unmodified file plus a normalized record with:

* model ID/checkpoint, workload, repetition, seed, exact CLI, vLLM/package and
  GPU identity;
* `median_output_tps`, `median_total_tps`, `median_ttft_ms`, `median_tpot_ms`,
  `median_request_throughput`, and p50/p90/p99 where available;
* prompt tokens, generated tokens, wall time, failures, error strings, startup
  seconds, peak allocated GPU memory (MiB), and estimated USD cost.

The runner maps vLLM field names (`output_throughput`,
`total_token_throughput`, `request_throughput`, and the reported TTFT/TPOT
percentiles) into this schema without inventing missing values. “Median” is the
median of the three measured repetition values; missing/failed repetitions are
not silently dropped from the failure count. Raw detailed per-request errors are
always retained.

The generated `summary/benchmark-report.md` must contain one table per workload,
one failure table, the environment and kernel verdict, startup/peak-memory and
cost tables, and these rankings:

1. interactive speed: ascending median TTFT, then TPOT;
2. aggregate throughput: descending median total TPS;
3. tokens per dollar: descending `median_total_tps * 3600 / hourly_rate`.

A model is not ranked when its compatibility gate or all measured repetitions
failed. Partial results remain valid and are labeled partial.

## 8. Failure taxonomy and preservation

Every failed phase has exactly one primary class: `provisioning`,
`dependency_installation`, `download`, `model_loading`, `kernel_selection`,
`oom`, `benchmark_execution`, or `timeout`. The normalized record includes the
class, command/phase, exit code, first and last relevant log lines, retry count,
and suggested operator action. Preserve setup logs, server logs, watchdog GPU
samples, environment reports, all raw JSON, normalized JSON, and the partial
Markdown report even after a failure.

## 9. Collection, verification, and shutdown

After the last permitted model (or an earlier budget/deadline stop), stop the
server, write `manifest.sha256` over every remote artifact, and download the
entire `environment/`, `logs/`, `results/`, `summary/`, and `config/` trees to:

```
results/<UTC timestamp>/
```

Verify locally that the manifest hashes match, all completed invocations have a
raw JSON and normalized record, the Markdown report parses as UTF-8, and the
environment report identifies GPU, capability, CUDA, PyTorch, vLLM, and all
three requested package versions. A failed or incomplete transfer is a
collection failure and must be retried before any Pod shutdown.

Only after verification may the operator stop/terminate the Pod, and only when
the user has explicitly authorized that operation. If authorization is absent,
report the exact console/API action needed and leave the Pod untouched. Never
delete the 80 GB volume until the local verification and the user’s retention
decision are complete.

## 10. Generated configuration and provenance

The uploader creates an immutable `config/benchmark-config.json` containing the
model table, all fixed workload/server values, UTC start/deadline, GPU hourly
rate and source, retry/timeouts, selected kernel, git/spec SHA-256, and a
boolean `shutdown_authorized`. It also records the exact SSH destination in
memory only; private key contents and tokens are never uploaded. The runner
writes its own version and SHA-256 into the manifest so a result can be replayed
or audited without credentials.

The operator confirms the remote identity before setup by comparing the supplied
RunPod Pod ID/host and the remote `hostname`, `nvidia-smi -L`, and (when exposed)
`RUNPOD_POD_ID`. A mismatch aborts before installation or downloads.

## 12. Separate quality and factuality campaign

The quality campaign is a second experiment, not another performance workload.
It uses `quality/dataset_v1.jsonl`, whose exact bytes, version, 100-prompt count,
category quotas, and source snapshot dates are committed in
`quality/dataset_manifest.json`. The prompt catalog in `quality/prompts.md` is a
human-readable view; JSONL remains authoritative.

All three checkpoints receive the same ordered prompt IDs with temperature 0,
top-p 1, top-k -1, seed 0, and a 256-token output cap. Models load serially
through the same vLLM server configuration from section 5. The runner writes a
complete raw record after each response, retains all attempts, skips successful
prompt IDs when resumed, and retries only failed requests. Records include
prompt and output text, request timing and token usage, errors, model/checkpoint
identity, hashes of the dataset, generation settings, server command, runner,
and cached model configuration, plus a reference to allow-listed environment
metadata. Tokens and complete environment dumps are prohibited.

The v1 evaluator uses no model judge. It applies normalized exact/numeric match,
required-fact precision/recall, structured instruction checks, and explicit
abstention/refusal checks. It reports per-category and aggregate scores, exact
match, reference-bound factual precision and recall, abstention correctness,
response lengths, failure classes, and cost per evaluated prompt. Reference-
bound precision detects curated contradictions; it is not a general
hallucination detector. Raw JSONL remains the audit authority for every score.

The default quality runner budget is the lesser of two hours or USD 1.50 at the
recorded GPU hourly rate. This budget is separate from the five-hour/USD 4.00
performance campaign. As with the performance campaign, ending the runner does
not stop Pod billing; collection must be verified and the Pod stopped through
an authorized operator path.

Run a small `--limit` smoke sample first. A final report is emitted only after
all three result directories contain the exact 100 prompt IDs. `--allow-partial`
is permitted only when the campaign is explicitly labeled partial. Local
acceptance requires dataset/schema/hash validation, evaluator unit tests,
identical generation-setting hashes, complete UTF-8 JSONL, a secret scan, and
preserved error records. Quality results and performance results remain in their
respective artifact trees.

The first complete quality campaign finished on 2026-08-08 UTC on Pod
`321hxgl8vi7a5q` (RTX 5090, USD 0.69/hour). Each model produced 100/100 raw
records with zero request errors or unanswered responses. The recorded
aggregate scores were E4B 85.5%, 12B 88.3%, and 26B A4B 88.3%; exact-match
accuracy was 90.9%, 94.5%, and 98.2%, respectively. Full category scores,
source hash, model revisions, startup times, costs, and raw-output links are in
`quality/summary/quality-report.md`. These measurements are a quality result,
not a performance ranking.

The 26B A4B startup additionally performed a first-run FlashInfer CUTLASS MoE
JIT build; its `nvcc`/`cicc` evidence remains in the captured server log. A
separate follow-up design for routing telemetry and active-parameter efficiency
is recorded in `quality/MOE_EVALUATION_PLAN.md` and was not mixed into this
campaign.

## 11. Execution record for the current campaign

This section is an append-only operational record for the campaign started on
2026-08-08 UTC. It contains no API token, private key, or secret environment
value.

### Provisioning

The initially supplied Pod (`5g92476znggs8c`) was stopped and could not be
restarted because its host had no free GPUs. It also did not satisfy this spec:
its volume was 50 GB and its image reported CUDA 12.8.1. A compliant replacement
was provisioned with:

| Field | Recorded value |
| --- | --- |
| Pod ID | `321hxgl8vi7a5q` |
| Name | `gemma4-nvfp4-benchmark` |
| Cloud | Community Cloud, France (`FR`) |
| GPU | 1 × NVIDIA GeForce RTX 5090 |
| GPU price | USD 0.69/hour |
| Image | `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster` |
| Container disk | 40 GB |
| Volume | 80 GB mounted at `/workspace` |
| SSH endpoint | `root@80.15.7.37`, TCP port `40974` |
| Ports | `8888/http`, `22/tcp` |
| Safety guard | automatic stop five hours after provisioning |

The preflight reported RTX 5090, compute capability 12.0, 32,607 MiB VRAM,
driver 580.159.03, an XFS `/workspace` volume with 80 GB available, and a 40 GB
container filesystem. The remote Pod did not expose `RUNPOD_POD_ID`; the CLI Pod
ID above is authoritative.

### Remote environment

The durable environment is `/workspace/gemma4-benchmark`. Python 3.13.14 was
installed with uv and the venv is at `/workspace/gemma4-benchmark/env`. Resolved
versions are Python 3.13.14, PyTorch 2.11.0 (CUDA 13.0), vLLM 0.26.0,
`flashinfer-python` 0.6.14, and `nvidia-cutlass-dsl` 4.6.0. `uv pip check`
passed for all 211 packages. The runner is uploaded at
`config/benchmark_runner.py`; raw output, logs, and summaries remain under the
documented `/workspace/gemma4-benchmark` tree.

The CUDA runtime for FlashInfer JIT compilation is explicitly pinned to the
image’s system toolkit with `CUDA_HOME=/usr/local/cuda-13.0` and
`PATH=/usr/local/cuda-13.0/bin:...`. This also makes the venv’s `ninja`
executable visible.

### Compatibility findings and changes

The first attempted forced policy, `--linear-backend flashinfer_b12x
--moe-backend flashinfer_b12x`, failed before benchmarking because vLLM reported
that no b12x kernel exists for the checkpoint’s FP8 linear layer. The next
attempt, forcing FlashInfer CUTLASS for both linear and MoE layers, correctly
identified that the checkpoint’s FP8 scales were not per-tensor. Both failures
were preserved in the server log and classified as `kernel_selection` failures.

The active policy is now `--linear-backend auto --moe-backend flashinfer_cutlass`.
The first checkpoint’s successful load log proves `CutlassFP8ScaledMMLinearKernel`
for its FP8 layers and `FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM. The
FlashInfer JIT initially failed because the pip CUDA compiler and image toolkit
headers were mismatched; pinning the system CUDA 13.0 toolkit resolved that
failure. No Marlin, emulation, CPU offload, or OOM path has been used.

### Current state

E4B and 12B completed their five warm-ups plus three interactive and three
throughput repetitions with zero failed requests. The 26B checkpoint downloaded
15.75 GiB, loaded in approximately 16.35 GiB of GPU memory, selected
`FlashInferCutlassNvFp4LinearKernel` and `FLASHINFER_CUTLASS`, and passed model
loading without OOM. Its first startup is longer because FlashInfer compiles
specialized SM120 CUTLASS MoE kernels and vLLM compiles the execution graph;
this compilation time is startup overhead, not benchmark latency. The runner
must wait for `/health` before beginning warm-ups. Measured JSON and the
generated Markdown report are collected only after the runner finishes or the
budget guard stops it.

The corrected 26B-only rerun then completed all five warm-ups and six measured
repetitions with zero failures. Its medians were 215.365 output tokens/s
interactive and 1,762.931 output tokens/s at concurrency 16. The original
26B gate failure was a false positive caused by matching `MARLIN` in vLLM's
list of potential backends; the runner now ignores that list and checks only
actual selected fallback/error lines. The final local snapshot is
`benchmarks/20260808T173800Z/`.

Reusable, non-destructive helpers are included at `scripts/`: run
`remote_preflight.sh` through the Pod's SSH command before setup, and use
`collect_results.sh` to copy durable artifacts. Neither helper stops or
terminates a Pod.

## References

* [vLLM `bench serve` CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/)
* [vLLM serve kernel backends](https://docs.vllm.ai/en/latest/cli/serve/)
* [vLLM FlashInfer NVFP4 kernels](https://docs.vllm.ai/en/latest/api/vllm/model_executor/kernels/linear/nvfp4/flashinfer/)
* [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html)
* [RunPod pricing and storage](https://docs.runpod.io/pods/pricing)
* [RunPod Pod management](https://docs.runpod.io/pods/manage-pods)
