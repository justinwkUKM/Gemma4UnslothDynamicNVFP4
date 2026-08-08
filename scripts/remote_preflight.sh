#!/usr/bin/env bash
set -Eeuo pipefail

# Read-only identity and capacity check. Run this through the SSH command
# supplied by RunPod before uploading the benchmark runner.
root_dir="${GEMMA_ROOT:-/workspace/gemma4-benchmark}"
printf 'utc='; date -u +%Y-%m-%dT%H:%M:%SZ
printf 'hostname='; hostname
printf 'runpod_pod_id=%s\n' "${RUNPOD_POD_ID:-unset}"
nvidia-smi -L
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
df -h "$root_dir" /tmp
python3 --version || true
if [[ -x "$root_dir/env/bin/python" ]]; then
  "$root_dir/env/bin/python" - <<'PY'
import importlib.metadata as m
for package in ("torch", "vllm", "flashinfer-python", "nvidia-cutlass-dsl"):
    try:
        print(f"{package}=={m.version(package)}")
    except m.PackageNotFoundError:
        print(f"{package}==MISSING")
PY
fi
