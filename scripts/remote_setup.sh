#!/usr/bin/env bash
set -Eeuo pipefail

# Idempotent dependency/bootstrap step for a fresh RunPod. This prepares the
# persistent volume only; it does not download models, start vLLM, or stop a Pod.
root_dir="${GEMMA_ROOT:-/workspace/gemma4-benchmark}"
python_dir="${UV_PYTHON_INSTALL_DIR:-/workspace/python}"
uv_dir="${UV_INSTALL_DIR:-/workspace/bin}"
export HF_HOME="$root_dir/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export UV_CACHE_DIR="$root_dir/cache/uv"
export UV_PYTHON_INSTALL_DIR="$python_dir"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$root_dir/env/bin:$uv_dir:$PATH"

mkdir -p "$root_dir"/{env,cache/huggingface,cache/uv,config,logs,results,environment,summary}
command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
if ! command -v uv >/dev/null 2>&1; then
  mkdir -p "$uv_dir"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$uv_dir" sh
fi
export PATH="$uv_dir:$PATH"
command -v uv >/dev/null

uv python install 3.13
if [[ ! -x "$root_dir/env/bin/python" ]]; then
  uv venv --python 3.13 "$root_dir/env"
fi
uv pip install --python "$root_dir/env/bin/python" \
  'vllm[bench]>=0.25.0' 'flashinfer-python>=0.6.13' 'nvidia-cutlass-dsl>=4.5.2'

"$root_dir/env/bin/python" -m pip check | tee "$root_dir/environment/pip-check.txt"
"$root_dir/env/bin/python" - <<'PY' | tee "$root_dir/environment/versions.txt"
import importlib.metadata as metadata
import platform, sys
print(f"python=={sys.version}")
print(f"platform=={platform.platform()}")
for package in ("torch", "vllm", "flashinfer-python", "nvidia-cutlass-dsl"):
    try: print(f"{package}=={metadata.version(package)}")
    except metadata.PackageNotFoundError: print(f"{package}==MISSING")
PY
"$root_dir/env/bin/python" -m pip freeze > "$root_dir/environment/packages.txt"
echo "setup complete: $root_dir"
