# Gemma 4 benchmark snapshot — partial

Captured from RunPod Pod `321hxgl8vi7a5q` at 2026-08-08T17:03:24Z while the
remote runner was still active. This snapshot is intentionally partial; later
commits will add the remaining model results and generated summary.

Included:

- `config/`: generated runner/configuration;
- `environment/`: Python, package, runtime, and benchmark CLI reports;
- `logs/`: server and GPU-monitor logs captured so far;
- `results/`: raw detailed vLLM JSON and normalized per-run records available at
  capture time.

Excluded deliberately: `.env`, API tokens, SSH keys, model/Hugging Face caches,
the Python virtual environment, and other credentials or transient build
artifacts.

The active Pod uses an RTX 5090 Community Cloud instance with CUDA 13.0, a
40 GB container disk, an 80 GB `/workspace` volume, and a five-hour automatic
stop guard. The authoritative procedure and troubleshooting history are in the
repository-root benchmark specification.
