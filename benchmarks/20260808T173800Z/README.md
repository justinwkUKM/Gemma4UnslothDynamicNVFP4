# Gemma 4 benchmark snapshot — final

Captured from RunPod Pod `321hxgl8vi7a5q` after the corrected 26B-only rerun at
2026-08-08T17:38Z. It includes the original E4B/12B campaign and the complete
26B campaign.

Included:

- `config/`: generated runner/configuration, including the corrected backend gate;
- `environment/`: Python, package, runtime, and benchmark CLI reports;
- `logs/`: server and GPU-monitor logs for each model;
- `results/`: raw detailed vLLM JSON and normalized per-run records;
- `summary/`: combined Markdown report for all three checkpoints.

Excluded deliberately: `.env`, API tokens, SSH keys, model/Hugging Face caches,
the Python virtual environment, and other credentials or transient build
artifacts. The 26B rerun reused cached weights and kernels; no model or
dependency reinstall was performed.
