# Security reasoning harness

This package implements the first replayable benchmark layer from
`../SECURITY_LLM_REASONING_BENCHMARK_PLAN.md`. It keeps ground truth outside
inference input and supports OTRF scenarios first, then the same canonical
schema for LANL, OpTC, and fresh cyber-range telemetry.

The implementation provides:

- canonical parsing, deterministic identifier/timestamp anonymization, and
  label stripping for public data;
- virtual or wall-clock timestamp replay;
- raw, windowed, triggered, stateful, and bounded tool-using modes;
- a strict JSON contract in which every claim cites context event IDs;
- resumable streaming inference with failed-request-only retries;
- separate Security Intelligence and Operational scorecards.

Ground-truth JSON is consumed only by `evaluator.py`, never `runner.py`. A
public-data campaign is not the unseen final track: publish both a reproducible
public campaign and a separately generated `--track unseen` cyber-range
campaign. Dataset acquisition is intentionally external so licenses and source
versions remain explicit.

Prepare public telemetry with the matching dataset config. The salt is read
from an environment variable and only its SHA-256 is recorded. When labels are
provided, their event IDs are remapped into a separate output so they still
match anonymized inference IDs:

```bash
export SECURITY_BENCHMARK_SALT='campaign-specific-random-value'
python3 -m quality.security.prepare \
  --input raw.jsonl --output canonical-anonymized.jsonl \
  --config quality/security/configs/otrf.json \
  --ground-truth-input raw-labels.json \
  --ground-truth-output anonymized-labels.json
```

Generate contexts without using a model or GPU:

```bash
python3 -m quality.security.runner \
  --events canonical-anonymized.jsonl --results-dir security/smoke \
  --dataset-version otrf-snapshot-SHA --scenario-id scenario-01 \
  --track public --model local-model --mode triggered --dry-run
```

Run inference only after local tests pass, then evaluate using a separate label
file:

```bash
python3 -m quality.security.evaluator \
  --results security/run/raw.jsonl --ground-truth labels.json \
  --output security/run/scorecards.json \
  --report security/run/security-report.md
```
