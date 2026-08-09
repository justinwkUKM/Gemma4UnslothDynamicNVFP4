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

The active matrix contains only Gemma 4 E4B, 12B, and 26B A4B. Run one
model-specific result tree at a time; Qwen3.6 35B is skipped.

Ground-truth JSON is consumed only by `evaluator.py`, never `runner.py`. A
public-data campaign is not the unseen final track: publish both a reproducible
public campaign and a separately generated `--track unseen` cyber-range
campaign.

## Dataset preparation status

`dataset_lock.json` pins the sources and `datasets.py` records what is really
present. As of the current local preparation:

| Track | Local state | Scored inference ready? |
| --- | --- | --- |
| OTRF | Full pinned repository downloaded; ten attack archives normalized (78,980 events) | No: benign controls and event-level truth still need curation |
| LANL | Adapter and multi-stream merger ready; files not downloaded | No: the official form requires the requestor's email and intended use |
| corrected OpTC | Documentation, ground truth PDF, exact daily sizes/checksums, and tar adapter ready; no daily archive downloaded | No: 939,321,856,000 archive bytes exceed local storage and custom terms need review |
| unseen cyber-range | Collection/release specification ready | No: collection must occur after model/prompt freeze |

This distinction is enforced in manifests. Prepared OTRF and preliminary LANL
truth set `scored_inference_allowed` to `false`; scenario-level labels are not
silently treated as event-level ground truth.

Create or refresh a machine-readable status report without a GPU:

```bash
python3 -m quality.security.datasets status \
  --output security-data/manifests/readiness.json
```

## Acquire pinned public sources

The local data root is `security-data/` and is gitignored. These commands are
idempotent and refuse to modify a checkout at an unexpected revision:

```bash
python3 -m quality.security.datasets acquire-otrf
python3 -m quality.security.datasets acquire-optc-docs
```

Treat every downloaded security sample as untrusted. The preparation code only
opens supported telemetry containers and never executes binaries, scripts, or
commands contained in a dataset. Keep `security-data/` outside shared model
caches and application import paths.

The pinned OTRF revision has an upstream licensing inconsistency: its actual
`LICENSE` file is MIT, while an old README heading says GPL-3.0. The lock file
records both facts; do not silently rewrite the upstream terms.

LANL cannot be fetched responsibly without supplying the requestor's details.
Use the official form at <https://csr.lanl.gov/data/cyber1/>, then put exactly
`auth.txt.gz`, `proc.txt.gz`, `flows.txt.gz`, `dns.txt.gz`, and
`redteam.txt.gz` in `security-data/raw/lanl/`. Do not rename another LANL
dataset to those names.

The corrected OpTC V1 manifest is pinned in `optc-corrected-v1.json`. A
download requires all three explicit controls: one archive name, a byte budget
at least as large as that archive, and acknowledgement that the custom terms
were reviewed. The downloader verifies published size and MD5 and supports
resume through a `.part` file:

```bash
python3 -m quality.security.datasets download-optc \
  --archive 2019-09-16.tar --max-bytes 13437603840 \
  --accept-custom-terms
```

Run it only after reserving space for both the selected archive and its
canonical output. The full corrected set is about 939 GB of archives before
canonical output.

## Normalize and separate truth

Use a campaign-specific secret salt of at least 16 characters. Only its hash
is saved. Prepare the pinned OTRF attack scenarios with:

```bash
export SECURITY_BENCHMARK_SALT='campaign-specific-random-value'
python3 -m quality.security.datasets prepare-otrf \
  --output security-data/canonical/otrf-attacks-v1
```

After the five official LANL files are present, merge its four inference
streams and map the tiny red-team file into a separate sealed truth artifact:

```bash
python3 -m quality.security.datasets prepare-lanl \
  --input security-data/raw/lanl \
  --output security-data/canonical/lanl/canonical.jsonl \
  --ground-truth-output security-data/sealed/lanl-ground-truth.json
```

The LANL truth remains non-scorable until its red-team events are grouped into
incident scenarios. `redteam.txt.gz` is explicitly rejected by the generic
inference-data preparation path.

For a selected corrected OpTC archive, the tar adapter accepts JSON/JSONL/log
members, including gzip members:

```bash
python3 -m quality.security.datasets prepare \
  --dataset optc \
  --input security-data/raw/optc-corrected/2019-09-16.tar \
  --output security-data/canonical/optc/2019-09-16.jsonl
```

Every output has a provenance sidecar containing source/output hashes, event
count, time bounds, and label/anonymization flags. Public identifiers are
deterministically remapped both in structured fields and when repeated inside
free-text messages.

The older generic canonical-JSONL preparation command remains available for
small inputs that have already been converted into canonical aliases; it is
not the streaming path for raw public corpora. When labels are
provided, their event IDs are remapped into a separate output so they still
match anonymized inference IDs:

```bash
export SECURITY_BENCHMARK_SALT='campaign-specific-random-value'
python3 -m quality.security.prepare \
  --input pre-adapted-canonical.jsonl --output canonical-anonymized.jsonl \
  --config quality/security/configs/otrf.json \
  --ground-truth-input raw-labels.json \
  --ground-truth-output anonymized-labels.json
```

Generate contexts without using a model or GPU:

```bash
python3 -m quality.security.runner \
  --events canonical-anonymized.jsonl --results-dir security/smoke \
  --dataset-version otrf-snapshot-SHA --scenario-id scenario-01 \
  --track public --model unsloth/gemma-4-E4B-it-NVFP4 \
  --mode triggered --dry-run
```

Run inference only after local tests pass **and** the selected dataset manifest
says `scored_inference_allowed: true`. Then evaluate using a separate label
file:

```bash
python3 -m quality.security.evaluator \
  --results security/run/raw.jsonl --ground-truth labels.json \
  --output security/run/scorecards.json \
  --report security/run/security-report.md
```
