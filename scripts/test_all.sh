#!/usr/bin/env bash
set -Eeuo pipefail

# Mandatory local pre-GPU gate. Keep every suite explicit so package-layout
# differences cannot silently omit campaign tests.
python3 -m compileall -q campaigns benchmarks/qwen36_runner.py quality
python3 -m unittest discover -s campaigns/tests -v
python3 -m unittest discover -s benchmarks/tests -v
python3 -m unittest discover -s quality/tests -v
python3 -m unittest discover -s quality/security/tests -v
python3 quality/evaluator.py --validate-only
