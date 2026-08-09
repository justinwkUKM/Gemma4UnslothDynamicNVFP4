#!/usr/bin/env python3
"""Prepare anonymized, label-free public telemetry and separate ground truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from campaigns.common import atomic_write_json, sha256_file, utc_now  # noqa: E402
from quality.security.parser import anonymized_event_id, prepare_public_jsonl  # noqa: E402


EVENT_ID_LIST_FIELDS = {
    "attack_event_ids",
    "contradicted_event_ids",
    "prompt_injection_event_ids",
    "evidence_ids",
}


def remap_truth_ids(value: Any, salt: str, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {child_key: remap_truth_ids(child, salt, child_key) for child_key, child in value.items()}
    if isinstance(value, list):
        if key in EVENT_ID_LIST_FIELDS:
            return [anonymized_event_id(str(item), salt) for item in value]
        return [remap_truth_ids(item, salt, key) for item in value]
    if key == "event_id" and isinstance(value, str):
        return anonymized_event_id(value, salt)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ground-truth-input", type=Path)
    parser.add_argument("--ground-truth-output", type=Path)
    parser.add_argument("--salt-env", default="SECURITY_BENCHMARK_SALT")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.ground_truth_input) != bool(args.ground_truth_output):
        print("error: ground-truth input and output must be provided together", file=sys.stderr)
        return 2
    salt = os.environ.get(args.salt_env)
    if not salt or len(salt) < 16:
        print(f"error: {args.salt_env} must contain at least 16 characters", file=sys.stderr)
        return 2
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if config.get("track") != "public" or config.get("anonymization_required") is not True:
            raise ValueError("prepare accepts only public configs requiring anonymization")
        events = prepare_public_jsonl(
            args.input,
            args.output,
            dataset=config["dataset"],
            salt=salt,
            field_map=config.get("field_map"),
        )
        truth_hash = None
        if args.ground_truth_input:
            truth = json.loads(args.ground_truth_input.read_text(encoding="utf-8"))
            remapped = remap_truth_ids(truth, salt)
            atomic_write_json(args.ground_truth_output, remapped)
            truth_hash = sha256_file(args.ground_truth_output)
        provenance = {
            "schema_version": 1,
            "prepared_at_utc": utc_now(),
            "dataset": config["dataset"],
            "config_sha256": sha256_file(args.config),
            "source_sha256": sha256_file(args.input),
            "output_sha256": sha256_file(args.output),
            "ground_truth_output_sha256": truth_hash,
            "event_count": len(events),
            "labels_stripped": True,
            "identifiers_and_timestamps_remapped": True,
            "salt_sha256": hashlib.sha256(salt.encode()).hexdigest(),
        }
        atomic_write_json(args.output.with_suffix(args.output.suffix + ".provenance.json"), provenance)
        print(json.dumps(provenance, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
