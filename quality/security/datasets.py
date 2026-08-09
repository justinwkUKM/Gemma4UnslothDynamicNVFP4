#!/usr/bin/env python3
"""Acquire-status, normalize, anonymize, and inventory security datasets."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from campaigns.common import atomic_write_json, atomic_write_jsonl, sha256_file, utc_now  # noqa: E402
from quality.security.adapters import adapt_lanl, adapt_optc, adapt_otrf  # noqa: E402
from quality.security.parser import (  # noqa: E402
    TelemetryError,
    anonymized_event_id,
    iter_anonymized_public_events,
    parse_timestamp,
)


LOCK_PATH = Path(__file__).with_name("dataset_lock.json")
SCENARIO_PATH = Path(__file__).with_name("scenarios") / "otrf-attacks-v1.json"
OPTC_MANIFEST_PATH = Path(__file__).with_name("optc-corrected-v1.json")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _git_revision(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _tree_stats(path: Path) -> dict[str, int]:
    files = [item for item in path.rglob("*") if item.is_file() and ".git" not in item.parts]
    return {"file_count": len(files), "bytes": sum(item.stat().st_size for item in files)}


def acquire_git_source(data_root: Path, source_name: str) -> dict[str, Any]:
    lock = _read_json(LOCK_PATH)["sources"]
    if source_name == "otrf":
        repository = lock["otrf"]["repository"]
        revision = lock["otrf"]["revision"]
        destination = data_root / lock["otrf"]["local_path"]
    else:
        repository = lock["optc"]["documentation_repository"]
        revision = lock["optc"]["documentation_revision"]
        destination = data_root / lock["optc"]["documentation_local_path"]
    actual = _git_revision(destination)
    if actual:
        if actual != revision:
            raise ValueError(
                f"{destination} already exists at {actual}; expected {revision}; refusing to modify it"
            )
        return {"source": source_name, "destination": str(destination), "revision": actual, "changed": False}
    if destination.exists():
        raise ValueError(f"{destination} exists but is not a readable git checkout")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", repository, str(destination)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", revision],
        check=True,
    )
    actual = _git_revision(destination)
    if actual != revision:
        raise ValueError(f"checkout verification failed: expected {revision}, got {actual}")
    return {
        "source": source_name,
        "destination": str(destination),
        "revision": actual,
        "changed": True,
        "tree": _tree_stats(destination),
    }


def build_status(data_root: Path) -> dict[str, Any]:
    lock = _read_json(LOCK_PATH)
    sources = lock["sources"]
    otrf_path = data_root / sources["otrf"]["local_path"]
    optc_docs_path = data_root / sources["optc"]["documentation_local_path"]
    lanl_path = data_root / sources["lanl"]["local_path"]
    optc_path = data_root / sources["optc"]["local_path"]
    otrf_revision = _git_revision(otrf_path)
    optc_revision = _git_revision(optc_docs_path)
    scenarios = _read_json(SCENARIO_PATH)["scenarios"]
    scenario_missing = [item["source"] for item in scenarios if not (otrf_path / item["source"]).is_file()]
    otrf_preparation_path = data_root / "canonical" / "otrf-attacks-v1" / "preparation-manifest.json"
    otrf_preparation = _read_json(otrf_preparation_path) if otrf_preparation_path.is_file() else None
    lanl_required = sources["lanl"]["required_files"]
    lanl_present = [name for name in lanl_required if (lanl_path / name).is_file()]
    optc_archives = sorted(item.name for item in optc_path.glob("*.tar")) if optc_path.exists() else []
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "gpu_required": False,
        "datasets": {
            "otrf": {
                "acquisition": "downloaded" if otrf_revision else "missing",
                "revision_expected": sources["otrf"]["revision"],
                "revision_actual": otrf_revision,
                "revision_verified": otrf_revision == sources["otrf"]["revision"],
                "tree": _tree_stats(otrf_path) if otrf_path.exists() else None,
                "selected_attack_scenarios": len(scenarios),
                "selected_sources_missing": scenario_missing,
                "canonical_prepared": bool(
                    otrf_preparation
                    and otrf_preparation.get("source_revision") == sources["otrf"]["revision"]
                    and otrf_preparation.get("scenario_count") == len(scenarios)
                    and "limit" in otrf_preparation
                    and otrf_preparation.get("limit") is None
                ),
                "prepared_event_count": (
                    sum(item.get("event_count", 0) for item in otrf_preparation.get("scenarios", []))
                    if otrf_preparation else 0
                ),
                "scored_inference_ready": False,
                "blockers": [
                    "benign scenario controls are not yet selected",
                    "event-level attack ground truth requires curation",
                ],
            },
            "lanl": {
                "acquisition": "complete" if len(lanl_present) == len(lanl_required) else "manual_download_required",
                "present_files": lanl_present,
                "missing_files": [name for name in lanl_required if name not in lanl_present],
                "adapter_ready": True,
                "acquisition_ready": len(lanl_present) == len(lanl_required),
                "scored_inference_ready": False,
                "blockers": (
                    ["red-team events must be grouped into incident-level evaluation truth"]
                    if len(lanl_present) == len(lanl_required)
                    else ["LANL requires the requestor's email and intended use in its download form"]
                ),
            },
            "optc": {
                "documentation_revision_expected": sources["optc"]["documentation_revision"],
                "documentation_revision_actual": optc_revision,
                "documentation_verified": optc_revision == sources["optc"]["documentation_revision"],
                "corrected_archive_manifest_ready": OPTC_MANIFEST_PATH.is_file(),
                "corrected_archives_present": optc_archives,
                "adapter_ready": True,
                "scored_inference_ready": False,
                "blockers": [
                    "corrected daily archives require an explicit storage budget and file selection",
                    "corrected dataset custom terms require review",
                    "multi-stage event-level truth has not been curated",
                ],
            },
            "cyber_range": {
                "acquisition": "not_generated_by_design",
                "generation_spec_ready": (REPO_ROOT / sources["cyber_range"]["specification"]).is_file(),
                "scored_inference_ready": False,
                "blockers": ["telemetry must be collected after model and prompt freeze"],
            },
        },
    }


def _limited(events: Iterable[dict[str, Any]], limit: int | None) -> Iterator[dict[str, Any]]:
    for index, event in enumerate(events):
        if limit is not None and index >= limit:
            break
        yield event


def _prepare_stream(
    events: Iterable[dict[str, Any]], destination: Path, *, salt: str,
    source: Path, dataset: str, limit: int | None = None,
) -> dict[str, Any]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    iterator = iter(_limited(events, limit))
    try:
        first_event = next(iterator)
    except StopIteration as exc:
        raise TelemetryError("source contains no adaptable telemetry events") from exc
    source_events = itertools.chain((first_event,), iterator)
    counters: dict[str, Any] = {"count": 0, "first": None, "last": None}

    def checked() -> Iterator[dict[str, Any]]:
        previous = None
        for event in iter_anonymized_public_events(source_events, salt=salt):
            timestamp = parse_timestamp(event["timestamp"])
            if previous is not None and timestamp < previous:
                raise TelemetryError("source events are not timestamp-ordered")
            previous = timestamp
            counters["count"] += 1
            counters["first"] = counters["first"] or event["timestamp"]
            counters["last"] = event["timestamp"]
            yield event

    atomic_write_jsonl(destination, checked())
    provenance = {
        "schema_version": 1,
        "prepared_at_utc": utc_now(),
        "dataset": dataset,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(destination),
        "event_count": counters["count"],
        "first_timestamp": counters["first"],
        "last_timestamp": counters["last"],
        "labels_stripped": True,
        "identifiers_and_timestamps_remapped": True,
        "salt_sha256": hashlib.sha256(salt.encode()).hexdigest(),
        "limit": limit,
    }
    atomic_write_json(destination.with_suffix(destination.suffix + ".provenance.json"), provenance)
    return provenance


def _salt(name: str) -> str:
    value = os.environ.get(name)
    if not value or len(value) < 16:
        raise ValueError(f"{name} must contain at least 16 characters")
    return value


def prepare_otrf(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _read_json(args.scenarios)
    source_root = args.data_root / "sources" / "otrf"
    revision = _git_revision(source_root)
    if revision != manifest["source_revision"]:
        raise ValueError(
            f"OTRF revision mismatch: expected {manifest['source_revision']}, got {revision or 'missing'}"
        )
    salt = _salt(args.salt_env)
    prepared = []
    for scenario in manifest["scenarios"]:
        source = source_root / scenario["source"]
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = args.output / f"{scenario['scenario_id']}.jsonl"
        # OTRF archives can interleave collector arrival times. Sort each bounded
        # laboratory scenario before replay; enterprise-scale adapters stay streaming.
        ordered_events = sorted(
            adapt_otrf(source),
            key=lambda event: (parse_timestamp(event["timestamp"]), event["event_id"]),
        )
        provenance = _prepare_stream(
            ordered_events, destination, salt=salt, source=source, dataset="otrf",
            limit=args.limit,
        )
        prepared.append({
            "scenario_id": scenario["scenario_id"],
            "metadata_id": scenario["metadata_id"],
            "output": destination.name,
            "event_count": provenance["event_count"],
            "output_sha256": provenance["output_sha256"],
        })
    result = {
        "schema_version": 1,
        "prepared_at_utc": utc_now(),
        "source_revision": revision,
        "scenario_manifest_sha256": sha256_file(args.scenarios),
        "scenario_count": len(prepared),
        "limit": args.limit,
        "scored_inference_allowed": False,
        "reason": "event-level truth and benign controls require curation",
        "scenarios": prepared,
    }
    atomic_write_json(args.output / "preparation-manifest.json", result)
    return result


def prepare_one(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset == "otrf":
        events = adapt_otrf(args.input)
    elif args.dataset == "lanl":
        if not args.stream:
            raise ValueError("--stream is required for LANL")
        if args.stream == "redteam":
            raise ValueError("redteam.txt.gz is sealed ground truth and cannot be prepared as inference telemetry")
        events = adapt_lanl(args.input, args.stream)
    else:
        events = adapt_optc(args.input)
    return _prepare_stream(
        events, args.output, salt=_salt(args.salt_env), source=args.input,
        dataset=args.dataset, limit=args.limit,
    )


def _lanl_truth_key(event: dict[str, Any], *, redteam: bool) -> tuple[str, str, str, str]:
    attributes = event["attributes"]
    user_key = "user" if redteam else "source_user"
    return (
        event["timestamp"],
        str(attributes.get(user_key, "")),
        str(attributes.get("source_computer", "")),
        str(attributes.get("destination_computer", "")),
    )


def prepare_lanl(args: argparse.Namespace) -> dict[str, Any]:
    files = {stream: args.input / f"{stream}.txt.gz" for stream in ("auth", "proc", "flows", "dns", "redteam")}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing LANL files: " + ", ".join(missing))
    redteam_events = list(adapt_lanl(files["redteam"], "redteam"))
    redteam_keys = {_lanl_truth_key(event, redteam=True) for event in redteam_events}
    matched_raw_ids: list[str] = []
    matched_redteam_keys: set[tuple[str, str, str, str]] = set()

    def auth_events() -> Iterator[dict[str, Any]]:
        for event in adapt_lanl(files["auth"], "auth"):
            truth_key = _lanl_truth_key(event, redteam=False)
            if truth_key in redteam_keys:
                matched_raw_ids.append(event["event_id"])
                matched_redteam_keys.add(truth_key)
            yield event

    streams = [auth_events()] + [adapt_lanl(files[name], name) for name in ("proc", "flows", "dns")]
    merged = heapq.merge(
        *streams,
        key=lambda event: (parse_timestamp(event["timestamp"]), event["event_id"]),
    )
    salt = _salt(args.salt_env)
    provenance = _prepare_stream(
        merged, args.output, salt=salt, source=files["auth"], dataset="lanl", limit=args.limit,
    )
    matched = sorted(anonymized_event_id(event_id, salt) for event_id in matched_raw_ids)
    truth = {
        "schema_version": 1,
        "dataset": "lanl",
        "ground_truth_status": (
            "prefix_only_redteam_matching_incomplete" if args.limit is not None
            else "redteam_events_matched_incident_grouping_required"
        ),
        "scored_inference_allowed": False,
        "preparation_limit": args.limit,
        "redteam_source_event_count": len(redteam_events),
        "matched_auth_event_count": len(matched),
        "matched_redteam_key_count": len(matched_redteam_keys),
        "unmatched_redteam_event_count": (
            None if args.limit is not None else len(redteam_keys - matched_redteam_keys)
        ),
        "attack_event_ids": matched,
    }
    atomic_write_json(args.ground_truth_output, truth)
    manifest = {
        "schema_version": 1,
        "prepared_at_utc": utc_now(),
        "output": str(args.output),
        "ground_truth_output": str(args.ground_truth_output),
        "source_files": {
            name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in files.items()
        },
        "provenance": provenance,
        "truth_summary": {key: truth[key] for key in (
            "ground_truth_status", "preparation_limit", "redteam_source_event_count",
            "matched_auth_event_count", "matched_redteam_key_count",
            "unmatched_redteam_event_count", "scored_inference_allowed",
        )},
    }
    atomic_write_json(args.output.with_suffix(args.output.suffix + ".lanl-manifest.json"), manifest)
    return manifest


def download_optc(args: argparse.Namespace) -> dict[str, Any]:
    if not args.accept_custom_terms:
        raise ValueError("--accept-custom-terms is required after reviewing the corrected dataset terms")
    manifest = _read_json(OPTC_MANIFEST_PATH)
    choices = {item["name"]: item for item in manifest["archives"]}
    archive = choices.get(args.archive)
    if archive is None:
        raise ValueError(f"unknown archive {args.archive!r}")
    if args.max_bytes < archive["bytes"]:
        raise ValueError(
            f"--max-bytes {args.max_bytes} is below the pinned archive size {archive['bytes']}"
        )
    destination = args.data_root / "raw" / "optc-corrected" / archive["name"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    def md5_file(path: Path) -> str:
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    if destination.exists():
        if destination.stat().st_size == archive["bytes"] and md5_file(destination) == archive["md5"]:
            return {
                "archive": archive["name"],
                "bytes": archive["bytes"],
                "md5": archive["md5"],
                "destination": str(destination),
                "verified": True,
                "changed": False,
            }
        raise ValueError(f"{destination} exists but does not match the pinned size/checksum; refusing to overwrite")
    free = shutil.disk_usage(destination.parent).free
    required = int(archive["bytes"] * 1.1)
    if free < required:
        raise ValueError(f"insufficient free space: need {required} bytes including safety margin, have {free}")
    if temporary.exists() and temporary.stat().st_size > archive["bytes"]:
        raise ValueError(f"{temporary} exceeds the pinned archive size; refusing to resume")
    url = f"https://entrepot.recherche.data.gouv.fr/api/access/datafile/{archive['datafile_id']}"
    subprocess.run(
        [
            "curl", "--fail", "--location", "--continue-at", "-",
            "--max-filesize", str(args.max_bytes), "--output", str(temporary), url,
        ],
        check=True,
    )
    if temporary.stat().st_size != archive["bytes"]:
        raise ValueError("downloaded archive size does not match the pinned manifest")
    if md5_file(temporary) != archive["md5"]:
        raise ValueError("downloaded archive MD5 does not match the repository metadata")
    temporary.replace(destination)
    return {
        "archive": archive["name"],
        "bytes": archive["bytes"],
        "md5": archive["md5"],
        "destination": str(destination),
        "verified": True,
        "changed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "security-data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Write a local acquisition/readiness inventory")
    status.add_argument("--output", type=Path)

    subparsers.add_parser("acquire-otrf", help="Clone and verify the pinned OTRF source tree")
    subparsers.add_parser("acquire-optc-docs", help="Clone and verify pinned OpTC documentation")

    otrf = subparsers.add_parser("prepare-otrf", help="Prepare the pinned ten-scenario OTRF attack set")
    otrf.add_argument("--scenarios", type=Path, default=SCENARIO_PATH)
    otrf.add_argument("--output", type=Path, required=True)
    otrf.add_argument("--salt-env", default="SECURITY_BENCHMARK_SALT")
    otrf.add_argument("--limit", type=int)

    lanl = subparsers.add_parser("prepare-lanl", help="Merge LANL streams and seal matched red-team truth")
    lanl.add_argument("--input", type=Path, required=True)
    lanl.add_argument("--output", type=Path, required=True)
    lanl.add_argument("--ground-truth-output", type=Path, required=True)
    lanl.add_argument("--salt-env", default="SECURITY_BENCHMARK_SALT")
    lanl.add_argument("--limit", type=int)

    optc_download = subparsers.add_parser("download-optc", help="Download one explicitly budgeted corrected archive")
    optc_download.add_argument("--archive", required=True)
    optc_download.add_argument("--max-bytes", type=int, required=True)
    optc_download.add_argument("--accept-custom-terms", action="store_true")

    one = subparsers.add_parser("prepare", help="Stream one source file into anonymized canonical JSONL")
    one.add_argument("--dataset", choices=("otrf", "lanl", "optc"), required=True)
    one.add_argument("--input", type=Path, required=True)
    one.add_argument("--output", type=Path, required=True)
    one.add_argument("--stream", choices=("auth", "proc", "flows", "dns", "redteam"))
    one.add_argument("--salt-env", default="SECURITY_BENCHMARK_SALT")
    one.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            result = build_status(args.data_root)
            if args.output:
                atomic_write_json(args.output, result)
        elif args.command == "acquire-otrf":
            result = acquire_git_source(args.data_root, "otrf")
        elif args.command == "acquire-optc-docs":
            result = acquire_git_source(args.data_root, "optc")
        elif args.command == "prepare-otrf":
            result = prepare_otrf(args)
        elif args.command == "prepare-lanl":
            result = prepare_lanl(args)
        elif args.command == "download-optc":
            result = download_optc(args)
        else:
            result = prepare_one(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError, TelemetryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
