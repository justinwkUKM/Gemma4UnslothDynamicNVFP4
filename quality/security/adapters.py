#!/usr/bin/env python3
"""Streaming adapters for the actual OTRF, LANL, and OpTC source formats."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import tarfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from quality.security.parser import LABEL_FIELDS, TelemetryError, parse_timestamp


LANL_COLUMNS = {
    "auth": (
        "time", "source_user", "destination_user", "source_computer",
        "destination_computer", "authentication_type", "logon_type",
        "orientation", "result",
    ),
    "proc": ("time", "user", "computer", "process", "lifecycle"),
    "flows": (
        "time", "duration", "source_computer", "source_port",
        "destination_computer", "destination_port", "protocol",
        "packet_count", "byte_count",
    ),
    "dns": ("time", "source_computer", "resolved_computer"),
    "redteam": ("time", "user", "source_computer", "destination_computer"),
}
LANL_ANCHOR = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _stable_id(dataset: str, source_name: str, line_number: int) -> str:
    digest = hashlib.sha256(f"{source_name}\0{line_number}".encode()).hexdigest()[:20]
    return f"{dataset}-{digest}"


def _attributes(record: dict[str, Any], consumed: Iterable[str]) -> dict[str, Any]:
    omitted = set(consumed)
    return {
        key: value
        for key, value in record.items()
        if key not in omitted and key.lower() not in LABEL_FIELDS
    }


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return path.open(encoding="utf-8", errors="replace", newline="")


def iter_json_objects(path: Path) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """Yield JSON objects from JSONL, gzip JSONL, or each JSONL member of a zip."""

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                name for name in archive.namelist()
                if not name.endswith("/") and Path(name).suffix.lower() in {".json", ".jsonl", ".log"}
            )
            if not members:
                raise TelemetryError(f"{path}: zip has no JSON/JSONL/log members")
            for member in members:
                with archive.open(member) as raw:
                    handle = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                    yield from _iter_json_handle(handle, f"{path.name}!{member}")
        return
    if path.suffix == ".tar":
        with tarfile.open(path, "r:*") as archive:
            members = sorted(
                (member for member in archive.getmembers() if member.isfile()),
                key=lambda member: member.name,
            )
            accepted = 0
            for member in members:
                lowered = member.name.lower()
                if not lowered.endswith((".json", ".jsonl", ".log", ".json.gz", ".jsonl.gz", ".log.gz")):
                    continue
                raw = archive.extractfile(member)
                if raw is None:
                    continue
                accepted += 1
                binary = gzip.GzipFile(fileobj=raw) if lowered.endswith(".gz") else raw
                with binary:
                    handle = io.TextIOWrapper(binary, encoding="utf-8", errors="replace")
                    yield from _iter_json_handle(handle, f"{path.name}!{member.name}")
            if not accepted:
                raise TelemetryError(f"{path}: tar has no JSON/JSONL/log members")
        return
    with _open_text(path) as handle:
        yield from _iter_json_handle(handle, path.name)


def _iter_json_handle(handle: TextIO, source_name: str) -> Iterator[tuple[str, int, dict[str, Any]]]:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TelemetryError(f"{source_name}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise TelemetryError(f"{source_name}:{line_number}: record must be an object")
        yield source_name, line_number, value


def _otrf_timestamp(record: dict[str, Any]) -> str:
    for key in ("@timestamp", "timestamp", "ts", "EventTime", "EventReceivedTime"):
        if key in record and record[key] not in (None, ""):
            return parse_timestamp(record[key]).isoformat()
    raise TelemetryError("OTRF record has no recognized timestamp")


def adapt_otrf(path: Path) -> Iterator[dict[str, Any]]:
    for source_name, line_number, record in iter_json_objects(path):
        entity = next((record.get(key) for key in (
            "Hostname", "hostname", "Computer", "computer", "host", "id_orig_h", "id_resp_h"
        ) if record.get(key) not in (None, "")), "unknown-entity")
        source_type = next((record.get(key) for key in (
            "Channel", "@stream", "SourceName", "source_type"
        ) if record.get(key) not in (None, "")), "unknown")
        category = next((record.get(key) for key in (
            "action", "EventType", "Category", "Opcode", "@stream"
        ) if record.get(key) not in (None, "")), None)
        event_number = record.get("EventID")
        action = category or (f"windows_event_{event_number}" if event_number is not None else "observed")
        consumed = {
            "@timestamp", "timestamp", "ts", "EventTime", "EventReceivedTime",
            "Hostname", "hostname", "Computer", "computer", "host", "Channel",
            "@stream", "SourceName", "source_type", "action", "EventType", "Category", "Opcode",
        }
        yield {
            "schema_version": 1,
            "event_id": _stable_id("otrf", source_name, line_number),
            "timestamp": _otrf_timestamp(record),
            "dataset": "otrf",
            "source_type": str(source_type),
            "entity_id": str(entity),
            "action": str(action),
            "outcome": str(record.get("EventType") or record.get("status") or "unknown"),
            "attributes": _attributes(record, consumed),
        }


def _lanl_timestamp(value: str) -> str:
    try:
        seconds = int(value)
    except ValueError as exc:
        raise TelemetryError(f"invalid LANL elapsed timestamp: {value!r}") from exc
    return (LANL_ANCHOR + timedelta(seconds=seconds - 1)).isoformat()


def adapt_lanl(path: Path, stream: str) -> Iterator[dict[str, Any]]:
    if stream not in LANL_COLUMNS:
        raise TelemetryError(f"unknown LANL stream {stream!r}")
    columns = LANL_COLUMNS[stream]
    with _open_text(path) as handle:
        for line_number, values in enumerate(csv.reader(handle), 1):
            if not values:
                continue
            if len(values) != len(columns):
                raise TelemetryError(
                    f"{path}:{line_number}: expected {len(columns)} fields for {stream}, got {len(values)}"
                )
            record = dict(zip(columns, values))
            entity = record.get("source_computer") or record.get("computer") or record.get("user") or "unknown-entity"
            if stream == "auth":
                action, outcome = f"authentication:{record['orientation']}", record["result"]
            elif stream == "proc":
                action, outcome = f"process:{record['lifecycle']}", "observed"
            elif stream == "flows":
                action, outcome = "network_flow", "observed"
            elif stream == "dns":
                action, outcome = "dns_lookup", "observed"
            else:
                action, outcome = "authentication_compromise", "known_redteam"
            yield {
                "schema_version": 1,
                "event_id": _stable_id("lanl", path.name, line_number),
                "timestamp": _lanl_timestamp(record["time"]),
                "dataset": "lanl",
                "source_type": stream,
                "entity_id": str(entity),
                "action": action,
                "outcome": outcome,
                "attributes": _attributes(record, {"time"}),
            }


def adapt_optc(path: Path) -> Iterator[dict[str, Any]]:
    for source_name, line_number, record in iter_json_objects(path):
        raw_timestamp = record.get("timestamp_ms", record.get("timestamp"))
        if raw_timestamp in (None, ""):
            raise TelemetryError(f"{source_name}:{line_number}: missing eCAR timestamp")
        number = float(raw_timestamp)
        if number > 10_000_000_000:
            number /= 1000.0
        timestamp = datetime.fromtimestamp(number, tz=timezone.utc).isoformat()
        entity = record.get("hostname") or record.get("host") or record.get("actorID") or "unknown-entity"
        consumed = {
            "timestamp", "timestamp_ms", "id", "hostname", "host", "object",
            "object_type", "action", "status",
        }
        yield {
            "schema_version": 1,
            "event_id": str(record.get("id") or _stable_id("optc", source_name, line_number)),
            "timestamp": timestamp,
            "dataset": "darpa-optc",
            "source_type": str(record.get("object") or record.get("object_type") or "unknown"),
            "entity_id": str(entity),
            "action": str(record.get("action") or "observed"),
            "outcome": str(record.get("status") or "unknown"),
            "attributes": _attributes(record, consumed),
        }
