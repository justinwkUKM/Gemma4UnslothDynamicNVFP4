#!/usr/bin/env python3
"""Canonical telemetry parsing and deterministic public-data anonymization."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from campaigns.common import atomic_write_jsonl


LABEL_FIELDS = {
    "label",
    "labels",
    "ground_truth",
    "attack",
    "attack_name",
    "malicious",
    "is_malicious",
    "scenario_label",
    "red_team",
}
SOURCE_MARKER_FIELDS = {"tags", "dataset_name", "collection_name", "scenario_name"}
ALIASES = {
    "event_id": ("event_id", "id", "event_uid", "EventID"),
    "timestamp": ("timestamp", "time", "event_time", "datetime", "TimeCreated"),
    "source_type": ("source_type", "source", "log_type", "event_source", "channel"),
    "entity_id": ("entity_id", "host", "hostname", "computer", "user", "principal"),
    "action": ("action", "event_type", "activity", "operation", "message"),
    "outcome": ("outcome", "result", "status"),
}
IDENTITY_KEY = re.compile(r"(?:user|account|principal|host|computer|device|asset)(?:name|id)?$", re.I)
IP_KEY = re.compile(r"(?:src|dst|source|destination)?_?ip(?:_address)?$", re.I)
PATH_KEY = re.compile(r"(?:file|file_path|path|image_path|process_path)$", re.I)
IDENTITY_NAMES = {
    "account", "accountname", "actorid", "computer", "computername", "device",
    "deviceid", "domain", "entity", "entityid", "host", "hostname", "objectid",
    "principal", "remoteuserid", "sid", "user", "userid", "username",
}
IP_NAMES = {
    "destaddress", "destinationaddress", "destinationip", "dstaddress", "dstip",
    "idorigh", "idresph", "ip", "ipaddress", "sourceaddress", "sourceip",
    "srcaddress", "srcip",
}
PATH_NAMES = {
    "application", "file", "filename", "filepath", "image",
    "imagepath", "parentimagepath", "path", "process", "processname", "processpath",
}


class TelemetryError(ValueError):
    pass


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            try:
                parsed = datetime.fromtimestamp(float(text), tz=timezone.utc)
            except ValueError as exc:
                raise TelemetryError(f"invalid timestamp: {value!r}") from exc
    else:
        raise TelemetryError(f"invalid timestamp: {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _first(record: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


class CanonicalTelemetryParser:
    """Parse OTRF/LANL/OpTC or already-canonical JSON using field aliases."""

    def __init__(self, *, dataset: str, field_map: Mapping[str, str] | None = None):
        self.dataset = dataset
        self.field_map = dict(field_map or {})

    def parse(self, record: Mapping[str, Any], sequence: int) -> dict[str, Any]:
        def field(name: str) -> Any:
            explicit = self.field_map.get(name)
            return record.get(explicit) if explicit else _first(record, ALIASES[name])

        timestamp = parse_timestamp(field("timestamp"))
        raw_id = field("event_id")
        event_id = str(raw_id) if raw_id not in (None, "") else f"{self.dataset}-{sequence:09d}"
        entity_id = field("entity_id")
        action = field("action")
        if entity_id in (None, "") or action in (None, ""):
            raise TelemetryError(f"event {event_id}: entity_id and action are required")
        consumed = {
            alias
            for canonical, aliases in ALIASES.items()
            for alias in aliases
            if (self.field_map.get(canonical) or alias) in record
        }
        consumed.update(self.field_map.values())
        attributes = {
            key: value
            for key, value in record.items()
            if key not in consumed and key.lower() not in LABEL_FIELDS
        }
        return {
            "schema_version": 1,
            "event_id": event_id,
            "timestamp": timestamp.isoformat(),
            "dataset": self.dataset,
            "source_type": str(field("source_type") or "unknown"),
            "entity_id": str(entity_id),
            "action": str(action),
            "outcome": str(field("outcome") or "unknown"),
            "attributes": attributes,
        }

    def parse_jsonl(self, source: Path) -> list[dict[str, Any]]:
        events = []
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TelemetryError(f"{source}:{line_number}: record must be an object")
                events.append(self.parse(value, line_number))
        events.sort(key=lambda item: (item["timestamp"], item["event_id"]))
        if len({item["event_id"] for item in events}) != len(events):
            raise TelemetryError("event IDs are not unique")
        return events


def _token(prefix: str, value: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}\0{prefix}\0{value}".encode()).hexdigest()[:12]
    return f"{prefix.upper()}-{digest}"


def anonymized_event_id(value: str, salt: str) -> str:
    return _token("event", value, salt)


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _value_kind(key: str, value: str) -> str | None:
    normalized = _normalized_key(key)
    if (
        normalized in IDENTITY_NAMES
        or normalized.endswith((
            "username", "userid", "accountname", "accountid", "hostname",
            "computername", "domainname", "principalname", "deviceid", "objectid", "actorid",
        ))
        or IDENTITY_KEY.search(key)
    ):
        return "entity"
    if normalized in IP_NAMES or IP_KEY.search(key):
        try:
            ipaddress.ip_address(value)
            return "ip"
        except ValueError:
            return "entity"
    if normalized in PATH_NAMES or PATH_KEY.search(key):
        return "artifact"
    return None


def _anonymize_value(key: str, value: Any, salt: str) -> Any:
    if isinstance(value, dict):
        return {
            child_key: _anonymize_value(child_key, child_value, salt)
            for child_key, child_value in value.items()
            if child_key.lower() not in LABEL_FIELDS | SOURCE_MARKER_FIELDS
        }
    if isinstance(value, list):
        return [_anonymize_value(key, item, salt) for item in value]
    if not isinstance(value, str):
        return value
    kind = _value_kind(key, value)
    if kind:
        return _token(kind, value, salt)
    return value


def _collect_sensitive(value: Any, salt: str, key: str = "") -> dict[str, str]:
    replacements: dict[str, str] = {}
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            replacements.update(_collect_sensitive(child_value, salt, child_key))
    elif isinstance(value, list):
        for child in value:
            replacements.update(_collect_sensitive(child, salt, key))
    elif isinstance(value, str) and len(value) >= 3:
        kind = _value_kind(key, value)
        if kind:
            replacements[value] = _token(kind, value, salt)
    return replacements


def _replace_sensitive(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _replace_sensitive(child, replacements) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_sensitive(child, replacements) for child in value]
    if isinstance(value, str):
        result = value
        for original in sorted(replacements, key=len, reverse=True):
            result = result.replace(original, replacements[original])
        return result
    return value


def anonymize_public_event(
    event: Mapping[str, Any], *, salt: str, first_timestamp: datetime
) -> dict[str, Any]:
    """Anonymize one canonical event without retaining a corpus in memory."""

    timestamp = datetime(2030, 1, 1, tzinfo=timezone.utc) + (
        parse_timestamp(event["timestamp"]) - first_timestamp
    )
    replacements = _collect_sensitive(event, salt)
    cleaned = {
        key: _anonymize_value(key, value, salt)
        for key, value in event.items()
        if key.lower() not in LABEL_FIELDS
    }
    cleaned = _replace_sensitive(cleaned, replacements)
    cleaned["event_id"] = anonymized_event_id(str(event["event_id"]), salt)
    cleaned["entity_id"] = _token("entity", str(event["entity_id"]), salt)
    cleaned["timestamp"] = timestamp.isoformat()
    cleaned["dataset"] = "public-telemetry"
    cleaned["public_data_anonymized"] = True
    return cleaned


def iter_anonymized_public_events(
    events: Iterable[Mapping[str, Any]], *, salt: str
) -> Iterator[dict[str, Any]]:
    """Stream deterministic anonymization for chronologically ordered events."""

    iterator = iter(events)
    try:
        first_event = next(iterator)
    except StopIteration:
        return
    first_timestamp = parse_timestamp(first_event["timestamp"])
    yield anonymize_public_event(first_event, salt=salt, first_timestamp=first_timestamp)
    for event in iterator:
        yield anonymize_public_event(event, salt=salt, first_timestamp=first_timestamp)


def anonymize_public_events(events: Iterable[Mapping[str, Any]], *, salt: str) -> list[dict[str, Any]]:
    """Remap identifiers and timestamps while preserving event intervals."""

    values = [dict(event) for event in events]
    if not values:
        return []
    first = min(parse_timestamp(event["timestamp"]) for event in values)
    anonymized = [
        anonymize_public_event(event, salt=salt, first_timestamp=first)
        for event in values
    ]
    return sorted(anonymized, key=lambda item: (item["timestamp"], item["event_id"]))


def prepare_public_jsonl(source: Path, destination: Path, *, dataset: str, salt: str, field_map: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    parser = CanonicalTelemetryParser(dataset=dataset, field_map=field_map)
    events = anonymize_public_events(parser.parse_jsonl(source), salt=salt)
    atomic_write_jsonl(destination, events)
    return events
