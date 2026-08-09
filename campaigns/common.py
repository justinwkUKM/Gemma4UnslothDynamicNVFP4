#!/usr/bin/env python3
"""Dependency-free provenance and persistence helpers shared by campaigns."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_STATUSES = {"planned", "running", "partial", "complete", "failed"}
TERMINAL_STATUSES = {"partial", "complete", "failed"}


class ManifestError(ValueError):
    """Raised when campaign provenance is incomplete or contradictory."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL record is not an object")
            records.append(value)
    return records


def _package_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _command_output(command: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, timeout=10, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def capture_environment(*, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Capture an allow-listed environment inventory without credentials."""

    gpu_query = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "captured_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": _package_versions(
            ("vllm", "torch", "transformers", "flashinfer-python", "nvidia-cutlass-dsl")
        ),
        "gpu": {"nvidia_smi_query": gpu_query},
    }
    if extra:
        value["campaign"] = dict(extra)
    value["environment_sha256"] = canonical_sha256(value)
    return value


def hash_artifacts(root: Path, *, exclude: Iterable[Path] = ()) -> dict[str, str]:
    """Hash all files below root using portable POSIX relative paths."""

    excluded = {path.resolve() for path in exclude}
    hashes: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() in excluded or path.name.endswith(".tmp"):
            continue
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


@dataclass
class CampaignManifest:
    """Mutable status around immutable campaign identity and provenance."""

    path: Path
    data: dict[str, Any]

    REQUIRED = {
        "campaign_id",
        "campaign_type",
        "dataset_versions",
        "model_versions",
        "backend",
        "context_limit",
        "seed",
        "prompt_hash",
        "environment_hash",
        "gpu_type",
        "started_at_utc",
        "deadline_utc",
        "hourly_rate_usd",
        "status",
    }

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        campaign_id: str,
        campaign_type: str,
        dataset_versions: Mapping[str, str],
        model_versions: Mapping[str, str | None],
        backend: str | Mapping[str, Any],
        context_limit: int,
        seed: int,
        prompt_hash: str,
        environment_hash: str,
        gpu_type: str | None,
        started_at_utc: str,
        deadline_utc: str,
        hourly_rate_usd: float,
        config_hash: str,
        status: str = "running",
        extra: Mapping[str, Any] | None = None,
    ) -> "CampaignManifest":
        data: dict[str, Any] = {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "campaign_type": campaign_type,
            "dataset_versions": dict(dataset_versions),
            "model_versions": dict(model_versions),
            "backend": backend,
            "context_limit": context_limit,
            "seed": seed,
            "prompt_hash": prompt_hash,
            "environment_hash": environment_hash,
            "gpu_type": gpu_type,
            "started_at_utc": started_at_utc,
            "deadline_utc": deadline_utc,
            "hourly_rate_usd": hourly_rate_usd,
            "config_sha256": config_hash,
            "status": status,
            "finished_at_utc": None,
            "requirements": {},
            "artifacts": {},
        }
        if extra:
            data.update(dict(extra))
        if path.exists():
            existing = cls.load(path)
            immutable = ("campaign_type", "context_limit", "seed", "prompt_hash", "config_sha256")
            changed = [key for key in immutable if existing.data.get(key) != data.get(key)]
            if changed:
                raise ManifestError("cannot resume campaign with changed identity fields: " + ", ".join(changed))
            existing.data["status"] = "running"
            existing.data["finished_at_utc"] = None
            existing.data["resume_count"] = int(existing.data.get("resume_count", 0)) + 1
            existing.data["resumed_at_utc"] = utc_now()
            existing.save()
            return existing
        manifest = cls(path=path, data=data)
        manifest.validate()
        manifest.save()
        return manifest

    @classmethod
    def load(cls, path: Path) -> "CampaignManifest":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ManifestError("campaign manifest must be a JSON object")
        manifest = cls(path=path, data=value)
        manifest.validate()
        return manifest

    def validate(self) -> None:
        missing = sorted(self.REQUIRED - self.data.keys())
        if missing:
            raise ManifestError("campaign manifest is missing: " + ", ".join(missing))
        if self.data["status"] not in MANIFEST_STATUSES:
            raise ManifestError(f"invalid campaign status: {self.data['status']}")
        if not isinstance(self.data["context_limit"], int) or self.data["context_limit"] <= 0:
            raise ManifestError("context_limit must be a positive integer")
        if not isinstance(self.data["seed"], int):
            raise ManifestError("seed must be an integer")
        if not isinstance(self.data["hourly_rate_usd"], (int, float)) or self.data["hourly_rate_usd"] <= 0:
            raise ManifestError("hourly_rate_usd must be positive")
        for key in ("prompt_hash", "environment_hash", "config_sha256"):
            if not isinstance(self.data.get(key), str) or len(self.data[key]) != 64:
                raise ManifestError(f"{key} must be a SHA-256 hex digest")
        if self.data["status"] == "complete":
            requirements = self.data.get("requirements", {})
            if not requirements or not all(value is True for value in requirements.values()):
                raise ManifestError("complete status requires every declared requirement to pass")

    def save(self) -> None:
        self.validate()
        atomic_write_json(self.path, self.data)

    def update_backend(self, backend: str | Mapping[str, Any]) -> None:
        self.data["backend"] = backend
        self.save()

    def finish(
        self,
        status: str,
        *,
        requirements: Mapping[str, bool],
        artifact_root: Path | None = None,
        detail: str | None = None,
    ) -> None:
        if status not in TERMINAL_STATUSES:
            raise ManifestError(f"finish status must be one of {sorted(TERMINAL_STATUSES)}")
        if status == "complete" and (not requirements or not all(requirements.values())):
            raise ManifestError("cannot mark an incomplete campaign complete")
        self.data["status"] = status
        self.data["finished_at_utc"] = utc_now()
        self.data["requirements"] = dict(requirements)
        if detail:
            self.data["status_detail"] = detail
        if artifact_root:
            self.data["artifacts"] = hash_artifacts(artifact_root, exclude=(self.path,))
        self.save()
