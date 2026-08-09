#!/usr/bin/env python3
"""Strict structured-output contract with evidence linkage."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


class ContractError(ValueError):
    pass


CLAIM_COLLECTIONS = {
    "observations": "summary",
    "hypotheses": "description",
    "attack_techniques": "technique_id",
    "recommendations": "action",
    "predicted_next_actions": "action",
}
REQUIRED = {
    "status",
    "risk_score",
    "confidence",
    *CLAIM_COLLECTIONS,
    "related_entities",
    "missing_information",
}


def parse_model_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        fenced = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```\s*(?:<turn\|>)?",
            stripped,
            flags=re.I | re.S,
        )
        if fenced is None:
            raise ContractError("model output contains text outside the JSON code fence")
        stripped = fenced.group(1)
    elif stripped.endswith("<turn|>"):
        stripped = stripped[: -len("<turn|>")].rstrip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ContractError("model output must be a JSON object")
    return value


def validate_analysis(value: Mapping[str, Any], context_event_ids: set[str]) -> dict[str, Any]:
    missing = REQUIRED - value.keys()
    if missing:
        raise ContractError("missing fields: " + ", ".join(sorted(missing)))
    if value["status"] not in {"benign", "suspicious", "malicious", "insufficient_evidence"}:
        raise ContractError("invalid status")
    if not isinstance(value["risk_score"], int) or not 0 <= value["risk_score"] <= 100:
        raise ContractError("risk_score must be an integer from 0 to 100")
    if not isinstance(value["confidence"], (int, float)) or not 0 <= value["confidence"] <= 1:
        raise ContractError("confidence must be between 0 and 1")
    unsupported_evidence: set[str] = set()
    claim_count = 0
    for collection, text_key in CLAIM_COLLECTIONS.items():
        items = value[collection]
        if not isinstance(items, list):
            raise ContractError(f"{collection} must be a list")
        for item in items:
            claim_count += 1
            if not isinstance(item, dict) or not isinstance(item.get(text_key), str) or not item[text_key].strip():
                raise ContractError(f"{collection} item requires {text_key}")
            evidence = item.get("evidence_ids")
            if not isinstance(evidence, list) or not evidence or not all(isinstance(event_id, str) for event_id in evidence):
                raise ContractError(f"every {collection} item requires evidence_ids")
            unsupported_evidence.update(set(evidence) - context_event_ids)
            if "confidence" in item and (not isinstance(item["confidence"], (int, float)) or not 0 <= item["confidence"] <= 1):
                raise ContractError(f"{collection} confidence must be between 0 and 1")
    for key in ("related_entities", "missing_information"):
        if not isinstance(value[key], list) or not all(isinstance(item, str) for item in value[key]):
            raise ContractError(f"{key} must be a list of strings")
    return {
        "valid": not unsupported_evidence,
        "unsupported_evidence_ids": sorted(unsupported_evidence),
        "claim_count": claim_count,
    }
