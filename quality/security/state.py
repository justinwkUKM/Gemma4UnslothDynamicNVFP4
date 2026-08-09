#!/usr/bin/env python3
"""Bounded incident memory and evidence-safe state updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class IncidentState:
    incident_id: str
    entity_id: str
    risk: int = 0
    observations: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)
    updated_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def update(self, analysis: Mapping[str, Any], valid_event_ids: set[str], *, max_items: int = 100) -> None:
        grounded_observations = [
            item for item in analysis.get("observations", [])
            if set(item.get("evidence_ids", [])) <= valid_event_ids
        ]
        grounded_hypotheses = [
            item for item in analysis.get("hypotheses", [])
            if item.get("evidence_ids") and set(item["evidence_ids"]) <= valid_event_ids
        ]
        self.risk = max(0, min(100, int(analysis.get("risk_score", self.risk))))
        self.observations = (self.observations + grounded_observations)[-max_items:]
        self.hypotheses = grounded_hypotheses[-max_items:]
        evidence = [item for group in grounded_observations + grounded_hypotheses for item in group["evidence_ids"]]
        self.evidence_ids = list(dict.fromkeys((self.evidence_ids + evidence)[-max_items:]))
        self.related_entities = list(dict.fromkeys(analysis.get("related_entities", [])))[:max_items]


class InvestigationTools:
    """Read-only, allow-listed investigation queries over canonical events."""

    ALLOWED = {
        "get_process_tree",
        "get_auth_history",
        "get_connections",
        "get_dns_history",
        "get_asset_information",
        "get_related_alerts",
    }

    def __init__(self, events: list[dict[str, Any]], *, max_results: int = 50):
        self.events = events
        self.max_results = max_results

    def call(self, name: str, entity: str) -> list[dict[str, Any]]:
        if name not in self.ALLOWED:
            raise ValueError(f"tool {name!r} is not allowed")
        source_terms = {
            "get_process_tree": ("process", "endpoint"),
            "get_auth_history": ("auth", "identity"),
            "get_connections": ("network", "connection"),
            "get_dns_history": ("dns",),
            "get_asset_information": ("asset", "inventory"),
            "get_related_alerts": ("alert", "detection"),
        }[name]
        selected = [
            event for event in self.events
            if entity in {event.get("entity_id"), *event.get("attributes", {}).values()}
            and any(term in event.get("source_type", "").lower() for term in source_terms)
        ]
        return selected[-self.max_results:]
