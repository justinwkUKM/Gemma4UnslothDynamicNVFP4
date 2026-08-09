#!/usr/bin/env python3
"""Deterministic timestamp replay and batching."""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Any, Iterable, Iterator

from .parser import parse_timestamp


class ReplayEngine:
    SPEEDS = (0.5, 1, 5, 10, 20, 50)

    def __init__(self, events: Iterable[dict[str, Any]], *, speed: float = 1.0, wall_clock: bool = False):
        if speed <= 0:
            raise ValueError("replay speed must be positive")
        self.events = sorted(events, key=lambda item: (item["timestamp"], item["event_id"]))
        self.speed = speed
        self.wall_clock = wall_clock

    def events_with_clock(self) -> Iterator[tuple[float, dict[str, Any]]]:
        if not self.events:
            return
        origin = parse_timestamp(self.events[0]["timestamp"])
        previous_offset = 0.0
        for event in self.events:
            offset = (parse_timestamp(event["timestamp"]) - origin).total_seconds() / self.speed
            if self.wall_clock and offset > previous_offset:
                time.sleep(offset - previous_offset)
            yield offset, event
            previous_offset = offset

    def windows(self, seconds: float) -> Iterator[list[dict[str, Any]]]:
        if seconds <= 0:
            raise ValueError("window must be positive")
        if not self.events:
            return
        start = parse_timestamp(self.events[0]["timestamp"])
        end = start + timedelta(seconds=seconds)
        batch: list[dict[str, Any]] = []
        for _, event in self.events_with_clock():
            timestamp = parse_timestamp(event["timestamp"])
            while timestamp >= end:
                if batch:
                    yield batch
                    batch = []
                start = end
                end = start + timedelta(seconds=seconds)
            batch.append(event)
        if batch:
            yield batch
