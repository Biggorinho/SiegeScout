"""Temporal dot confidence tracker.

Per-detector frame-stability tracker. Each dot position gets a confidence
that accumulates over consecutive frames and decays when missed. Returns
the input dots that have reached `confirm_threshold` this frame.

Replaces the simple integer counter `ENEMY_CONFIRM_FRAMES` with a per-spot
counter so single-frame mask jitter doesn't reset confirmation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Entry:
    confidence: int
    last_seen: float
    last_dot: list                # 4-tuple [x, y, tier, kind] from latest match


class TemporalDotConfidence:
    def __init__(
        self,
        match_radius_px: int,
        confirm_threshold: int,
        decay_per_miss: int = 1,
        max_confidence: int = 8,
    ) -> None:
        self._match_radius_px = int(match_radius_px)
        self._radius_sq = self._match_radius_px * self._match_radius_px
        self._confirm_threshold = int(confirm_threshold)
        self._decay_per_miss = int(decay_per_miss)
        self._max_confidence = int(max_confidence)
        self._entries: dict[tuple[int, int], _Entry] = {}

    def update(self, dots, now: float) -> list:
        """Returns the input dots whose entry confidence is >= confirm_threshold."""
        if dots is None:
            dots = []
        touched_keys: set = set()
        confirmed: list = []
        for d in dots:
            key = self._match(d)
            if key is None:
                key = (int(d[0]), int(d[1]))
                self._entries[key] = _Entry(confidence=1, last_seen=now, last_dot=list(d))
            else:
                entry = self._entries[key]
                entry.confidence = min(entry.confidence + 1, self._max_confidence)
                entry.last_seen = now
                entry.last_dot = list(d)
            touched_keys.add(key)
            if self._entries[key].confidence >= self._confirm_threshold:
                confirmed.append(list(self._entries[key].last_dot))
        # Decay untouched entries
        for k in list(self._entries.keys()):
            if k in touched_keys:
                continue
            self._entries[k].confidence -= self._decay_per_miss
            if self._entries[k].confidence <= 0:
                del self._entries[k]
        return confirmed

    def reset(self) -> None:
        self._entries.clear()

    def _match(self, dot) -> tuple[int, int] | None:
        x, y = int(dot[0]), int(dot[1])
        for key in self._entries.keys():
            dx = key[0] - x
            dy = key[1] - y
            if dx * dx + dy * dy <= self._radius_sq:
                return key
        return None
