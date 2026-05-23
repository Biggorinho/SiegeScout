"""Minimap-driven resource targeting (ore pickaxes + item stars).

The class is intentionally I/O-free: callers pass already-classified dot
lists from `detectors.classify_item_dots` / `classify_ore_dots`, and the
class returns the active target. Pickup itself is handled by the existing
on-screen logic in main.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResourceTarget:
    pos: tuple[int, int]        # minimap coordinates
    kind: str                   # "item" | "ore"
    acquired_at: float          # time.time() when selected
    close_since: float | None   # first time within pickup distance, else None


class ResourceTargeter:
    def __init__(
        self,
        memory_seconds: float = 12.0,
        max_travel_seconds: float = 8.0,
        spot_match_radius_px: int = 8,
        close_attempt_seconds: float = 4.0,
    ) -> None:
        self._memory_seconds = float(memory_seconds)
        self._max_travel_seconds = float(max_travel_seconds)
        self._spot_match_radius_px = int(spot_match_radius_px)
        self._close_attempt_seconds = float(close_attempt_seconds)
        self._active: ResourceTarget | None = None
        self._visited: list[tuple[tuple[int, int], float]] = []

    @property
    def active(self) -> ResourceTarget | None:
        return self._active

    def reset(self) -> None:
        self._active = None
        self._visited = []

    def release(self) -> None:
        """Drop the current target without adding it to visited (e.g. enemy
        pre-empts, or the dot legitimately vanished from the minimap)."""
        self._active = None
