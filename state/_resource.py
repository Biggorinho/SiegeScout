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

    def mark_attempted(self, now: float) -> None:
        """Treat the current target as a failed/attempted pickup. Stores its
        position in visited so we don't re-select it until memory expires."""
        if self._active is None:
            return
        self._visited.append((self._active.pos, now + self._memory_seconds))
        self._active = None

    def tick(
        self,
        now: float,
        center: tuple[int, int],
        item_dots: list,
        ore_dots: list,
        close_threshold_px: int | None = None,
    ) -> ResourceTarget | None:
        """Update or pick the active target. Returns the active target, or None.

        When `close_threshold_px` is given, the targeter also tracks how long the
        player has been within that radius and auto-marks the target attempted
        after `close_attempt_seconds`.
        """
        if self._active is not None:
            self._active = self._reconfirm_or_drop(now, item_dots, ore_dots)

        if self._active is not None and close_threshold_px is not None:
            self._update_close_since(now, center, close_threshold_px)
            # _update_close_since may have nulled self._active via auto-attempt.

        if self._active is None:
            self._active = self._pick_new(now, center, item_dots, ore_dots)
        return self._active

    def is_close(self, center: tuple[int, int], threshold_px: int) -> bool:
        if self._active is None:
            return False
        dx = self._active.pos[0] - center[0]
        dy = self._active.pos[1] - center[1]
        return (dx * dx + dy * dy) <= (threshold_px * threshold_px)

    def _update_close_since(
        self,
        now: float,
        center: tuple[int, int],
        threshold_px: int,
    ) -> None:
        target = self._active
        assert target is not None
        dx = target.pos[0] - center[0]
        dy = target.pos[1] - center[1]
        within = (dx * dx + dy * dy) <= (threshold_px * threshold_px)
        if within:
            if target.close_since is None:
                target.close_since = now
            elif (now - target.close_since) >= self._close_attempt_seconds:
                self._visited.append((target.pos, now + self._memory_seconds))
                self._active = None
        else:
            # Walked back out of range — reset the counter.
            target.close_since = None

    def _reconfirm_or_drop(
        self,
        now: float,
        item_dots: list,
        ore_dots: list,
    ) -> ResourceTarget | None:
        target = self._active
        assert target is not None

        # Hard travel timeout — record as attempted so we don't pick it again.
        if (now - target.acquired_at) >= self._max_travel_seconds:
            self._visited.append((target.pos, now + self._memory_seconds))
            return None

        # Still on the minimap (within match radius)?
        dots = item_dots if target.kind == "item" else ore_dots
        if not self._dot_present(dots, target.pos):
            return None  # disappeared — release without visited entry

        return target

    def _dot_present(self, dots: list, pos: tuple[int, int]) -> bool:
        r2 = self._spot_match_radius_px * self._spot_match_radius_px
        for d in dots:
            if (int(d[0]) - pos[0]) ** 2 + (int(d[1]) - pos[1]) ** 2 <= r2:
                return True
        return False

    def _pick_new(
        self,
        now: float,
        center: tuple[int, int],
        item_dots: list,
        ore_dots: list,
    ) -> ResourceTarget | None:
        self._prune_visited(now)
        for kind, dots in (("item", item_dots), ("ore", ore_dots)):
            best = self._nearest_unvisited(dots, center)
            if best is not None:
                x, y = best
                return ResourceTarget(
                    pos=(int(x), int(y)),
                    kind=kind,
                    acquired_at=now,
                    close_since=None,
                )
        return None

    def _nearest_unvisited(self, dots: list, center: tuple[int, int]) -> tuple[int, int] | None:
        if not dots:
            return None
        cx, cy = center
        candidates = [d for d in dots if not self._is_visited((int(d[0]), int(d[1])))]
        if not candidates:
            return None
        best = min(candidates, key=lambda d: (d[0] - cx) ** 2 + (d[1] - cy) ** 2)
        return (int(best[0]), int(best[1]))

    def _is_visited(self, pos: tuple[int, int]) -> bool:
        r2 = self._spot_match_radius_px * self._spot_match_radius_px
        for vpos, _expires in self._visited:
            if (vpos[0] - pos[0]) ** 2 + (vpos[1] - pos[1]) ** 2 <= r2:
                return True
        return False

    def _prune_visited(self, now: float) -> None:
        self._visited = [(p, exp) for (p, exp) in self._visited if exp > now]
