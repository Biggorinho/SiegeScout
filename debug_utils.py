"""Debug snapshot recorder for the SiegeScout bot.

Builds a 4-panel mosaic per event (annotated minimap, enemy color mask,
walkable mask, full game frame with overlays), rate-limited per event type
and rotated by file count. All I/O lives here so the rest of the loop stays
free of disk concerns.
"""

from __future__ import annotations

import glob
import os
import time

import cv2
import numpy as np


class DebugRecorder:
    def __init__(
        self,
        enabled: bool,
        debug_dir: str,
        periodic_interval: float,
        max_files: int,
        event_rate_limit_seconds: float,
    ) -> None:
        self._enabled = bool(enabled)
        self._debug_dir = str(debug_dir)
        self._periodic_interval = float(periodic_interval)
        self._max_files = int(max_files)
        self._event_rate_limit_seconds = float(event_rate_limit_seconds)
        self._last_event_at: dict[str, float] = {}
        self._last_periodic_at: float = 0.0
        self._disabled_due_to_error = False
        self._disk_error_logged = False
        if self._enabled:
            try:
                os.makedirs(self._debug_dir, exist_ok=True)
            except OSError as exc:
                self._handle_disk_error(exc)

    def maybe_record(
        self,
        now: float,
        event: str,
        frame_bgr,
        minimap_bgr,
        center,
        enemy_dots,
        item_dots,
        ore_dots,
        companion_dot,
        nearest_enemy,
        resource_target,
        walkable_mask,
        enemy_color_mask,
        state_label: str,
        hp_percent,
        counters_snapshot: dict,
    ) -> str | None:
        if not self._enabled or self._disabled_due_to_error:
            return None
        # Rate-limit per event type.
        last = self._last_event_at.get(event, -1e9)
        if (now - last) < self._event_rate_limit_seconds:
            return None
        path = self._write(now, event, frame_bgr, minimap_bgr, center,
                           enemy_dots, item_dots, ore_dots,
                           companion_dot, nearest_enemy, resource_target,
                           walkable_mask, enemy_color_mask,
                           state_label, hp_percent, counters_snapshot)
        if path is not None:
            self._last_event_at[event] = now
        return path

    def force_record(
        self,
        now: float,
        event: str,
        frame_bgr,
        minimap_bgr,
        center,
        enemy_dots,
        item_dots,
        ore_dots,
        companion_dot,
        nearest_enemy,
        resource_target,
        walkable_mask,
        enemy_color_mask,
        state_label: str,
        hp_percent,
        counters_snapshot: dict,
    ) -> str | None:
        """Bypass rate limit (used for periodic ticks the recorder itself owns)."""
        if not self._enabled or self._disabled_due_to_error:
            return None
        return self._write(now, event, frame_bgr, minimap_bgr, center,
                           enemy_dots, item_dots, ore_dots,
                           companion_dot, nearest_enemy, resource_target,
                           walkable_mask, enemy_color_mask,
                           state_label, hp_percent, counters_snapshot)

    def _write(self, now, event, frame_bgr, minimap_bgr, center,
               enemy_dots, item_dots, ore_dots,
               companion_dot, nearest_enemy, resource_target,
               walkable_mask, enemy_color_mask,
               state_label, hp_percent, counters_snapshot) -> str | None:
        # Mosaic builder lands in Task 2; for now write the full frame only
        # so the scaffold tests have something to check.
        stamp = int(now * 1000)
        filename = f"{event}_{stamp}.png"
        path = os.path.join(self._debug_dir, filename)
        try:
            ok = cv2.imwrite(path, frame_bgr)
            if not ok:
                raise OSError(f"cv2.imwrite returned False for {path}")
        except OSError as exc:
            self._handle_disk_error(exc)
            return None
        self._rotate()
        return path

    def _rotate(self) -> None:
        try:
            files = sorted(
                glob.glob(os.path.join(self._debug_dir, "*.png")),
                key=lambda p: os.path.getmtime(p),
            )
            excess = len(files) - self._max_files
            for old in files[:max(0, excess)]:
                try:
                    os.remove(old)
                except OSError:
                    pass  # in-use, skip; next rotation will retry
        except OSError:
            pass  # listing failed (e.g. dir vanished) — non-fatal

    def _handle_disk_error(self, exc: Exception) -> None:
        self._disabled_due_to_error = True
        if not self._disk_error_logged:
            print(f"[DEBUG] disk error: {exc} — snapshots disabled")
            self._disk_error_logged = True
