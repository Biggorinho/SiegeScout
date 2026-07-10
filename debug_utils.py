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
        panel_height: int = 360,
    ) -> None:
        self._enabled = bool(enabled)
        self._debug_dir = str(debug_dir)
        self._periodic_interval = float(periodic_interval)
        self._max_files = int(max_files)
        self._event_rate_limit_seconds = float(event_rate_limit_seconds)
        self._panel_height = int(panel_height)
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
        stamp = int(now * 1000)
        try:
            img = self._build_mosaic(
                frame_bgr, minimap_bgr, center,
                enemy_dots, item_dots, ore_dots,
                companion_dot, nearest_enemy, resource_target,
                walkable_mask, enemy_color_mask,
                state_label, hp_percent, counters_snapshot, event,
            )
            filename = f"{event}_{stamp}.png"
        except Exception as exc:
            print(f"[DEBUG] mosaic build failed: {type(exc).__name__}: {exc}")
            img = frame_bgr
            filename = f"{event}_fallback_{stamp}.png"
        path = os.path.join(self._debug_dir, filename)
        try:
            ok = cv2.imwrite(path, img)
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

    def _build_mosaic(
        self,
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
        event: str,
    ):
        # Build the four panels, then assemble in 2x2 grid.
        p1 = self._panel_minimap(minimap_bgr, center, enemy_dots, item_dots,
                                  ore_dots, companion_dot, nearest_enemy,
                                  resource_target, state_label, hp_percent, event)
        p2 = self._panel_mask(enemy_color_mask, "ENEMY MASK")
        p3 = self._panel_mask(walkable_mask, "WALKABLE")
        p4 = self._panel_frame(frame_bgr, counters_snapshot)

        # Resize all to uniform panel height while preserving aspect ratio.
        panels = [self._resize_to_height(p, self._panel_height)
                  for p in (p1, p2, p3, p4)]
        # Pad each panel to the max width in its row, so concat works.
        top_w = max(panels[0].shape[1], panels[1].shape[1])
        bot_w = max(panels[2].shape[1], panels[3].shape[1])
        row_w = max(top_w, bot_w)
        for i in range(4):
            panels[i] = self._pad_to_width(panels[i], row_w)
        top = cv2.hconcat([panels[0], panels[1]])
        bot = cv2.hconcat([panels[2], panels[3]])
        return cv2.vconcat([top, bot])

    @staticmethod
    def _resize_to_height(img, target_h):
        if img is None or img.size == 0:
            return np.zeros((target_h, target_h, 3), dtype=np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        if h == target_h:
            return img
        scale = target_h / float(h)
        new_w = max(1, int(w * scale))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _pad_to_width(img, target_w):
        h, w = img.shape[:2]
        if w == target_w:
            return img
        pad = np.zeros((h, target_w - w, 3), dtype=np.uint8)
        return cv2.hconcat([img, pad])

    @staticmethod
    def _panel_minimap(minimap_bgr, center, enemy_dots, item_dots, ore_dots,
                       companion_dot, nearest_enemy, resource_target,
                       state_label, hp_percent, event):
        if minimap_bgr is None or minimap_bgr.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        img = minimap_bgr.copy()
        cx, cy = int(center[0]), int(center[1])
        # Companion (magenta ring)
        if companion_dot is not None:
            cv2.circle(img, (int(companion_dot[0]), int(companion_dot[1])), 8, (255, 0, 255), 2)
        # Enemy dots colored by tier (skull=white, yellow=cyan, purple=magenta, red=red)
        tier_color = {1: (255, 255, 255), 2: (255, 255, 0), 3: (255, 0, 255), 4: (0, 0, 255)}
        for d in enemy_dots:
            x, y = int(d[0]), int(d[1])
            tier = d[2] if len(d) > 2 else 1
            cv2.circle(img, (x, y), 5, tier_color.get(tier, (0, 255, 0)), 1)
        # Item stars (yellow filled cross)
        for d in item_dots:
            cv2.drawMarker(img, (int(d[0]), int(d[1])), (0, 255, 255),
                           cv2.MARKER_STAR, 8, 1)
        # Ore pickaxes (gold square)
        for d in ore_dots:
            cv2.drawMarker(img, (int(d[0]), int(d[1])), (0, 165, 255),
                           cv2.MARKER_SQUARE, 8, 2)
        # Active resource target — cyan line from player
        if resource_target is not None:
            tx, ty = int(resource_target.pos[0]), int(resource_target.pos[1])
            cv2.line(img, (cx, cy), (tx, ty), (255, 220, 0), 1)
            cv2.circle(img, (tx, ty), 6, (255, 220, 0), 2)
        # Nearest enemy (red ring)
        if nearest_enemy is not None:
            cv2.circle(img, (int(nearest_enemy[0]), int(nearest_enemy[1])),
                       9, (0, 0, 255), 2)
        # Player cross (orange)
        cv2.drawMarker(img, (cx, cy), (255, 100, 0), cv2.MARKER_CROSS, 14, 2)
        # Watermark
        hp_txt = f"{hp_percent:.0%}" if hp_percent is not None else "?"
        cv2.putText(img, f"{state_label} HP={hp_txt} {event}", (4, 14),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 220, 255), 1)
        return img

    @staticmethod
    def _panel_mask(mask, label):
        if mask is None or getattr(mask, "size", 0) == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        if mask.ndim == 2:
            img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            img = mask.copy()
        cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_PLAIN, 0.9,
                    (0, 220, 255), 1)
        return img

    @staticmethod
    def _panel_frame(frame_bgr, counters_snapshot):
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        img = frame_bgr.copy()
        h, w = img.shape[:2]
        # Counter watermark across bottom
        y = h - 6
        text = " ".join(f"{k}={v}" for k, v in counters_snapshot.items())
        if text:
            cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (0, 220, 255), 1)
        return img


class FrameCapture:
    """Saves raw, unannotated frame + minimap pairs for the offline eval harness.

    Unlike DebugRecorder mosaics, these images carry no overlays so they can
    be labeled and scored (see eval/). Capped by stopping at max_pairs rather
    than rotating: rotation would delete early-session zones and destroy
    dataset diversity. Fails soft on disk errors, like DebugRecorder.
    """

    def __init__(
        self,
        enabled: bool,
        base_dir: str,
        interval_seconds: float,
        max_pairs: int,
    ) -> None:
        self._enabled = bool(enabled)
        self._base_dir = str(base_dir)
        self._interval = float(interval_seconds)
        self._max_pairs = int(max_pairs)
        self._session_dir: str | None = None
        self._last_save_at: float = -1e9
        self._saved_pairs = 0
        self._disabled_due_to_error = False
        self._cap_logged = False

    def maybe_save(self, now: float, frame_bgr, minimap_bgr) -> str | None:
        """Save a frame+minimap pair if the interval elapsed. Returns frame path."""
        if not self._enabled or self._disabled_due_to_error:
            return None
        if self._saved_pairs >= self._max_pairs:
            if not self._cap_logged:
                print(f"[CAPTURE] reached {self._max_pairs} pairs — capture stopped")
                self._cap_logged = True
            return None
        if (now - self._last_save_at) < self._interval:
            return None
        if frame_bgr is None or minimap_bgr is None:
            return None
        if getattr(frame_bgr, "size", 0) == 0 or getattr(minimap_bgr, "size", 0) == 0:
            return None
        try:
            if self._session_dir is None:
                session = time.strftime("session_%Y%m%d-%H%M%S")
                self._session_dir = os.path.join(self._base_dir, session)
                os.makedirs(self._session_dir, exist_ok=True)
            stamp = int(now * 1000)
            frame_path = os.path.join(self._session_dir, f"frame_{stamp}.png")
            minimap_path = os.path.join(self._session_dir, f"minimap_{stamp}.png")
            for path, img in ((frame_path, frame_bgr), (minimap_path, minimap_bgr)):
                if not cv2.imwrite(path, img):
                    raise OSError(f"cv2.imwrite returned False for {path}")
        except OSError as exc:
            self._disabled_due_to_error = True
            print(f"[CAPTURE] disk error: {exc} — capture disabled")
            return None
        self._last_save_at = now
        self._saved_pairs += 1
        return frame_path
