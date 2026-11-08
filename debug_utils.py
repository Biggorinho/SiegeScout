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
        loot_enabled: bool = False,
        loot_interval_seconds: float = 1.5,
        loot_max: int = 300,
        loot_reason_max: dict | None = None,
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
        # Targeted loot-frame capture (own rate-limit / cap / counter).
        self._loot_enabled = bool(loot_enabled)
        self._loot_interval = float(loot_interval_seconds)
        self._loot_max = int(loot_max)
        self._loot_last_save: float = -1e9
        self._loot_saved = 0
        self._loot_cap_logged = False
        self._loot_reason_max = dict(loot_reason_max or {})
        self._loot_by_reason: dict[str, int] = {}
        self._loot_reason_capped: set = set()

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

    def save_loot_frame(self, now: float, frame_bgr, reason: str | None = None) -> str | None:
        """Save the full frame to a loot/ subdir when resources are on screen.

        A targeted dataset (full frames only, no minimap) for labeling an
        on-screen ore/item detector. Independent rate-limit, cap, and counter
        from the eval-pair capture; shares the session dir and fail-soft
        disable. Returns the saved path or None.
        """
        if not self._loot_enabled or self._disabled_due_to_error:
            return None
        if self._loot_saved >= self._loot_max:
            if not self._loot_cap_logged:
                print(f"[LOOTCAP] reached {self._loot_max} loot frames — stopped")
                self._loot_cap_logged = True
            return None
        # Per-trigger quota. Without it the common trigger starves the rare
        # one: the post-kill window fires on every kill, and at 51 zones/hour
        # it spent the whole 300-frame budget in the first 23 minutes of an
        # 11.6-hour run — so the remaining 11.2 hours, including every moment
        # the bot stood beside an item, recorded nothing at all.
        limit = self._loot_reason_max.get(reason)
        if limit is not None and self._loot_by_reason.get(reason, 0) >= limit:
            if reason not in self._loot_reason_capped:
                print(f"[LOOTCAP] '{reason}' quota {limit} reached — "
                      f"other triggers keep their budget")
                self._loot_reason_capped.add(reason)
            return None
        if (now - self._loot_last_save) < self._loot_interval:
            return None
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return None
        try:
            if self._session_dir is None:
                session = time.strftime("session_%Y%m%d-%H%M%S")
                self._session_dir = os.path.join(self._base_dir, session)
                os.makedirs(self._session_dir, exist_ok=True)
            loot_dir = os.path.join(self._session_dir, "loot")
            os.makedirs(loot_dir, exist_ok=True)
            path = os.path.join(loot_dir, f"lootframe_{int(now * 1000)}.png")
            if not cv2.imwrite(path, frame_bgr):
                raise OSError(f"cv2.imwrite returned False for {path}")
        except OSError as exc:
            self._disabled_due_to_error = True
            print(f"[LOOTCAP] disk error: {exc} — capture disabled")
            return None
        self._loot_last_save = now
        self._loot_saved += 1
        self._loot_by_reason[reason] = self._loot_by_reason.get(reason, 0) + 1
        return path


def loot_capture_reason(now, kill_capture_until, center, item_dots, ore_dots,
                        near_px, capture_on_ore):
    """Why this frame is worth saving for on-screen loot labeling, or None.

    The original gate fired whenever an ore OR item dot sat within `near_px`
    on the minimap. Ore is far more common than loot, so it ate the capture
    budget: ranking 250 candidates out of 3.5k captured frames surfaced TWO
    with a labelable item, while ore vein text was on frame after frame. The
    screen model already has 214 world-size ore annotations against 10 item
    ones, so ore no longer earns capture budget -- `capture_on_ore` keeps that
    reversible.

    The post-kill window is the strongest trigger and deliberately needs no
    detector at all: loot drops when something dies, so its label is on screen
    in the seconds after a kill. Every detector-based trigger is circular here
    -- the on-screen item detector is the thing these frames exist to train.
    """
    if now < kill_capture_until:
        return "post_kill"
    r2 = near_px * near_px
    cx, cy = center

    def _near(dots):
        return any((int(d[0]) - cx) ** 2 + (int(d[1]) - cy) ** 2 <= r2 for d in dots)

    if _near(item_dots):
        return "item_dot"
    if capture_on_ore and _near(ore_dots):
        return "ore_dot"
    return None
