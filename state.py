import cv2
import math
import time
import numpy as np
from enum import Enum

from config import (
    EXPLORE_HEATMAP_CELL_SIZE, EXPLORE_HEATMAP_PENALTY,
    COMPANION_TRACK_RADIUS, COMPANION_CONFIRM_FRAMES,
    COMPANION_SUPPRESS_RADIUS, COMPANION_REACQUIRE_RADIUS,
    COMPANION_FORGET_SECONDS, COMPANION_CENTER_MAX_DIST,
    ENEMY_CHASE_REPLAN_SECONDS, ENEMY_CHASE_MOVE_THRESHOLD,
    ENEMY_CHASE_PROGRESS_MIN_PX, ENEMY_CHASE_NO_PROGRESS_MAX,
    ATTACK_CHASE_MINIMAP_DIST,
    PATH_WAYPOINT_REACH_PX,
    SECTOR_TRACKER_DECAY_EVERY, SECTOR_TRACKER_PENALTY,
    TERRAIN_BLOCKED_PATCH_RADIUS,
    TERRAIN_BLOCKED_MARK_AHEAD_PX,
    TERRAIN_BLOCKED_MARK_STRENGTH,
    TERRAIN_BLOCKED_THRESHOLD,
    TERRAIN_BLOCKED_DECAY_EVERY_UPDATES,
    TERRAIN_BLOCKED_DECAY_VALUE,
)
from minimap import dot_distance, dot_to_screen_delta, build_raw_walkable_minimap_mask
from pathfinding import (
    buffered_walkable_mask,
    _a_star_full_path,
    smooth_path,
    ray_walkability_score,
    octant_escape_delta,
    random_explore_delta
)


class State(Enum):
    EXPLORE = "EXPLORE"
    ATTACK  = "ATTACK"
    LOOT    = "LOOT"
    FLEE    = "FLEE"


class StuckDetector:
    def __init__(self, window=5, min_move=8):
        self.positions = []
        self.window    = window
        self.min_move  = min_move

    def reset(self):
        self.positions.clear()

    def update(self, x, y):
        self.positions.append((x, y))
        if len(self.positions) > self.window:
            self.positions.pop(0)

    def is_stuck(self):
        if len(self.positions) < self.window:
            return False
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        return (max(xs) - min(xs) + max(ys) - min(ys)) < self.min_move


class IntentStuckDetector:
    """Detects stuck state when movement is commanded but minimap barely moves."""

    def __init__(self, window=12, intent_min=85.0, min_motion_px=0.55, stuck_ratio=0.70):
        self.window = max(3, int(window))
        self.intent_min = float(intent_min)
        self.min_motion_px = float(min_motion_px)
        self.stuck_ratio = float(stuck_ratio)
        self._prev_gray = None
        self._flags = []

    def reset(self):
        self._prev_gray = None
        self._flags = []

    def update(self, minimap_bgr, move_dx, move_dy):
        if minimap_bgr is None or minimap_bgr.size == 0:
            return
        gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return

        try:
            (sx, sy), response = cv2.phaseCorrelate(
                self._prev_gray.astype(np.float32),
                gray.astype(np.float32),
            )
            motion = math.hypot(float(sx), float(sy))
            if response < 0.01:
                motion = 0.0
        except cv2.error:
            motion = 0.0

        intent = math.hypot(float(move_dx), float(move_dy))
        stuck_like = (intent >= self.intent_min) and (motion < self.min_motion_px)
        self._flags.append(1 if stuck_like else 0)
        if len(self._flags) > self.window:
            self._flags.pop(0)

        self._prev_gray = gray

    def is_stuck(self):
        if len(self._flags) < self.window:
            return False
        stuck_frames = sum(self._flags)
        return stuck_frames >= int(math.ceil(self.window * self.stuck_ratio))


class ExploreHeatMap:
    """Tracks visited minimap cells to penalise re-exploration of the same area."""

    def __init__(self, cell_size=EXPLORE_HEATMAP_CELL_SIZE):
        self.cell_size   = cell_size
        self._visits     = {}
        self._last_shape = None
        self._updates    = 0
        self._max_visits = 24
        self._decay_every_updates = 90

    def reset(self):
        self._visits = {}
        self._last_shape = None
        self._updates = 0

    def _decay(self):
        if not self._visits:
            return
        next_visits = {}
        for cell, count in self._visits.items():
            decayed = count - 1
            if decayed > 0:
                next_visits[cell] = decayed
        self._visits = next_visits

    def update(self, center, minimap_shape):
        if self._last_shape is not None and minimap_shape != self._last_shape:
            self._visits = {}
        elif self._last_shape is None or minimap_shape == self._last_shape:
            cell = (center[0] // self.cell_size, center[1] // self.cell_size)
            self._visits[cell] = min(self._max_visits, self._visits.get(cell, 0) + 1)
            # Spread a lighter penalty to immediately adjacent cells for broader coverage
            for dcx, dcy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                adj = (cell[0] + dcx, cell[1] + dcy)
                prev = self._visits.get(adj, 0)
                if prev < self._max_visits // 2:
                    self._visits[adj] = prev + 1
            self._updates += 1
            if self._updates % self._decay_every_updates == 0:
                self._decay()
        self._last_shape = minimap_shape

    def penalty(self, x, y):
        cell = (x // self.cell_size, y // self.cell_size)
        return self._visits.get(cell, 0) * EXPLORE_HEATMAP_PENALTY


class SectorTracker:
    """Counts explore visits per 45° compass sector to penalise re-explored directions."""

    SECTOR_COUNT = 8

    def __init__(self, decay_every=SECTOR_TRACKER_DECAY_EVERY, penalty_per_visit=SECTOR_TRACKER_PENALTY):
        self._visits  = [0] * self.SECTOR_COUNT
        self._updates = 0
        self._decay_every      = decay_every
        self.penalty_per_visit = penalty_per_visit

    def reset(self):
        self._visits  = [0] * self.SECTOR_COUNT
        self._updates = 0

    def _angle_to_sector(self, angle):
        normalized = angle % (2 * math.pi)
        return int(normalized / (2 * math.pi) * self.SECTOR_COUNT) % self.SECTOR_COUNT

    def update(self, angle):
        sector = self._angle_to_sector(angle)
        self._visits[sector] = min(20, self._visits[sector] + 1)
        self._updates += 1
        if self._decay_every and self._updates % self._decay_every == 0:
            self._visits = [max(0, v - 1) for v in self._visits]

    def visits(self, sector):
        return self._visits[sector]

    def penalty(self, angle):
        sector = self._angle_to_sector(angle)
        return self._visits[sector] * self.penalty_per_visit


class TerrainMemory:
    """Accumulates walkable minimap observations so routing learns terrain over time."""

    def __init__(self):
        self._known_walkable = None
        self._known_blocked = None
        self._shape = None
        self._updates = 0

    def reset(self):
        self._known_walkable = None
        self._known_blocked = None
        self._shape = None
        self._updates = 0

    def _ensure_shape(self, shape):
        if self._shape is not None and self._shape != shape:
            self.reset()
        if self._known_blocked is None:
            self._known_blocked = np.zeros(shape, dtype=np.uint8)

    def update(self, minimap_bgr):
        if minimap_bgr is None or minimap_bgr.size == 0:
            return
        observed = build_raw_walkable_minimap_mask(minimap_bgr)
        if observed is None or observed.size == 0:
            return
        self._ensure_shape(observed.shape)
        if self._known_walkable is None:
            self._known_walkable = observed.copy()
        else:
            self._known_walkable = cv2.bitwise_or(self._known_walkable, observed)
        self._updates += 1
        if self._updates % TERRAIN_BLOCKED_DECAY_EVERY_UPDATES == 0 and self._known_blocked is not None:
            self._known_blocked = cv2.subtract(
                self._known_blocked,
                np.full_like(self._known_blocked, TERRAIN_BLOCKED_DECAY_VALUE),
            )
        self._shape = observed.shape

    def mark_blocked(self, center, move_dx, move_dy):
        if center is None or self._shape is None:
            return
        self._ensure_shape(self._shape)
        if self._known_blocked is None:
            return

        h, w = self._shape
        cx = int(max(0, min(w - 1, center[0])))
        cy = int(max(0, min(h - 1, center[1])))

        vlen = math.hypot(move_dx, move_dy)
        if vlen < 1e-6:
            tx, ty = cx, cy
        else:
            tx = int(cx + (move_dx / vlen) * TERRAIN_BLOCKED_MARK_AHEAD_PX)
            ty = int(cy + (move_dy / vlen) * TERRAIN_BLOCKED_MARK_AHEAD_PX)
            tx = int(max(0, min(w - 1, tx)))
            ty = int(max(0, min(h - 1, ty)))

        patch = np.zeros((h, w), dtype=np.uint8)
        # Thick line toward the wall
        cv2.line(patch, (cx, cy), (tx, ty), max(12, TERRAIN_BLOCKED_MARK_STRENGTH // 2), 3)
        # Primary block circle ahead
        cv2.circle(patch, (tx, ty), TERRAIN_BLOCKED_PATCH_RADIUS, TERRAIN_BLOCKED_MARK_STRENGTH, -1)
        # Smaller secondary circle at current position so the standing point is also penalised
        cv2.circle(patch, (cx, cy), TERRAIN_BLOCKED_PATCH_RADIUS // 2, TERRAIN_BLOCKED_MARK_STRENGTH // 3, -1)
        self._known_blocked = cv2.add(self._known_blocked, patch)

    def apply(self, mask):
        if mask is None or mask.size == 0:
            return mask
        if self._shape is not None and mask.shape != self._shape:
            return mask

        fused = mask
        if self._known_walkable is not None and self._known_walkable.size != 0:
            fused = cv2.bitwise_or(fused, self._known_walkable)

        if self._known_blocked is None or self._known_blocked.size == 0:
            return fused

        blocked_bin = np.zeros_like(self._known_blocked, dtype=np.uint8)
        blocked_bin[self._known_blocked >= TERRAIN_BLOCKED_THRESHOLD] = 255
        return cv2.bitwise_and(fused, cv2.bitwise_not(blocked_bin))

    def debug_blocked_mask(self, binary=True):
        if self._known_blocked is None or self._known_blocked.size == 0:
            return None
        if not binary:
            return self._known_blocked.copy()
        blocked_bin = np.zeros_like(self._known_blocked, dtype=np.uint8)
        blocked_bin[self._known_blocked >= TERRAIN_BLOCKED_THRESHOLD] = 255
        return blocked_bin

    def blocked_pixel_count(self):
        blocked = self.debug_blocked_mask(binary=True)
        if blocked is None:
            return 0
        return int(np.count_nonzero(blocked))


class CompanionSuppressor:
    def __init__(
        self,
        track_radius=COMPANION_TRACK_RADIUS,
        confirm_frames=COMPANION_CONFIRM_FRAMES,
        suppress_radius=COMPANION_SUPPRESS_RADIUS,
        reacquire_radius=COMPANION_REACQUIRE_RADIUS,
        forget_seconds=COMPANION_FORGET_SECONDS,
        center_max_dist=COMPANION_CENTER_MAX_DIST,
    ):
        self.track_radius    = track_radius
        self.confirm_frames  = confirm_frames
        self.suppress_radius = suppress_radius
        self.reacquire_radius = reacquire_radius
        self.forget_seconds  = forget_seconds
        self.center_max_dist = center_max_dist
        self._candidate      = None
        self._hits           = 0
        self._last_seen      = 0.0

    def _pick_candidate(self, dots, center):
        if not dots:
            return None
        if self._candidate is not None:
            near_prev = [
                d for d in dots
                if dot_distance(d, self._candidate) <= self.reacquire_radius
                and dot_distance(d, center) <= self.center_max_dist
            ]
            if near_prev:
                return min(near_prev, key=lambda d: dot_distance(d, self._candidate))
            return None

        close_dots = [d for d in dots if dot_distance(d, center) <= self.track_radius]
        if not close_dots:
            return None
        candidate = min(close_dots, key=lambda d: dot_distance(d, center))
        if len(close_dots) == 1:
            return candidate
        return min(close_dots, key=lambda d: dot_distance(d, center) + dot_distance(d, candidate) * 0.25)

    def update(self, dots, center):
        candidate = self._pick_candidate(dots, center)
        now       = time.time()

        if candidate is None:
            if now - self._last_seen > self.forget_seconds:
                self._candidate = None
                self._hits      = 0
            return self._candidate

        if dot_distance(candidate, center) > self.center_max_dist:
            if now - self._last_seen > self.forget_seconds:
                self._candidate = None
                self._hits      = 0
            return self._candidate

        if self._candidate is None:
            if dot_distance(candidate, center) <= self.center_max_dist:
                self._candidate = candidate
                self._hits      = 1
        else:
            if dot_distance(candidate, self._candidate) <= self.reacquire_radius:
                self._hits     += 1
                self._candidate = candidate
            else:
                self._candidate = candidate
                self._hits      = 1

        if self._hits >= self.confirm_frames:
            return self._candidate
        return None

    def filter(self, dots):
        if self._candidate is None or self._hits < self.confirm_frames:
            return dots
        return [d for d in dots if dot_distance(d, self._candidate) > self.suppress_radius]


class EnemyChaser:
    """Navigates toward the target enemy using a smoothed A* full-path queue."""

    def __init__(self):
        self.path_queue       = []
        self.last_enemy_pos   = None
        self.last_dist        = None
        self.no_progress_count = 0
        self._replan_t        = 0.0
        self._blocked_angles  = []
        self._prev_gray       = None
        self._stall_frames    = 0
        self._last_cmd_mag    = 0.0
        self._last_cmd_angle  = None
        self._same_cmd_ticks  = 0
        self._stagnant_ticks  = 0
        self._best_dist_seen  = None
        self._bypass_until    = 0.0
        self._bypass_delta    = None
        self.chase_start_t    = 0.0

    def reset(self):
        self.path_queue        = []
        self.last_enemy_pos    = None
        self.last_dist         = None
        self.no_progress_count = 0
        self._blocked_angles   = []
        self._prev_gray        = None
        self._stall_frames     = 0
        self._last_cmd_mag     = 0.0
        self._last_cmd_angle   = None
        self._same_cmd_ticks   = 0
        self._stagnant_ticks   = 0
        self._best_dist_seen   = None
        self._bypass_until     = 0.0
        self._bypass_delta     = None
        self.chase_start_t     = 0.0

    def _estimate_scene_motion(self, minimap_bgr):
        if minimap_bgr is None or minimap_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return 0.0
        try:
            (sx, sy), response = cv2.phaseCorrelate(
                self._prev_gray.astype(np.float32),
                gray.astype(np.float32),
            )
            motion = math.hypot(float(sx), float(sy))
            if response < 0.01:
                motion = 0.0
        except cv2.error:
            motion = 0.0
        self._prev_gray = gray
        return float(motion)

    def _pick_bypass_delta(self, mask, center, nearest):
        direct_dx = float(nearest[0] - center[0])
        direct_dy = float(nearest[1] - center[1])
        vlen = math.hypot(direct_dx, direct_dy)
        if vlen < 1e-6:
            return None

        pdx = -direct_dy / vlen
        pdy = direct_dx / vlen

        side_dist = 24.0
        fwd_push = 10.0
        h, w = mask.shape[:2] if mask is not None else (0, 0)

        def _target(sign):
            tx = int(center[0] + pdx * sign * side_dist + (direct_dx / vlen) * fwd_push)
            ty = int(center[1] + pdy * sign * side_dist + (direct_dy / vlen) * fwd_push)
            if mask is not None:
                tx = int(max(0, min(w - 1, tx)))
                ty = int(max(0, min(h - 1, ty)))
            return tx, ty

        if mask is None:
            t = _target(1.0)
            return dot_to_screen_delta(t, center)

        t_pos = _target(1.0)
        t_neg = _target(-1.0)
        s_pos = ray_walkability_score(mask, center, t_pos, samples=12)
        s_neg = ray_walkability_score(mask, center, t_neg, samples=12)
        target = t_pos if s_pos >= s_neg else t_neg
        ddx, ddy = dot_to_screen_delta(target, center)
        mag = math.hypot(ddx, ddy)
        if mag > 1e-6 and mag < 120.0:
            scale = 120.0 / mag
            ddx = int(ddx * scale)
            ddy = int(ddy * scale)
        return ddx, ddy

    def get_chase_delta(self, minimap_bgr, center, nearest, cached_mask=None):
        """Return (dx, dy) screen-space delta toward the enemy using A* path queue."""
        now = time.time()
        chase_far_threshold = ATTACK_CHASE_MINIMAP_DIST + 4

        # ── Timeout + Escape (verhindert Endlosschleife) ─────────────────────
        if self.chase_start_t == 0.0:
            self.chase_start_t = now

        if now - self.chase_start_t > 4.0 and dot_distance(nearest, center) > chase_far_threshold:
            print(f"[CHASE] TIMEOUT nach {now-self.chase_start_t:.1f}s bei {dot_distance(nearest, center):.0f}px → ESCAPE")
            self.path_queue = []
            self.chase_start_t = 0.0
            mask = cached_mask if cached_mask is not None else buffered_walkable_mask(minimap_bgr, center)
            esc = octant_escape_delta(mask, center, 0, 0, recent_angles=None)
            return esc if esc else random_explore_delta()

        # ── restlicher Original-Code (unverändert) ───────────────────────
        scene_motion = self._estimate_scene_motion(minimap_bgr)

        if self._last_cmd_mag >= 90.0 and scene_motion < 0.55:
            self._stall_frames = min(12, self._stall_frames + 1)
        else:
            self._stall_frames = max(0, self._stall_frames - 1)

        if now < self._bypass_until and self._bypass_delta is not None:
            self._last_cmd_mag = math.hypot(self._bypass_delta[0], self._bypass_delta[1])
            return self._bypass_delta

        enemy_moved = (
            self.last_enemy_pos is None
            or dot_distance(nearest, self.last_enemy_pos) > ENEMY_CHASE_MOVE_THRESHOLD
        )
        replan_due = (now - self._replan_t) > ENEMY_CHASE_REPLAN_SECONDS

        cur_dist = dot_distance(nearest, center)
        if self._best_dist_seen is None or cur_dist < (self._best_dist_seen - ENEMY_CHASE_PROGRESS_MIN_PX):
            self._best_dist_seen = cur_dist
            self._stagnant_ticks = 0
        else:
            self._stagnant_ticks = min(20, self._stagnant_ticks + 1)

        if self.last_dist is not None:
            if cur_dist < self.last_dist - ENEMY_CHASE_PROGRESS_MIN_PX:
                self.no_progress_count = 0
            else:
                self.no_progress_count += 1
        self.last_dist = cur_dist

        if self._stall_frames >= 5:
            self.no_progress_count = max(self.no_progress_count, ENEMY_CHASE_NO_PROGRESS_MAX)

        stagnation_trigger = (
            self._stagnant_ticks >= 8
            and cur_dist > chase_far_threshold
        )

        if self.no_progress_count >= ENEMY_CHASE_NO_PROGRESS_MAX or stagnation_trigger:
            if self.path_queue:
                adx = self.path_queue[0][0] - center[0]
                ady = self.path_queue[0][1] - center[1]
                self._blocked_angles.append(math.atan2(ady, adx))
            direct_dx = nearest[0] - center[0]
            direct_dy = nearest[1] - center[1]
            if math.hypot(direct_dx, direct_dy) > 0:
                self._blocked_angles.append(math.atan2(direct_dy, direct_dx))
            self._blocked_angles   = self._blocked_angles[-6:]
            mask = cached_mask if cached_mask is not None else buffered_walkable_mask(minimap_bgr, center)
            self._bypass_delta = self._pick_bypass_delta(mask, center, nearest)
            if self._bypass_delta is not None:
                self._bypass_until = now + 1.00
            self.path_queue        = []
            self.no_progress_count = 0
            self._stall_frames     = 0
            self._stagnant_ticks   = 0
            self._best_dist_seen   = cur_dist

        if self.path_queue:
            if dot_distance(self.path_queue[0], center) < PATH_WAYPOINT_REACH_PX:
                self.path_queue.pop(0)

        if not self.path_queue or enemy_moved or replan_due:
            mask = cached_mask if cached_mask is not None else buffered_walkable_mask(minimap_bgr, center)
            if mask is not None:
                raw = _a_star_full_path(mask, center, nearest)
                if raw:
                    self.path_queue = smooth_path(raw, mask)
            self.last_enemy_pos = nearest
            self._replan_t      = now

        if not self.path_queue:
            direct = dot_to_screen_delta(nearest, center)
            self._last_cmd_mag = math.hypot(direct[0], direct[1])
            ang = math.atan2(direct[1], direct[0]) if self._last_cmd_mag > 1e-6 else None
            if ang is not None and self._last_cmd_angle is not None:
                diff = abs((ang - self._last_cmd_angle + math.pi) % (2 * math.pi) - math.pi)
                self._same_cmd_ticks = self._same_cmd_ticks + 1 if diff < math.radians(8) else 0
            else:
                self._same_cmd_ticks = 0
            self._last_cmd_angle = ang
            return direct

        nxt = dot_to_screen_delta(self.path_queue[0], center)
        self._last_cmd_mag = math.hypot(nxt[0], nxt[1])
        ang = math.atan2(nxt[1], nxt[0]) if self._last_cmd_mag > 1e-6 else None
        if ang is not None and self._last_cmd_angle is not None:
            diff = abs((ang - self._last_cmd_angle + math.pi) % (2 * math.pi) - math.pi)
            self._same_cmd_ticks = self._same_cmd_ticks + 1 if diff < math.radians(8) else 0
        else:
            self._same_cmd_ticks = 0
        self._last_cmd_angle = ang

        if self._same_cmd_ticks >= 6 and cur_dist > chase_far_threshold:
            mask = cached_mask if cached_mask is not None else buffered_walkable_mask(minimap_bgr, center)
            bypass = self._pick_bypass_delta(mask, center, nearest)
            if bypass is not None:
                self._bypass_delta = bypass
                self._bypass_until = now + 1.00
                self._same_cmd_ticks = 0
                self._last_cmd_mag = math.hypot(bypass[0], bypass[1])
                return bypass
        return nxt