import cv2
import math
import time
from enum import Enum

from config import (
    EXPLORE_HEATMAP_CELL_SIZE, EXPLORE_HEATMAP_PENALTY,
    COMPANION_TRACK_RADIUS, COMPANION_CONFIRM_FRAMES,
    COMPANION_SUPPRESS_RADIUS, COMPANION_REACQUIRE_RADIUS,
    COMPANION_FORGET_SECONDS, COMPANION_CENTER_MAX_DIST,
    ENEMY_CHASE_REPLAN_SECONDS, ENEMY_CHASE_MOVE_THRESHOLD,
    ENEMY_CHASE_PROGRESS_MIN_PX, ENEMY_CHASE_NO_PROGRESS_MAX,
    PATH_WAYPOINT_REACH_PX,
)
from minimap import dot_distance, dot_to_screen_delta, build_raw_walkable_minimap_mask
from pathfinding import buffered_walkable_mask, _a_star_full_path, smooth_path


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


class ExploreHeatMap:
    """Tracks visited minimap cells to penalise re-exploration of the same area."""

    def __init__(self, cell_size=EXPLORE_HEATMAP_CELL_SIZE):
        self.cell_size   = cell_size
        self._visits     = {}
        self._last_shape = None
        self._updates    = 0

        # Keep heatmap influence bounded and slowly decaying so it remains useful
        # during long sessions instead of over-penalizing old routes forever.
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
            self._updates += 1
            if self._updates % self._decay_every_updates == 0:
                self._decay()
        self._last_shape = minimap_shape

    def penalty(self, x, y):
        cell = (x // self.cell_size, y // self.cell_size)
        return self._visits.get(cell, 0) * EXPLORE_HEATMAP_PENALTY


class TerrainMemory:
    """Accumulates walkable minimap observations so routing learns terrain over time."""

    def __init__(self):
        self._known_walkable = None
        self._shape = None

    def reset(self):
        self._known_walkable = None
        self._shape = None

    def update(self, minimap_bgr):
        if minimap_bgr is None or minimap_bgr.size == 0:
            return
        observed = build_raw_walkable_minimap_mask(minimap_bgr)
        if observed is None or observed.size == 0:
            return
        if self._shape is not None and self._shape != observed.shape:
            self.reset()
        if self._known_walkable is None:
            self._known_walkable = observed.copy()
        else:
            self._known_walkable = cv2.bitwise_or(self._known_walkable, observed)
        self._shape = observed.shape

    def apply(self, mask):
        if mask is None or mask.size == 0:
            return mask
        if self._known_walkable is None or self._known_walkable.size == 0:
            return mask
        if self._shape is not None and mask.shape != self._shape:
            return mask
        return cv2.bitwise_or(mask, self._known_walkable)


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

    def reset(self):
        self.path_queue        = []
        self.last_enemy_pos    = None
        self.last_dist         = None
        self.no_progress_count = 0
        self._blocked_angles   = []

    def get_chase_delta(self, minimap_bgr, center, nearest, cached_mask=None):
        """Return (dx, dy) screen-space delta toward the enemy using A* path queue."""
        now = time.time()

        enemy_moved = (
            self.last_enemy_pos is None
            or dot_distance(nearest, self.last_enemy_pos) > ENEMY_CHASE_MOVE_THRESHOLD
        )
        replan_due = (now - self._replan_t) > ENEMY_CHASE_REPLAN_SECONDS

        cur_dist = dot_distance(nearest, center)
        if self.last_dist is not None:
            if cur_dist < self.last_dist - ENEMY_CHASE_PROGRESS_MIN_PX:
                self.no_progress_count = 0
            else:
                self.no_progress_count += 1
        self.last_dist = cur_dist

        if self.no_progress_count >= ENEMY_CHASE_NO_PROGRESS_MAX:
            if self.path_queue:
                adx = self.path_queue[0][0] - center[0]
                ady = self.path_queue[0][1] - center[1]
                self._blocked_angles.append(math.atan2(ady, adx))
                self._blocked_angles = self._blocked_angles[-6:]
            self.path_queue        = []
            self.no_progress_count = 0

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
            return dot_to_screen_delta(nearest, center)
        return dot_to_screen_delta(self.path_queue[0], center)
