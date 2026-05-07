"""Exploration state: heatmap, sector tracker, terrain memory, and navigator."""

import math

import cv2
import numpy as np

from config import (
    EXPLORE_HEATMAP_CELL_SIZE, EXPLORE_HEATMAP_PENALTY,
    SECTOR_TRACKER_DECAY_EVERY, SECTOR_TRACKER_PENALTY,
    TERRAIN_BLOCKED_PATCH_RADIUS,
    TERRAIN_BLOCKED_MARK_AHEAD_PX,
    TERRAIN_BLOCKED_MARK_STRENGTH,
    TERRAIN_BLOCKED_THRESHOLD,
    TERRAIN_BLOCKED_DECAY_EVERY_UPDATES,
    TERRAIN_BLOCKED_DECAY_VALUE,
    EXPLORE_NAV_STALL_FRAMES,
    EXPLORE_NAV_STALL_MOTION_PX,
    PATH_WAYPOINT_REACH_PX,
)
from minimap import dot_distance, dot_to_screen_delta, build_raw_walkable_minimap_mask
from pathfinding import _a_star_full_path, smooth_path

# Spatial decay for TerrainMemory's _known_blocked: marks within this many
# minimap pixels of the player (= minimap geometric centre) stay sharp; marks
# beyond that radius fade. Tuned for a ~225×216 minimap so the keep-area
# covers the immediate vicinity while edges (which the player is moving away
# from) decay normally.
TERRAIN_BLOCKED_DECAY_KEEP_RADIUS_PX = 60


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
        self._prev_gray = None

    def reset(self):
        self._known_walkable = None
        self._known_blocked = None
        self._shape = None
        self._updates = 0
        self._prev_gray = None

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

        # Scroll compensation: shift accumulated maps to stay world-aligned.
        # When the player moves, the minimap scrolls; old marks drift unless corrected.
        gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.GaussianBlur(gray, (3, 3), 0)
        if self._prev_gray is not None and self._prev_gray.shape == gray_b.shape:
            # Skip the FFT-based scroll detection when the minimap is essentially
            # unchanged (player stationary). phaseCorrelate is the most expensive
            # call in this method (~1–2 ms on 225×216), and on uint8 grayscale
            # a mean abs-diff < 1.0 indicates frames differ by less than one
            # intensity level on average — well below any real scroll signal.
            diff_norm = float(np.mean(cv2.absdiff(self._prev_gray, gray_b)))
            if diff_norm > 1.0:
                try:
                    (sx, sy), response = cv2.phaseCorrelate(
                        self._prev_gray.astype(np.float32),
                        gray_b.astype(np.float32),
                    )
                    if response > 0.02 and math.hypot(float(sx), float(sy)) > 0.5:
                        h_mm, w_mm = observed.shape[:2]
                        M = np.float32([[1, 0, -sx], [0, 1, -sy]])
                        warp_kw = dict(
                            dsize=(w_mm, h_mm),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        if (self._known_walkable is not None
                                and self._known_walkable.shape == observed.shape):
                            self._known_walkable = cv2.warpAffine(
                                self._known_walkable, M, **warp_kw)
                        if (self._known_blocked is not None
                                and self._known_blocked.shape == observed.shape):
                            self._known_blocked = cv2.warpAffine(
                                self._known_blocked, M, **warp_kw)
                except cv2.error:
                    pass
        self._prev_gray = gray_b

        self._ensure_shape(observed.shape)
        if self._known_walkable is None:
            self._known_walkable = observed.copy()
        else:
            self._known_walkable = cv2.bitwise_or(self._known_walkable, observed)
        self._updates += 1
        if self._updates % TERRAIN_BLOCKED_DECAY_EVERY_UPDATES == 0 and self._known_blocked is not None:
            # Spatial decay: only fade marks beyond the keep-radius from the
            # player. The player is at the minimap's geometric centre, so we
            # use that as the radius origin. Walls just bumped (near the
            # player) keep their full strength; older marks scrolled toward
            # the edges fade as the player moves away.
            h, w = self._known_blocked.shape
            cx_pl, cy_pl = w // 2, h // 2
            ys, xs = np.ogrid[:h, :w]
            far_mask = ((xs - cx_pl) ** 2 + (ys - cy_pl) ** 2) > TERRAIN_BLOCKED_DECAY_KEEP_RADIUS_PX ** 2
            decay = np.where(far_mask, TERRAIN_BLOCKED_DECAY_VALUE, 0).astype(np.uint8)
            self._known_blocked = cv2.subtract(self._known_blocked, decay)
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


class ExploreNavigator:
    """Waypoint-following path navigator for the EXPLORE state."""

    def __init__(self):
        self.path_queue: list = []
        self.current_target = None
        self._prev_gray = None
        self._stall_frames: int = 0

    def reset(self) -> None:
        self.path_queue = []
        self.current_target = None
        self._prev_gray = None
        self._stall_frames = 0

    def plan(self, minimap_bgr, center, target, cached_mask) -> bool:  # minimap_bgr reserved for future heuristics
        """Compute smoothed A* path to *target*. Returns True if a path was found."""
        self._stall_frames = 0
        self._prev_gray = None
        self.current_target = target
        if cached_mask is None:
            self.path_queue = []
            return False
        raw = _a_star_full_path(cached_mask, center, target)
        if not raw:
            self.path_queue = []
            return False
        self.path_queue = smooth_path(raw, cached_mask)
        return bool(self.path_queue)

    def get_move(self, minimap_bgr, center, cached_mask) -> "tuple[int, int] | None":
        """Return screen-space (dx, dy) toward the next waypoint, or None when done."""
        if minimap_bgr is not None and minimap_bgr.size > 0:
            gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
            gray_b = cv2.GaussianBlur(gray, (3, 3), 0)
            if self._prev_gray is not None and self._prev_gray.shape == gray_b.shape:
                try:
                    (sx, sy), response = cv2.phaseCorrelate(
                        self._prev_gray.astype(np.float32),
                        gray_b.astype(np.float32),
                    )
                    motion = math.hypot(float(sx), float(sy)) if response > 0.02 else 0.0
                except cv2.error:
                    motion = 0.0
                if motion < EXPLORE_NAV_STALL_MOTION_PX:
                    self._stall_frames = min(EXPLORE_NAV_STALL_FRAMES + 2,  # +2 headroom for hysteresis
                                             self._stall_frames + 1)
                else:
                    self._stall_frames = max(0, self._stall_frames - 1)
                if self._stall_frames >= EXPLORE_NAV_STALL_FRAMES:
                    if self.path_queue:
                        self.path_queue.pop(0)
                    self._stall_frames = 0
                    if self.current_target is not None and cached_mask is not None:
                        self.plan(minimap_bgr, center, self.current_target, cached_mask)
            self._prev_gray = gray_b

        while self.path_queue and dot_distance(self.path_queue[0], center) < PATH_WAYPOINT_REACH_PX:
            self.path_queue.pop(0)

        if not self.path_queue:
            return None

        return dot_to_screen_delta(self.path_queue[0], center)
