import cv2
import ctypes
import ctypes.wintypes
import mss
import numpy as np
import time
import random
import keyboard
import math
import os
import heapq
from enum import Enum

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1920, 1080

# Minimap regions (small corner map and larger overlay map).
# Calibrate both for your setup; press F6 while bot runs to switch mode.
MINIMAP_SMALL = {"left": 1673, "top": 7, "width": 225, "height": 216}
MINIMAP_LARGE = {
    "left": max(0, SCREEN_W // 2 - 430),
    "top": max(0, SCREEN_H // 2 - 350),
    "width": 860,
    "height": 700,
}
MINIMAP_MODE = "large"  # "small" or "large"

# HP bar region
HP_BAR = {"x": 112, "y": 39, "w": 180, "h": 8}
HP_FLEE_THRESHOLD  = 0.30
HP_RESUME_THRESHOLD = 0.50

# Legacy fallback range for APPROACH -> ATTACK.
# Primary behavior now attacks immediately from EXPLORE when an enemy dot is visible.
ATTACK_RANGE_PX = 35

# Seconds without a visible enemy dot before ATTACK exits to LOOT
# (covers the case where the enemy is inside the exclusion zone but not dead yet)
ATTACK_LOST_GRACE = 2.0

# Small center exclusion so near enemies are still detected.
PLAYER_EXCLUSION_RADIUS = 14
PLAYER_EXCLUSION_RADIUS_FRAC = 0.06

# Deadzone for movement axis mapping; smaller values allow easier diagonal movement.
MOVE_AXIS_DEADZONE = 10

# Re-plan minimap-guided exploration path this often (seconds)
EXPLORE_REPLAN_SECONDS = 5.0
EXPLORE_MOVE_MIN_SECONDS = 1.20
EXPLORE_MOVE_MAX_SECONDS = 2.10
POST_KILL_SCAN_MIN_SECONDS = 0.40
POST_KILL_SCAN_MAX_SECONDS = 0.80

# Reduce repeated stuck log spam.
STUCK_LOG_COOLDOWN = 2.5

# Re-plan movement while attacking so the bot keeps running/kiting.
ATTACK_MOVE_REPLAN_SECONDS = 0.25
ATTACK_KITE_CHANCE = 0.70
ATTACK_CHASE_MINIMAP_DIST = 32

# Track and suppress the stable companion dot near the player.
COMPANION_TRACK_RADIUS = 42
COMPANION_CONFIRM_FRAMES = 5
COMPANION_SUPPRESS_RADIUS = 12
COMPANION_REACQUIRE_RADIUS = 10
COMPANION_FORGET_SECONDS = 2.5
COMPANION_CENTER_MAX_DIST = 38

# Jump support while moving in the direction the cursor is facing.
JUMP_COOLDOWN = 0.9
JUMP_ALIGNMENT_THRESHOLD = 0.72

# Keep the cursor a bit farther away from the character so movement/jumps extend.
AIM_MIN_DISTANCE = 180
AIM_MAX_DISTANCE = 280

# Coarse-grid minimap pathfinding settings.
PATH_GRID_STEP = 3
PATH_WAYPOINT_STEPS = 7
WALL_BUFFER_ITERATIONS        = 1      # Erosion passes on walkable mask (wall clearance)
PATH_WAYPOINT_REACH_PX        = 8      # Distance to consider a waypoint reached
STUCK_BACKTRACK_DURATION_MIN  = 0.20   # Min reverse-move duration on stuck
STUCK_BACKTRACK_DURATION_MAX  = 0.35   # Max reverse-move duration on stuck
STUCK_BACKTRACK_COOLDOWN      = 0.80   # Seconds between backtrack attempts
EXPLORE_HEATMAP_CELL_SIZE     = 8      # Minimap pixels per heatmap cell
EXPLORE_HEATMAP_PENALTY       = 15.0   # Score penalty per visit count
ENEMY_CHASE_REPLAN_SECONDS    = 0.40   # How often EnemyChaser replans path
ENEMY_CHASE_MOVE_THRESHOLD    = 10     # Enemy dot px movement to force replan
ENEMY_CHASE_PROGRESS_MIN_PX   = 4      # Min distance reduction to count as progress
ENEMY_CHASE_NO_PROGRESS_MAX   = 3      # Replans with no progress before new approach angle
KITE_COMMIT_MIN_SECONDS       = 0.50   # Min time to hold a kite direction
KITE_COMMIT_MAX_SECONDS       = 0.80   # Max time to hold a kite direction
ATTACK_STUCK_WINDOW           = 6      # Frames for attack-phase stuck detector
ATTACK_STUCK_MIN_MOVE         = 6      # Min minimap px movement to not be stuck


# Item templates
ITEM_TEMPLATE_PATHS = ["templates/gooditem.png", "templates/goldvein.png"]
ITEM_MATCH_THRESHOLD = 0.78

# Timeouts
ATTACK_TIMEOUT = 10.0
LOOT_TIMEOUT   = 8.0
FLEE_MAX_WAIT  = 15.0

# Temporarily disable loot handling until pickup is finalized.
ENABLE_LOOT = False


class State(Enum):
    EXPLORE = "EXPLORE"
    ATTACK  = "ATTACK"
    LOOT    = "LOOT"
    FLEE    = "FLEE"


# ── DETECTORS ─────────────────────────────────────────────────────────────────

def find_enemy_dots(minimap_bgr):
    """Return list of (x, y) dot centers in minimap image coordinates."""
    if minimap_bgr is None or minimap_bgr.size == 0:
        return []
    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)

    # Monster colors (red, purple, gold) with robust thresholds for dark maps.
    r1 = cv2.inRange(hsv, np.array([0,   135, 125]), np.array([10,  255, 255]))
    r2 = cv2.inRange(hsv, np.array([170, 135, 125]), np.array([180, 255, 255]))
    p  = cv2.inRange(hsv, np.array([128,  90, 100]), np.array([160, 255, 255]))
    y  = cv2.inRange(hsv, np.array([18,  135, 135]), np.array([34,  255, 255]))

    mask = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(r1, r2), p), y)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = minimap_bgr.shape[:2]
    cx0 = w // 2
    cy0 = h // 2
    # Keep center exclusion small; rely mostly on color + circularity filters.
    exclusion_radius = max(6, min(PLAYER_EXCLUSION_RADIUS, int(min(w, h) * PLAYER_EXCLUSION_RADIUS_FRAC)))
    dots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (4 < area < 180):
            continue
        # Only keep roughly circular blobs – filters out companion icons, star shapes, etc.
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity < 0.62:
            continue
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5 < exclusion_radius:
                continue
            dots.append((cx, cy))
    return dots


def get_hp_percent(frame_bgr):
    """Return HP as 0.0–1.0 by measuring rightmost red pixel in HP bar region."""
    if frame_bgr is None or frame_bgr.size == 0:
        return 1.0
    bar = frame_bgr[HP_BAR["y"]:HP_BAR["y"] + HP_BAR["h"],
                    HP_BAR["x"]:HP_BAR["x"] + HP_BAR["w"]]
    hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    r1 = cv2.inRange(hsv, np.array([0,   100, 100]), np.array([10,  255, 255]))
    r2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(r1, r2)
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return 0.0
    return float(cols[-1] + 1) / HP_BAR["w"]


def find_items_on_screen(frame_bgr, templates):
    """Return (cx, cy) of best item match or None."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    best_pt, best_score = None, -1
    for tpl in templates:
        if tpl is None:
            continue
        w, h = tpl.shape[::-1]
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score >= ITEM_MATCH_THRESHOLD and score > best_score:
            best_score = score
            best_pt = (loc[0] + w // 2, loc[1] + h // 2)
    return best_pt


# ── ANTI-STUCK ────────────────────────────────────────────────────────────────

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
        self.cell_size = cell_size
        self._visits = {}
        self._last_shape = None

    def update(self, center, minimap_shape):
        if self._last_shape is not None and minimap_shape != self._last_shape:
            self._visits = {}
        elif self._last_shape is None or minimap_shape == self._last_shape:
            cell = (center[0] // self.cell_size, center[1] // self.cell_size)
            self._visits[cell] = self._visits.get(cell, 0) + 1
        self._last_shape = minimap_shape

    def penalty(self, x, y):
        cell = (x // self.cell_size, y // self.cell_size)
        return self._visits.get(cell, 0) * EXPLORE_HEATMAP_PENALTY


# ── HELPERS ───────────────────────────────────────────────────────────────────

def minimap_center():
    return MINIMAP_SMALL["width"] // 2, MINIMAP_SMALL["height"] // 2


def minimap_center_from_image(minimap_bgr):
    if minimap_bgr is None or minimap_bgr.size == 0:
        return minimap_center()
    return minimap_bgr.shape[1] // 2, minimap_bgr.shape[0] // 2


def get_minimap_region(mode):
    return MINIMAP_LARGE if mode == "large" else MINIMAP_SMALL


def nearest_dot(dots, center):
    if not dots:
        return None
    cx, cy = center
    return min(dots, key=lambda d: (d[0] - cx) ** 2 + (d[1] - cy) ** 2)


def dot_distance(dot, center):
    return ((dot[0] - center[0]) ** 2 + (dot[1] - center[1]) ** 2) ** 0.5


def dot_to_screen_delta(dot, center):
    """Convert minimap dot offset → screen pixel offset to click."""
    dx = dot[0] - center[0]
    dy = dot[1] - center[1]
    scale = 4.0
    return int(dx * scale), int(dy * scale)


def random_explore_delta():
    angle = random.uniform(0, 2 * math.pi)
    dist  = random.randint(80, 200)
    return int(dist * math.cos(angle)), int(dist * math.sin(angle))


def build_walkable_minimap_mask(minimap_bgr):
    """Estimate walkable minimap area and include unexplored cyan for progression."""
    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)

    # Cyan/blue marks unexplored territory that should be explored.
    unknown_fog = cv2.inRange(hsv, np.array([78, 80, 55]), np.array([110, 255, 255]))

    # Keep non-black explored region and include unknown-fog as valid pathing area.
    explored = cv2.inRange(hsv, np.array([0, 0, 16]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(explored, unknown_fog)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return mask


def connected_walkable_mask(mask, center):
    """Keep only the walkable component reachable from the player-center area."""
    if mask is None or mask.size == 0:
        return None

    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return mask

    h, w = binary.shape
    cx = int(max(0, min(w - 1, center[0])))
    cy = int(max(0, min(h - 1, center[1])))

    seed_x, seed_y = cx, cy
    if binary[cy, cx] == 0:
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return mask
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        i = int(np.argmin(d2))
        seed_x, seed_y = int(xs[i]), int(ys[i])

    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    if num_labels <= 1:
        return mask

    seed_label = labels[seed_y, seed_x]
    if seed_label == 0:
        return mask

    connected = np.zeros_like(mask)
    connected[labels == seed_label] = 255
    return connected


def buffered_walkable_mask(minimap_bgr, center):
    """Build walkable mask with wall buffer so A* paths stay away from edges."""
    if minimap_bgr is None or minimap_bgr.size == 0:
        return None
    mask = build_walkable_minimap_mask(minimap_bgr)
    mask = connected_walkable_mask(mask, center)
    if mask is None:
        return None
    if WALL_BUFFER_ITERATIONS > 0:
        mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=WALL_BUFFER_ITERATIONS)
    return mask


def minimap_explore_target(minimap_bgr, center, recent_angles, blocked_angles=None):
    """Pick an exploration target from bright/open minimap regions."""
    if minimap_bgr is None or minimap_bgr.size == 0:
        return None

    walkable = build_walkable_minimap_mask(minimap_bgr)
    walkable = connected_walkable_mask(walkable, center)
    if walkable is None:
        return None

    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    unknown_fog = cv2.inRange(hsv, np.array([78, 80, 55]), np.array([110, 255, 255]))

    cx, cy = center
    cv2.circle(walkable, (cx, cy), PLAYER_EXCLUSION_RADIUS + 4, 0, -1)

    contours, _ = cv2.findContours(walkable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_dot = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 18:
            continue
        points = cnt.reshape(-1, 2)
        if points.size == 0:
            continue

        # Prefer far points, but penalize repeating the same heading.
        distances = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        idx = int(np.argmax(distances))
        tx, ty = int(points[idx][0]), int(points[idx][1])
        dist = float(distances[idx])
        if dist < PLAYER_EXCLUSION_RADIUS + 6:
            continue

        angle = math.atan2(ty - cy, tx - cx)
        if angle_is_blocked(angle, blocked_angles):
            continue
        if recent_angles:
            min_diff = min(abs((angle - a + math.pi) % (2 * math.pi) - math.pi) for a in recent_angles)
            repeat_penalty = max(0.0, (0.60 - min_diff) / 0.60) * 35.0
        else:
            repeat_penalty = 0.0

        path_score = ray_walkability_score(walkable, center, (tx, ty), samples=18)
        if path_score < 0.45:
            continue

        score = dist + (path_score * 120.0) - repeat_penalty
        # Prefer unexplored (cyan) targets to push map exploration forward.
        if unknown_fog[ty, tx] > 0:
            score += 70.0
        # Avoid targets that look like they run into walls, especially downward.
        if ty > cy + 10 and path_score < 0.75:
            score -= 20.0
        if score > best_score:
            best_score = score
            best_dot = (tx, ty)

    return best_dot


def angle_is_blocked(angle, blocked_angles, tolerance=0.45):
    if not blocked_angles:
        return False
    for blocked in blocked_angles:
        diff = abs((angle - blocked + math.pi) % (2 * math.pi) - math.pi)
        if diff < tolerance:
            return True
    return False


def choose_explore_delta(minimap_bgr, center, recent_angles, blocked_angles=None):
    dot = minimap_explore_target(minimap_bgr, center, recent_angles, blocked_angles)
    if dot is None:
        return random_explore_delta(), None
    dx, dy = path_delta_from_center(minimap_bgr, center, dot)
    return (dx, dy), dot


def scale_vector(dx, dy, min_len=95, max_len=220):
    """Scale movement vector into a useful WASD magnitude range."""
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return random_explore_delta()
    target = random.uniform(min_len, max_len)
    factor = target / length
    return int(dx * factor), int(dy * factor)


def ray_walkability_score(mask, center, target, samples=16):
    """Score how much of the path from center to target stays on walkable minimap area."""
    if mask is None or mask.size == 0:
        return 0.0

    cx, cy = center
    tx, ty = target
    xs = np.linspace(cx, tx, num=samples, dtype=np.int32)
    ys = np.linspace(cy, ty, num=samples, dtype=np.int32)

    h, w = mask.shape[:2]
    hits = 0
    weighted_hits = 0.0
    for i, (x, y) in enumerate(zip(xs, ys)):
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            hits += 1
            weighted_hits += 1.0 + (i / max(1, samples - 1)) * 0.5

    return weighted_hits / samples


def _nearest_walkable_cell(grid, gx, gy, radius=6):
    h, w = grid.shape
    if 0 <= gx < w and 0 <= gy < h and grid[gy, gx]:
        return gx, gy
    best = None
    best_d2 = None
    for yy in range(max(0, gy - radius), min(h, gy + radius + 1)):
        for xx in range(max(0, gx - radius), min(w, gx + radius + 1)):
            if not grid[yy, xx]:
                continue
            d2 = (xx - gx) ** 2 + (yy - gy) ** 2
            if best_d2 is None or d2 < best_d2:
                best = (xx, yy)
                best_d2 = d2
    return best


def _a_star_next_waypoint(mask, center, target, step=PATH_GRID_STEP, waypoint_steps=PATH_WAYPOINT_STEPS):
    """Return a waypoint on a walkable A* path from center to target (minimap coords)."""
    if mask is None or mask.size == 0:
        return None

    grid = (mask[::step, ::step] > 0)
    gh, gw = grid.shape
    if gh == 0 or gw == 0:
        return None

    sx = int(max(0, min(gw - 1, center[0] // step)))
    sy = int(max(0, min(gh - 1, center[1] // step)))
    tx = int(max(0, min(gw - 1, target[0] // step)))
    ty = int(max(0, min(gh - 1, target[1] // step)))

    s = _nearest_walkable_cell(grid, sx, sy)
    t = _nearest_walkable_cell(grid, tx, ty)
    if s is None or t is None:
        return None

    if s == t:
        return target

    neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    open_heap = []
    heapq.heappush(open_heap, (0.0, s))
    g_score = {s: 0.0}
    came_from = {}
    closed = set()

    def h_cost(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == t:
            break
        closed.add(current)

        cx, cy = current
        for dx, dy, c in neighbors:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < gw and 0 <= ny < gh):
                continue
            if not grid[ny, nx]:
                continue
            nxt = (nx, ny)
            tentative = g_score[current] + c
            if tentative < g_score.get(nxt, float('inf')):
                came_from[nxt] = current
                g_score[nxt] = tentative
                f = tentative + h_cost(nxt, t)
                heapq.heappush(open_heap, (f, nxt))

    if t not in came_from:
        return None

    path = [t]
    cur = t
    while cur != s:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()

    idx = min(waypoint_steps, len(path) - 1)
    wx, wy = path[idx]
    return (wx * step + step // 2, wy * step + step // 2)


def _a_star_full_path(mask, center, target, step=PATH_GRID_STEP):
    """Return full A* path as list of minimap (x, y) coords from center to target, or None."""
    if mask is None or mask.size == 0:
        return None

    grid = (mask[::step, ::step] > 0)
    gh, gw = grid.shape
    if gh == 0 or gw == 0:
        return None

    sx = int(max(0, min(gw - 1, center[0] // step)))
    sy = int(max(0, min(gh - 1, center[1] // step)))
    tx = int(max(0, min(gw - 1, target[0] // step)))
    ty = int(max(0, min(gh - 1, target[1] // step)))

    s = _nearest_walkable_cell(grid, sx, sy)
    t = _nearest_walkable_cell(grid, tx, ty)
    if s is None or t is None:
        return None
    if s == t:
        return [target]

    neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    open_heap = []
    heapq.heappush(open_heap, (0.0, s))
    g_score = {s: 0.0}
    came_from = {}
    closed = set()

    def h_cost(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == t:
            break
        closed.add(current)
        ncx, ncy = current
        for dx, dy, c in neighbors:
            nx2, ny2 = ncx + dx, ncy + dy
            if not (0 <= nx2 < gw and 0 <= ny2 < gh):
                continue
            if not grid[ny2, nx2]:
                continue
            nxt = (nx2, ny2)
            tentative = g_score[current] + c
            if tentative < g_score.get(nxt, float('inf')):
                came_from[nxt] = current
                g_score[nxt] = tentative
                f = tentative + h_cost(nxt, t)
                heapq.heappush(open_heap, (f, nxt))

    if t not in came_from:
        return None

    path = [t]
    cur = t
    while cur != s:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()

    return [(gx * step + step // 2, gy * step + step // 2) for gx, gy in path]


def smooth_path(path, mask):
    """String-pull an A* path: skip intermediate waypoints when line-of-sight is clear."""
    if not path or len(path) < 3 or mask is None:
        return path
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if ray_walkability_score(mask, path[i], path[j], samples=20) >= 0.95:
                break
            j -= 1
        smoothed.append(path[j])
        i = j
    return smoothed


def path_delta_from_center(minimap_bgr, center, target_dot):
    walkable = build_walkable_minimap_mask(minimap_bgr)
    walkable = connected_walkable_mask(walkable, center)
    if walkable is None:
        return dot_to_screen_delta(target_dot, center)

    waypoint = _a_star_next_waypoint(walkable, center, target_dot)
    if waypoint is None:
        return dot_to_screen_delta(target_dot, center)
    return dot_to_screen_delta(waypoint, center)

def choose_attack_move_delta(
    aim_dx, aim_dy, minimap_bgr, center, recent_angles, strafe_sign,
    enemy_dist=None, enemy_dot=None,
    enemy_chaser=None,
    kite_committed_until=0.0, kite_cached_move=None,
    cached_mask=None,
):
    """Pick movement while attacking. Returns (move_delta, strafe_sign, kite_committed_until, kite_cached_move)."""
    now = time.time()

    if enemy_dist is not None and enemy_dist > ATTACK_CHASE_MINIMAP_DIST:
        if enemy_chaser is not None and enemy_dot is not None:
            delta = enemy_chaser.get_chase_delta(minimap_bgr, center, enemy_dot, cached_mask=cached_mask)
            chase_delta = scale_vector(delta[0], delta[1], min_len=170, max_len=300)
        elif enemy_dot is not None:
            direct_dx, direct_dy = dot_to_screen_delta(enemy_dot, center)
            chase_delta = scale_vector(direct_dx, direct_dy, min_len=170, max_len=300)
        else:
            chase_delta = scale_vector(aim_dx, aim_dy, min_len=140, max_len=250)
        return chase_delta, strafe_sign, kite_committed_until, kite_cached_move

    if now < kite_committed_until and kite_cached_move is not None:
        return kite_cached_move, strafe_sign, kite_committed_until, kite_cached_move

    if random.random() < ATTACK_KITE_CHANCE and (abs(aim_dx) + abs(aim_dy)) > 20:
        strafe_x = -aim_dy * strafe_sign
        strafe_y = aim_dx * strafe_sign
        move_x = strafe_x - 0.45 * aim_dx
        move_y = strafe_y - 0.45 * aim_dy
        new_move = scale_vector(move_x, move_y, min_len=90, max_len=210)
        new_strafe = -strafe_sign
    else:
        explore_move, _ = choose_explore_delta(minimap_bgr, center, recent_angles, None)
        new_move = scale_vector(explore_move[0], explore_move[1], min_len=95, max_len=220)
        new_strafe = strafe_sign

    commit_until = now + random.uniform(KITE_COMMIT_MIN_SECONDS, KITE_COMMIT_MAX_SECONDS)
    return new_move, new_strafe, commit_until, new_move


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
        self.track_radius = track_radius
        self.confirm_frames = confirm_frames
        self.suppress_radius = suppress_radius
        self.reacquire_radius = reacquire_radius
        self.forget_seconds = forget_seconds
        self.center_max_dist = center_max_dist
        self._candidate = None
        self._hits = 0
        self._last_seen = 0.0

    def _pick_candidate(self, dots, center):
        if not dots:
            return None
        if self._candidate is not None:
            near_prev = [
                dot for dot in dots
                if dot_distance(dot, self._candidate) <= self.reacquire_radius
                and dot_distance(dot, center) <= self.center_max_dist
            ]
            if near_prev:
                return min(near_prev, key=lambda dot: dot_distance(dot, self._candidate))
            # Keep previous candidate if briefly occluded instead of jumping to enemies.
            return None

        close_dots = [dot for dot in dots if dot_distance(dot, center) <= self.track_radius]
        if not close_dots:
            return None
        # Prefer the dot closest to center, but only if it is clearly not a monster cluster.
        candidate = min(close_dots, key=lambda dot: dot_distance(dot, center))
        if len(close_dots) == 1:
            return candidate

        # Multiple nearby dots: keep the most stable one by favoring the one closest to the prior center sample.
        return min(close_dots, key=lambda dot: dot_distance(dot, center) + dot_distance(dot, candidate) * 0.25)

    def update(self, dots, center):
        candidate = self._pick_candidate(dots, center)
        now = time.time()

        if candidate is None:
            if now - self._last_seen > self.forget_seconds:
                self._candidate = None
                self._hits = 0
            return self._candidate

        # Companion marker should remain close to player center; do not drift to enemy packs.
        if dot_distance(candidate, center) > self.center_max_dist:
            if now - self._last_seen > self.forget_seconds:
                self._candidate = None
                self._hits = 0
            return self._candidate

        if self._candidate is None:
            if dot_distance(candidate, center) <= self.center_max_dist:
                self._candidate = candidate
                self._hits = 1
        else:
            if dot_distance(candidate, self._candidate) <= self.reacquire_radius:
                self._hits += 1
                self._candidate = candidate
            else:
                self._candidate = candidate
                self._hits = 1

        if self._hits >= self.confirm_frames:
            return self._candidate
        return None

    def filter(self, dots):
        if self._candidate is None or self._hits < self.confirm_frames:
            return dots
        return [dot for dot in dots if dot_distance(dot, self._candidate) > self.suppress_radius]


class EnemyChaser:
    """Navigates toward the target enemy using a smoothed A* full-path queue."""

    def __init__(self):
        self.path_queue = []
        self.last_enemy_pos = None
        self.last_dist = None
        self.no_progress_count = 0
        self._replan_t = 0.0
        self._blocked_angles = []

    def reset(self):
        self.path_queue = []
        self.last_enemy_pos = None
        self.last_dist = None
        self.no_progress_count = 0
        self._blocked_angles = []

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
            self.path_queue = []
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
            self._replan_t = now

        if not self.path_queue:
            return dot_to_screen_delta(nearest, center)

        return dot_to_screen_delta(self.path_queue[0], center)


def sleep_interruptible(duration, step=0.02):
    """Sleep in short slices; return False if user requested pause/stop."""
    end_t = time.time() + max(0.0, duration)
    while time.time() < end_t:
        if keyboard.is_pressed('q') or keyboard.is_pressed('p'):
            return False
        time.sleep(min(step, max(0.0, end_t - time.time())))
    return True


def wasd_move(dx, dy, duration=0.9):
    """Press WASD keys in the direction of (dx, dy) for `duration` seconds."""
    keys = []
    if dy < -MOVE_AXIS_DEADZONE:
        keys.append('w')
    if dy > MOVE_AXIS_DEADZONE:
        keys.append('s')
    if dx < -MOVE_AXIS_DEADZONE:
        keys.append('a')
    if dx > MOVE_AXIS_DEADZONE:
        keys.append('d')
    if not keys:
        keys = ['d']
    try:
        for k in keys:
            keyboard.press(k)
        sleep_interruptible(duration)
    finally:
        for k in keys:
            keyboard.release(k)


def backtrack_move(dx, dy, last_backtrack_t):
    """Reverse briefly to escape a corner. `last_backtrack_t` is a one-element list used as mutable ref."""
    now = time.time()
    if now - last_backtrack_t[0] < STUCK_BACKTRACK_COOLDOWN:
        return
    duration = random.uniform(STUCK_BACKTRACK_DURATION_MIN, STUCK_BACKTRACK_DURATION_MAX)
    wasd_move(-dx, -dy, duration)
    last_backtrack_t[0] = now


def left_mouse_down():
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN


def left_mouse_up():
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP


def tap_space(hold_time=0.04):
    keyboard.press('space')
    sleep_interruptible(hold_time)
    keyboard.release('space')


def maybe_jump_with_movement(move_dx, move_dy, cooldown_tracker):
    """Tap jump if cursor and movement point roughly the same way."""
    now = time.time()
    if now - cooldown_tracker[0] < JUMP_COOLDOWN:
        return

    try:
        pt = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    except Exception:
        return

    cursor_dx = pt.x - (SCREEN_W // 2)
    cursor_dy = pt.y - (SCREEN_H // 2)
    move_len = math.hypot(move_dx, move_dy)
    cursor_len = math.hypot(cursor_dx, cursor_dy)
    if move_len < 1e-6 or cursor_len < 1e-6:
        return

    alignment = (move_dx * cursor_dx + move_dy * cursor_dy) / (move_len * cursor_len)
    if alignment >= JUMP_ALIGNMENT_THRESHOLD:
        tap_space()
        cooldown_tracker[0] = now


def release_all_inputs():
    """Emergency input release for movement keys and held mouse button."""
    for k in ('w', 'a', 's', 'd'):
        keyboard.release(k)
    left_mouse_up()


def wasd_move_and_attack(move_dx, move_dy, duration=0.32, aim_dx=None, aim_dy=None):
    """Move with WASD while left-click is held for continuous auto-fire."""
    if aim_dx is None:
        aim_dx = move_dx
    if aim_dy is None:
        aim_dy = move_dy

    keys = []
    if move_dy < -MOVE_AXIS_DEADZONE:
        keys.append('w')
    if move_dy > MOVE_AXIS_DEADZONE:
        keys.append('s')
    if move_dx < -MOVE_AXIS_DEADZONE:
        keys.append('a')
    if move_dx > MOVE_AXIS_DEADZONE:
        keys.append('d')
    if not keys:
        keys = ['d']

    try:
        for k in keys:
            keyboard.press(k)

        end_t = time.time() + duration
        while time.time() < end_t:
            if keyboard.is_pressed('q') or keyboard.is_pressed('p'):
                break
            attack_toward(aim_dx, aim_dy)
            if not sleep_interruptible(0.02):
                break
    finally:
        for k in keys:
            keyboard.release(k)


def attack_toward(dx, dy):
    cx, cy = SCREEN_W // 2, SCREEN_H // 2
    length = math.hypot(dx, dy)
    if length < 1e-6:
        tx, ty = cx, cy
    else:
        target_distance = random.uniform(AIM_MIN_DISTANCE, AIM_MAX_DISTANCE)
        scale = target_distance / length
        tx = int(cx + dx * scale + random.randint(-10, 10))
        ty = int(cy + dy * scale + random.randint(-10, 10))
    tx = max(50, min(SCREEN_W - 50, tx))
    ty = max(50, min(SCREEN_H - 50, ty))
    ctypes.windll.user32.SetCursorPos(tx, ty)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    item_templates = []
    for path in ITEM_TEMPLATE_PATHS:
        if os.path.exists(path):
            item_templates.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
            print(f"Template: {path}")
        else:
            item_templates.append(None)
            print(f"Fehlt: {path}")

    print("Bot gestartet | q=Stop | p=Pause | esc=Emergency stop | f6=Minimap Mode")
    print("Spiel muss sichtbar sein (kein Hintergrundmodus mit mss)")
    print(f"Minimap mode: {MINIMAP_MODE}")

    state              = State.EXPLORE
    paused             = False
    stuck              = StuckDetector()
    state_start        = time.time()
    last_flee_dir      = (0, 0)
    last_enemy_seen_t  = 0.0
    adx, ady           = 0, 0
    explore_dx, explore_dy = random_explore_delta()
    last_explore_plan_t = 0.0
    recent_explore_angles = []
    blocked_explore_angles = []
    attack_held = False
    last_stuck_log_t = 0.0
    attack_strafe_sign = 1
    last_attack_move_plan_t = 0.0
    attack_move_dx, attack_move_dy = random_explore_delta()
    explore_pause_until = 0.0
    last_jump_t = [0.0]
    enemy_chaser         = EnemyChaser()
    kite_committed_until = 0.0
    kite_cached_move     = None
    last_backtrack_t     = [0.0]
    explore_heatmap      = ExploreHeatMap()
    attack_stuck         = StuckDetector(window=ATTACK_STUCK_WINDOW, min_move=ATTACK_STUCK_MIN_MOVE)
    cached_mask          = None
    companion = CompanionSuppressor()
    minimap_mode = MINIMAP_MODE

    with mss.mss() as sct:
        full_mon = {"top": 0, "left": 0, "width": SCREEN_W, "height": SCREEN_H}

        while True:
            if keyboard.is_pressed('esc'):
                release_all_inputs()
                attack_held = False
                print("EMERGENCY STOP (esc)")
                break
            if keyboard.is_pressed('q'):
                if attack_held:
                    left_mouse_up()
                    attack_held = False
                print("Bot gestoppt.")
                break
            if keyboard.is_pressed('p'):
                paused = not paused
                if paused and attack_held:
                    left_mouse_up()
                    attack_held = False
                print("PAUSE" if paused else "RESUME")
                time.sleep(0.5)
                continue
            if keyboard.is_pressed('f6'):
                minimap_mode = "large" if minimap_mode == "small" else "small"
                print(f"Minimap mode -> {minimap_mode}")
                time.sleep(0.35)
                continue
            if paused:
                time.sleep(0.1)
                continue

            frame   = cv2.cvtColor(np.array(sct.grab(full_mon)), cv2.COLOR_BGRA2BGR)
            minimap_region = get_minimap_region(minimap_mode)
            minimap = cv2.cvtColor(np.array(sct.grab(minimap_region)), cv2.COLOR_BGRA2BGR)

            hp = get_hp_percent(frame)
            if hp < HP_FLEE_THRESHOLD and state != State.FLEE:
                print(f"HP {hp:.0%} -> FLEE")
                state = State.FLEE
                state_start = time.time()

            dots    = find_enemy_dots(minimap)
            center  = minimap_center_from_image(minimap)
            companion.update(dots, center)
            filtered_dots = companion.filter(dots)
            if len(filtered_dots) == 1 and dot_distance(filtered_dots[0], center) <= (COMPANION_TRACK_RADIUS * 0.85):
                filtered_dots = []

            cached_mask = buffered_walkable_mask(minimap, center)

            if len(filtered_dots) > 1 and cached_mask is not None:
                def _dot_score(dot):
                    return ray_walkability_score(cached_mask, center, dot, samples=12) * 100 - dot_distance(dot, center) * 0.3
                nearest = max(filtered_dots, key=_dot_score)
            else:
                nearest = nearest_dot(filtered_dots, center)

            # Enemy detection has highest priority while not fleeing.
            if state != State.FLEE and nearest:
                adx, ady = dot_to_screen_delta(nearest, center)
                last_flee_dir = (-adx, -ady)
                last_enemy_seen_t = time.time()
                if state != State.ATTACK:
                    print("Gegner gefunden -> ATTACK (Prio)")
                    state = State.ATTACK
                    state_start = time.time()
                    continue

            if state == State.EXPLORE:
                if attack_held:
                    left_mouse_up()
                    attack_held = False

                if time.time() < explore_pause_until:
                    sleep_interruptible(0.05)
                    continue

                pt = ctypes.wintypes.POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                stuck.update(pt.x, pt.y)

                explore_heatmap.update(center, minimap.shape[:2])

                replanning_due = (time.time() - last_explore_plan_t) > EXPLORE_REPLAN_SECONDS

                if stuck.is_stuck():
                    now = time.time()
                    if (now - last_stuck_log_t) > STUCK_LOG_COOLDOWN:
                        print("Stuck – neue Minimap-Richtung")
                        last_stuck_log_t = now
                    backtrack_move(explore_dx, explore_dy, last_backtrack_t)
                    current_angle = math.atan2(explore_dy, explore_dx)
                    blocked_explore_angles.append(current_angle)
                    blocked_explore_angles = blocked_explore_angles[-8:]
                    replanning_due = True

                if replanning_due:
                    (explore_dx, explore_dy), target = choose_explore_delta(minimap, center, recent_explore_angles, blocked_explore_angles)
                    if target is not None:
                        angle = math.atan2(target[1] - center[1], target[0] - center[0])
                        recent_explore_angles.append(angle)
                        recent_explore_angles = recent_explore_angles[-6:]
                    last_explore_plan_t = time.time()

                if cached_mask is not None:
                    walk_score = ray_walkability_score(cached_mask, center, (center[0] + explore_dx, center[1] + explore_dy), samples=14)
                    if walk_score < 0.55:
                        (explore_dx, explore_dy), target = choose_explore_delta(minimap, center, recent_explore_angles, blocked_explore_angles)
                        if target is not None:
                            angle = math.atan2(target[1] - center[1], target[0] - center[0])
                            recent_explore_angles.append(angle)
                            recent_explore_angles = recent_explore_angles[-6:]
                        last_explore_plan_t = time.time()

                wasd_move(
                    explore_dx,
                    explore_dy,
                    random.uniform(EXPLORE_MOVE_MIN_SECONDS, EXPLORE_MOVE_MAX_SECONDS),
                )
                maybe_jump_with_movement(explore_dx, explore_dy, last_jump_t)

            elif state == State.ATTACK:
                if not attack_held:
                    left_mouse_down()
                    attack_held = True

                attack_stuck.update(center[0], center[1])
                if attack_stuck.is_stuck():
                    backtrack_move(attack_move_dx, attack_move_dy, last_backtrack_t)
                    enemy_chaser.reset()
                    perp_dx = -attack_move_dy
                    perp_dy = attack_move_dx
                    slide_score_pos = ray_walkability_score(cached_mask, center, (center[0] + perp_dx, center[1] + perp_dy), samples=10) if cached_mask is not None else 0.0
                    slide_score_neg = ray_walkability_score(cached_mask, center, (center[0] - perp_dx, center[1] - perp_dy), samples=10) if cached_mask is not None else 0.0
                    slide_dx, slide_dy = (perp_dx, perp_dy) if slide_score_pos >= slide_score_neg else (-perp_dx, -perp_dy)
                    wasd_move_and_attack(slide_dx, slide_dy, duration=0.18, aim_dx=adx, aim_dy=ady)

                enemy_dist = None
                enemy_dot = None
                if nearest:
                    last_enemy_seen_t = time.time()
                    enemy_dot = nearest
                    adx, ady = dot_to_screen_delta(nearest, center)
                    enemy_dist = dot_distance(nearest, center)
                    last_flee_dir = (-adx, -ady)
                enemy_gone = (time.time() - last_enemy_seen_t) > ATTACK_LOST_GRACE
                timed_out  = (time.time() - state_start) > ATTACK_TIMEOUT
                if enemy_gone:
                    if attack_held:
                        left_mouse_up()
                        attack_held = False
                    if ENABLE_LOOT:
                        print("Gegner tot -> LOOT")
                        state = State.LOOT
                    else:
                        print("Gegner tot -> EXPLORE")
                        state = State.EXPLORE
                        explore_pause_until = time.time() + random.uniform(
                            POST_KILL_SCAN_MIN_SECONDS,
                            POST_KILL_SCAN_MAX_SECONDS,
                        )
                    state_start = time.time()
                    continue

                # Timeout should never drop combat while enemies are still visible.
                if timed_out and nearest:
                    state_start = time.time()
                elif timed_out and not nearest:
                    if attack_held:
                        left_mouse_up()
                        attack_held = False
                    print("ATTACK Timeout ohne Sichtkontakt -> EXPLORE")
                    state = State.EXPLORE
                    state_start = time.time()
                    continue

                # Keep pressure while repositioning so ranged combat does not pause damage.
                if (time.time() - last_attack_move_plan_t) > ATTACK_MOVE_REPLAN_SECONDS:
                    (attack_move_dx, attack_move_dy), attack_strafe_sign, kite_committed_until, kite_cached_move = choose_attack_move_delta(
                        adx,
                        ady,
                        minimap,
                        center,
                        recent_explore_angles,
                        attack_strafe_sign,
                        enemy_dist,
                        enemy_dot,
                        enemy_chaser=enemy_chaser,
                        kite_committed_until=kite_committed_until,
                        kite_cached_move=kite_cached_move,
                        cached_mask=cached_mask,
                    )
                    last_attack_move_plan_t = time.time()

                wasd_move_and_attack(
                    attack_move_dx,
                    attack_move_dy,
                    duration=random.uniform(0.95, 1.45),
                    aim_dx=adx,
                    aim_dy=ady,
                )
                maybe_jump_with_movement(attack_move_dx, attack_move_dy, last_jump_t)

            elif state == State.LOOT:
                if attack_held:
                    left_mouse_up()
                    attack_held = False

                if (time.time() - state_start) > LOOT_TIMEOUT:
                    state = State.EXPLORE
                    state_start = time.time()
                    continue

                item = find_items_on_screen(frame, item_templates)
                if item:
                    print(f"Item {item} -> pickup")
                    ctypes.windll.user32.SetCursorPos(item[0], item[1])
                    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
                    sleep_interruptible(0.05)
                    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
                    sleep_interruptible(2.0)
                else:
                    state = State.EXPLORE
                    state_start = time.time()

            elif state == State.FLEE:
                if attack_held:
                    left_mouse_up()
                    attack_held = False

                flee_delta = None
                if cached_mask is not None:
                    lx, ly = last_flee_dir if last_flee_dir != (0, 0) else random_explore_delta()
                    flee_len = math.hypot(lx, ly)
                    if flee_len > 1e-6:
                        norm_x = lx / flee_len
                        norm_y = ly / flee_len
                        cx_mm, cy_mm = center
                        flee_target = (
                            int(max(0, min(minimap.shape[1] - 1, cx_mm + norm_x * 40))),
                            int(max(0, min(minimap.shape[0] - 1, cy_mm + norm_y * 40))),
                        )
                        waypoint = _a_star_next_waypoint(cached_mask, center, flee_target)
                        if waypoint is not None:
                            flee_delta = dot_to_screen_delta(waypoint, center)
                if flee_delta is None:
                    fx, fy = last_flee_dir if last_flee_dir != (0, 0) else random_explore_delta()
                    flee_delta = (fx, fy)
                wasd_move(flee_delta[0], flee_delta[1], random.uniform(0.5, 1.0))

                hp = get_hp_percent(frame)
                if hp >= HP_RESUME_THRESHOLD or (time.time() - state_start) > FLEE_MAX_WAIT:
                    print(f"HP {hp:.0%} -> EXPLORE")
                    state = State.EXPLORE
                    state_start = time.time()

            sleep_interruptible(random.uniform(0.05, 0.12))

    print("Bot beendet.")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("Bot per Ctrl+C gestoppt.")
