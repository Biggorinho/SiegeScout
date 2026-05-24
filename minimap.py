import cv2
import numpy as np
import math
import random

from config import (
    MINIMAP_SMALL, MINIMAP_LARGE,
    MINIMAP_SMALL_NUDGE, MINIMAP_LARGE_NUDGE,
    SCREEN_W, SCREEN_H,
    PLAYER_EXCLUSION_RADIUS, WALL_BUFFER_ITERATIONS,
)


def normalize_capture_region(region, nudge):
    left   = int(region.get("left",   0) + nudge.get("left",   0))
    top    = int(region.get("top",    0) + nudge.get("top",    0))
    width  = int(region.get("width",  0) + nudge.get("right",  0) - nudge.get("left", 0))
    height = int(region.get("height", 0) + nudge.get("bottom", 0) - nudge.get("top",  0))
    left   = max(0, min(SCREEN_W - 1, left))
    top    = max(0, min(SCREEN_H - 1, top))
    width  = max(1, min(SCREEN_W - left, width))
    height = max(1, min(SCREEN_H - top,  height))
    return {"left": left, "top": top, "width": width, "height": height}


def get_minimap_region(mode):
    if mode == "large":
        return normalize_capture_region(MINIMAP_LARGE, MINIMAP_LARGE_NUDGE)
    return normalize_capture_region(MINIMAP_SMALL, MINIMAP_SMALL_NUDGE)


def minimap_center():
    return MINIMAP_SMALL["width"] // 2, MINIMAP_SMALL["height"] // 2


def minimap_center_from_image(minimap_bgr):
    if minimap_bgr is None or minimap_bgr.size == 0:
        return minimap_center()
    return minimap_bgr.shape[1] // 2, minimap_bgr.shape[0] // 2


def nearest_dot(dots, center):
    if not dots:
        return None
    cx, cy = center
    return min(dots, key=lambda d: (d[0] - cx) ** 2 + (d[1] - cy) ** 2)


def dot_distance(dot, center):
    return ((dot[0] - center[0]) ** 2 + (dot[1] - center[1]) ** 2) ** 0.5


def dot_to_screen_delta(dot, center):
    """Convert minimap dot offset to screen pixel offset."""
    scale = 4.0
    return int((dot[0] - center[0]) * scale), int((dot[1] - center[1]) * scale)


def random_explore_delta():
    angle = random.uniform(0, 2 * math.pi)
    dist  = random.randint(80, 200)
    return int(dist * math.cos(angle)), int(dist * math.sin(angle))


def _build_terrain_wall_mask(minimap_bgr):
    """Detect thin terrain linework on the minimap and mark it as blocked."""
    if minimap_bgr is None or minimap_bgr.size == 0:
        return None

    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)

    # Two wall categories:
    #   1) Mid-grey linework (dim strokes, low S, mid V) — the original heuristic
    #      but narrowed to avoid overlap with cyan fog (S>=80) which must stay
    #      walkable.
    #   2) Bright structures (grave markers, buildings, fences) — high V, low S.
    #      These were missed by the old V<=190 cap.
    linework_dim    = cv2.inRange(hsv, np.array([0, 0,  25]), np.array([180, 70, 190]))
    linework_bright = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 70, 255]))
    linework = cv2.bitwise_or(linework_dim, linework_bright)
    # Lower Canny low-threshold so faint structure edges are picked up too.
    edges    = cv2.Canny(gray, 25, 110)

    wall_mask = cv2.bitwise_or(linework, edges)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    wall_mask = cv2.dilate(wall_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return wall_mask


def build_raw_walkable_minimap_mask(minimap_bgr):
    """Estimate walkable minimap area from the current minimap frame only."""
    if minimap_bgr is None or minimap_bgr.size == 0:
        return None
    hsv         = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    unknown_fog = cv2.inRange(hsv, np.array([78,  80, 55]), np.array([110, 255, 255]))
    explored    = cv2.inRange(hsv, np.array([0,    0, 16]), np.array([180, 255, 255]))
    mask        = cv2.bitwise_or(explored, unknown_fog)
    wall_mask   = _build_terrain_wall_mask(minimap_bgr)
    if wall_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(wall_mask))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((2, 2), dtype=np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return mask


def build_walkable_minimap_mask(minimap_bgr, terrain_memory=None):
    """Estimate walkable minimap area, optionally fused with persistent terrain memory."""
    mask = build_raw_walkable_minimap_mask(minimap_bgr)
    if mask is None:
        return None
    if terrain_memory is not None:
        mask = terrain_memory.apply(mask)
    return mask


def connected_walkable_mask(mask, center):
    """Keep only the walkable component reachable from the player-center."""
    if mask is None or mask.size == 0:
        return None

    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return mask

    h, w   = binary.shape
    cx     = int(max(0, min(w - 1, center[0])))
    cy     = int(max(0, min(h - 1, center[1])))
    seed_x, seed_y = cx, cy

    if binary[cy, cx] == 0:
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return mask
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        i  = int(np.argmin(d2))
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

