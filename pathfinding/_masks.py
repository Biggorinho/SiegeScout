"""Walkable-mask construction with adaptive wall buffer."""

import cv2
import numpy as np

from config import WALL_BUFFER_ITERATIONS
from minimap import build_walkable_minimap_mask, connected_walkable_mask
from pathfinding._astar import _a_star_full_path


def adaptive_walkable_mask(minimap_bgr, center, target=None):
    """Walkable mask with adaptive wall buffer.

    Tries WALL_BUFFER_ITERATIONS first (keeps paths wall-distant), then falls back
    to 1 and 0 so narrow corridors are reachable when no full-buffer path exists.
    If target is given, only falls back when A* cannot route to target.
    """
    base = build_walkable_minimap_mask(minimap_bgr)
    base = connected_walkable_mask(base, center)
    if base is None:
        return None
    last_mask = None
    for iterations in dict.fromkeys((WALL_BUFFER_ITERATIONS, max(0, WALL_BUFFER_ITERATIONS - 1), 0)):
        eroded = cv2.erode(base, np.ones((3, 3), dtype=np.uint8), iterations=iterations) if iterations > 0 else base
        last_mask = eroded
        if target is None:
            return eroded
        if _a_star_full_path(eroded, center, target) is not None:
            return eroded
    return last_mask
