"""Low-level A* pathfinding primitives, line-of-sight, and string-pull smoothing.

Pure grid math with no game-specific knowledge. Other pathfinding submodules
(mask construction, navigation heuristics) build on top of this.
"""

import heapq

import numpy as np

from config import PATH_GRID_STEP, PATH_WAYPOINT_STEPS


def ray_walkability_score(mask, center, target, samples=16):
    """Score 0–1 for how much of the path center→target stays on walkable area."""
    if mask is None or mask.size == 0:
        return 0.0
    cx, cy = center
    tx, ty = target
    xs = np.linspace(cx, tx, num=samples, dtype=np.int32)
    ys = np.linspace(cy, ty, num=samples, dtype=np.int32)
    h, w = mask.shape[:2]
    weighted_hits = 0.0
    for i, (x, y) in enumerate(zip(xs, ys)):
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            weighted_hits += 1.0 + (i / max(1, samples - 1)) * 0.5
    return weighted_hits / samples


def _nearest_walkable_cell(grid, gx, gy, radius=6):
    h, w = grid.shape
    if 0 <= gx < w and 0 <= gy < h and grid[gy, gx]:
        return gx, gy
    best, best_d2 = None, None
    for yy in range(max(0, gy - radius), min(h, gy + radius + 1)):
        for xx in range(max(0, gx - radius), min(w, gx + radius + 1)):
            if not grid[yy, xx]:
                continue
            d2 = (xx - gx) ** 2 + (yy - gy) ** 2
            if best_d2 is None or d2 < best_d2:
                best, best_d2 = (xx, yy), d2
    return best


_OCTILE_DIAG = 0.41421356  # √2 − 1


def _run_astar(grid, s, t):
    """Core A* search on a boolean 8-connected grid. Returns came_from dict or None on failure.

    Uses the octile heuristic — admissible and tighter than Euclidean for unit-cost
    grids with diagonal cost √2, so fewer nodes get expanded.
    """
    gh, gw = grid.shape
    neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    open_heap = []
    heapq.heappush(open_heap, (0.0, s))
    g_score  = {s: 0.0}
    came_from = {}
    closed   = set()

    def h(a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + _OCTILE_DIAG * min(dx, dy)

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
            if not (0 <= nx < gw and 0 <= ny < gh) or not grid[ny, nx]:
                continue
            nxt       = (nx, ny)
            tentative = g_score[current] + c
            if tentative < g_score.get(nxt, float('inf')):
                came_from[nxt] = current
                g_score[nxt]   = tentative
                heapq.heappush(open_heap, (tentative + h(nxt, t), nxt))

    return came_from if t in came_from else None


def _astar_grid_path(mask, center, target, step):
    """A* on a step-downsampled walkable mask. Returns list of (gx, gy) grid cells.

    Returns:
        None              — invalid mask, no walkable cell near s/t, or no path exists
        [s]               — start cell == target cell (grid quantisation collapsed them)
        [s, ..., t]       — full grid path including both endpoints
    """
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
        return [s]

    came_from = _run_astar(grid, s, t)
    if came_from is None:
        return None

    path = [t]
    cur  = t
    while cur != s:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def _a_star_next_waypoint(mask, center, target, step=PATH_GRID_STEP, waypoint_steps=PATH_WAYPOINT_STEPS):
    """Return a waypoint on a walkable A* path from center to target (minimap coords)."""
    path = _astar_grid_path(mask, center, target, step)
    if path is None:
        return None
    if len(path) <= 1:
        return target
    idx    = min(waypoint_steps, len(path) - 1)
    gx, gy = path[idx]
    return (gx * step + step // 2, gy * step + step // 2)


def _a_star_full_path(mask, center, target, step=PATH_GRID_STEP):
    """Return full A* path as list of minimap (x, y) coords, or None."""
    path = _astar_grid_path(mask, center, target, step)
    if path is None:
        return None
    if len(path) <= 1:
        return [target]
    return [(gx * step + step // 2, gy * step + step // 2) for gx, gy in path]


def _line_is_clear(mask, p0, p1):
    """Bresenham line walk: True iff every pixel on segment p0→p1 is walkable.

    More accurate than `ray_walkability_score(samples=N)` for line-of-sight checks
    because it visits *every* pixel along the line rather than N evenly-spaced
    samples — a 1-pixel-wide wall slipping between samples is impossible.
    """
    if mask is None or mask.size == 0:
        return False
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    h, w = mask.shape[:2]

    dx =  abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if not (0 <= x0 < w and 0 <= y0 < h) or mask[y0, x0] == 0:
            return False
        if x0 == x1 and y0 == y1:
            return True
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0  += sx
        if e2 <= dx:
            err += dx
            y0  += sy


def smooth_path(path, mask):
    """String-pull an A* path: skip intermediate waypoints when line-of-sight is clear."""
    if not path or len(path) < 3 or mask is None:
        return path
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _line_is_clear(mask, path[i], path[j]):
                break
            j -= 1
        smoothed.append(path[j])
        i = j
    return smoothed
