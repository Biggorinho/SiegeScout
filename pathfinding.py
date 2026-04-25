import cv2
import numpy as np
import math
import heapq
import random
import time

from config import (
    PATH_GRID_STEP, PATH_WAYPOINT_STEPS, PATH_WAYPOINT_REACH_PX,
    PLAYER_EXCLUSION_RADIUS, EXPLORE_HEATMAP_PENALTY,
    LAST_SEEN_MONSTER_MIN_DIST,
    LAST_SEEN_MONSTER_FALLBACK_DIST_MIN, LAST_SEEN_MONSTER_FALLBACK_DIST_MAX,
    ATTACK_KITE_CHANCE, ATTACK_CHASE_MINIMAP_DIST, ATTACK_FLEE_MINIMAP_DIST,
    ATTACK_BACKPEDAL_STRENGTH, ATTACK_ORBIT_BACKPEDAL_STRENGTH,
    KITE_COMMIT_MIN_SECONDS, KITE_COMMIT_MAX_SECONDS,
    WALL_BUFFER_ITERATIONS,
)
from minimap import (
    build_walkable_minimap_mask, connected_walkable_mask, buffered_walkable_mask,
    dot_distance, dot_to_screen_delta, random_explore_delta,
)


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


def _run_astar(grid, s, t):
    """Core A* search on a boolean grid. Returns came_from dict or None on failure."""
    gh, gw = grid.shape
    neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    open_heap = []
    heapq.heappush(open_heap, (0.0, s))
    g_score  = {s: 0.0}
    came_from = {}
    closed   = set()

    def h(a, b):
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
            if not (0 <= nx < gw and 0 <= ny < gh) or not grid[ny, nx]:
                continue
            nxt       = (nx, ny)
            tentative = g_score[current] + c
            if tentative < g_score.get(nxt, float('inf')):
                came_from[nxt] = current
                g_score[nxt]   = tentative
                heapq.heappush(open_heap, (tentative + h(nxt, t), nxt))

    return came_from if t in came_from else None


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

    came_from = _run_astar(grid, s, t)
    if came_from is None:
        return None

    path = [t]
    cur  = t
    while cur != s:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()

    idx  = min(waypoint_steps, len(path) - 1)
    wx, wy = path[idx]
    return (wx * step + step // 2, wy * step + step // 2)


def _a_star_full_path(mask, center, target, step=PATH_GRID_STEP):
    """Return full A* path as list of minimap (x, y) coords, or None."""
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

    came_from = _run_astar(grid, s, t)
    if came_from is None:
        return None

    path = [t]
    cur  = t
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


def octant_escape_delta(mask, center, current_dx, current_dy, probe_dist=32, samples=14,
                        recent_angles=None):
    """Test 16 directions (22.5° steps); return screen-space delta for the most walkable one.

    Excludes directions within 30° of current_dx/dy (the direction that failed).
    Scores each direction by blending a near probe and a far probe so open corridors
    are preferred over directions that clear the wall but immediately hit another.
    Penalises directions that are the reverse of recent explore angles.
    """
    if mask is None:
        return random_explore_delta()

    current_angle = (
        math.atan2(current_dy, current_dx)
        if (abs(current_dx) + abs(current_dy)) > 1e-6
        else None
    )

    h, w = mask.shape[:2]
    best_angle, best_score = None, -1.0

    for i in range(16):
        angle = i * math.pi / 8
        if current_angle is not None:
            diff = abs((angle - current_angle + math.pi) % (2 * math.pi) - math.pi)
            if diff <= math.pi / 6:  # 30° exclusion — less aggressive than 45°
                continue
        tx = int(max(0, min(w - 1, center[0] + math.cos(angle) * probe_dist)))
        ty = int(max(0, min(h - 1, center[1] + math.sin(angle) * probe_dist)))
        near_score = ray_walkability_score(mask, center, (tx, ty), samples=samples)

        # Far probe rewards directions with genuinely open space beyond the near wall
        far_tx = int(max(0, min(w - 1, center[0] + math.cos(angle) * probe_dist * 1.9)))
        far_ty = int(max(0, min(h - 1, center[1] + math.sin(angle) * probe_dist * 1.9)))
        far_score = ray_walkability_score(mask, center, (far_tx, far_ty), samples=10)

        score = near_score * 0.65 + far_score * 0.35

        if recent_angles:
            min_rev_diff = min(
                abs((angle - (a + math.pi) + math.pi) % (2 * math.pi) - math.pi)
                for a in recent_angles
            )
            score -= max(0.0, (0.55 - min_rev_diff) / 0.55) * 0.40
        if score > best_score:
            best_score = score
            best_angle = angle

    if best_angle is None or best_score < 0.25:
        return random_explore_delta()

    dist = random.randint(100, 200)
    return int(math.cos(best_angle) * dist), int(math.sin(best_angle) * dist)


def path_delta_from_center(minimap_bgr, center, target_dot, terrain_memory=None):
    for buf in dict.fromkeys((WALL_BUFFER_ITERATIONS, max(0, WALL_BUFFER_ITERATIONS - 1), 0)):
        walkable = build_walkable_minimap_mask(minimap_bgr, terrain_memory=terrain_memory)
        walkable = connected_walkable_mask(walkable, center)
        if walkable is None:
            return dot_to_screen_delta(target_dot, center)
        if buf > 0:
            walkable = cv2.erode(walkable, np.ones((3, 3), dtype=np.uint8), iterations=buf)
        waypoint = _a_star_next_waypoint(walkable, center, target_dot)
        if waypoint is not None:
            return dot_to_screen_delta(waypoint, center)
    return dot_to_screen_delta(target_dot, center)


def angle_is_blocked(angle, blocked_angles, tolerance=0.45):
    if not blocked_angles:
        return False
    for blocked in blocked_angles:
        if abs((angle - blocked + math.pi) % (2 * math.pi) - math.pi) < tolerance:
            return True
    return False


def scale_vector(dx, dy, min_len=95, max_len=220):
    """Scale movement vector into a useful WASD magnitude range."""
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return random_explore_delta()
    factor = random.uniform(min_len, max_len) / length
    return int(dx * factor), int(dy * factor)


def steer_delta_away_from_walls(mask, center, move_dx, move_dy, probe_dist=30):
    """Steer a movement vector toward nearby safer lanes when forward is wall-heavy."""
    if mask is None or mask.size == 0:
        return move_dx, move_dy

    mag = math.hypot(move_dx, move_dy)
    if mag < 1e-6:
        return move_dx, move_dy

    base_angle = math.atan2(move_dy, move_dx)
    h, w = mask.shape[:2]
    offsets = [0.0, math.radians(18), -math.radians(18), math.radians(35), -math.radians(35),
               math.radians(55), -math.radians(55), math.radians(72), -math.radians(72)]

    best_angle = base_angle
    best_score = -1.0
    base_score = None

    for off in offsets:
        ang = base_angle + off
        tx = int(max(0, min(w - 1, center[0] + math.cos(ang) * probe_dist)))
        ty = int(max(0, min(h - 1, center[1] + math.sin(ang) * probe_dist)))

        forward = ray_walkability_score(mask, center, (tx, ty), samples=12)

        # Side probes reward lanes that are not hugging one wall.
        lat = math.radians(28)
        ltx = int(max(0, min(w - 1, center[0] + math.cos(ang + lat) * (probe_dist * 0.7))))
        lty = int(max(0, min(h - 1, center[1] + math.sin(ang + lat) * (probe_dist * 0.7))))
        rtx = int(max(0, min(w - 1, center[0] + math.cos(ang - lat) * (probe_dist * 0.7))))
        rty = int(max(0, min(h - 1, center[1] + math.sin(ang - lat) * (probe_dist * 0.7))))
        side_l = ray_walkability_score(mask, center, (ltx, lty), samples=9)
        side_r = ray_walkability_score(mask, center, (rtx, rty), samples=9)

        score = (forward * 0.70) + (min(side_l, side_r) * 0.30)
        if off == 0.0:
            base_score = score
        if score > best_score:
            best_score = score
            best_angle = ang

    # Keep current heading unless the alternative lane is clearly better.
    if base_score is not None and best_score < (base_score + 0.10):
        return move_dx, move_dy

    return int(math.cos(best_angle) * mag), int(math.sin(best_angle) * mag)


def minimap_explore_target(minimap_bgr, center, recent_angles, blocked_angles=None, heatmap=None, terrain_memory=None, sector_tracker=None):
    """Pick an exploration target from bright/open minimap regions, prioritising frontier cells.

    Two-phase approach:
      Phase 1 – contour-based sampling: tests edges of the walkable+fog mask.
      Phase 2 – frontier sampling: directly tests walkable cells adjacent to unexplored
                 fog so the bot can target the explored/unknown boundary even when no
                 contour point samples it.
    """
    if minimap_bgr is None or minimap_bgr.size == 0:
        return None

    walkable = build_walkable_minimap_mask(minimap_bgr, terrain_memory=terrain_memory)
    walkable = connected_walkable_mask(walkable, center)
    if walkable is None:
        return None

    hsv         = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    unknown_fog = cv2.inRange(hsv, np.array([78, 80, 55]), np.array([110, 255, 255]))

    # Frontier mask: walkable pixels immediately adjacent to unexplored fog.
    # These are the highest-value exploration targets — one step from new territory.
    frontier_mask = None
    if np.any(unknown_fog):
        fog_dilated   = cv2.dilate(unknown_fog, np.ones((9, 9), dtype=np.uint8), iterations=2)
        frontier_mask = cv2.bitwise_and(walkable, fog_dilated)

    cx, cy = center
    h, w = walkable.shape[:2]
    clearance_map = cv2.distanceTransform((walkable > 0).astype(np.uint8), cv2.DIST_L2, 3)

    # Exclude the player's immediate vicinity from targeting
    walkable_cnt = walkable.copy()
    cv2.circle(walkable_cnt, (cx, cy), PLAYER_EXCLUSION_RADIUS + 4, 0, -1)

    best_dot, best_score = None, -1.0

    def _eval(tx, ty):
        nonlocal best_dot, best_score
        dist = math.hypot(tx - cx, ty - cy)
        if dist < PLAYER_EXCLUSION_RADIUS + 6:
            return
        if not (0 <= tx < w and 0 <= ty < h):
            return
        angle = math.atan2(ty - cy, tx - cx)
        if angle_is_blocked(angle, blocked_angles):
            return

        if recent_angles:
            min_diff       = min(abs((angle - a + math.pi) % (2 * math.pi) - math.pi) for a in recent_angles)
            repeat_penalty = max(0.0, (0.60 - min_diff) / 0.60) * 80.0
            min_rev_diff   = min(abs((angle - (a + math.pi) + math.pi) % (2 * math.pi) - math.pi) for a in recent_angles)
            repeat_penalty += max(0.0, (0.55 - min_rev_diff) / 0.55) * 65.0
        else:
            repeat_penalty = 0.0

        if sector_tracker is not None:
            repeat_penalty += sector_tracker.penalty(angle)

        path_score = ray_walkability_score(walkable, center, (tx, ty), samples=18)
        if path_score < 0.42:
            return

        clearance = float(clearance_map[ty, tx])
        if clearance < 1.0:
            return

        probe_tx = int(max(0, min(w - 1, cx + math.cos(angle) * min(28.0, dist * 0.4))))
        probe_ty = int(max(0, min(h - 1, cy + math.sin(angle) * min(28.0, dist * 0.4))))
        forward_score = ray_walkability_score(walkable, center, (probe_tx, probe_ty), samples=12)
        if forward_score < 0.38:
            return

        clearance_bonus = min(42.0, clearance * 8.5)
        score = dist * 2.5 + (path_score * 120.0) + (forward_score * 70.0) + clearance_bonus - repeat_penalty

        # Frontier bonus: cell sits right on the edge of unexplored territory
        if frontier_mask is not None and frontier_mask[ty, tx] > 0:
            score += 280.0
        elif unknown_fog[ty, tx] > 0:
            score += 180.0  # inside fog (less precise target, but still novel)

        if heatmap is not None:
            score -= heatmap.penalty(tx, ty)
        if ty > cy + 10 and path_score < 0.75:
            score -= 20.0

        if score > best_score:
            best_score = score
            best_dot   = (tx, ty)

    # ── Phase 1: contour edge sampling ───────────────────────────────────────
    contours, _ = cv2.findContours(walkable_cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 14:
            continue
        points = cnt.reshape(-1, 2)
        if points.size == 0:
            continue
        distances = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        order = np.argsort(distances)[::-1]
        for idx in order[:min(20, len(order))]:  # 20 samples (was 10)
            _eval(int(points[idx][0]), int(points[idx][1]))

    # ── Phase 2: direct frontier cell sampling ───────────────────────────────
    # Explicitly sample walkable cells that touch the fog boundary. These are
    # often interior walkable cells not caught by Phase 1's contour edges.
    if frontier_mask is not None and np.any(frontier_mask):
        fy, fx = np.where(frontier_mask > 0)
        if len(fx) > 0:
            n = min(40, len(fx))
            indices = np.random.choice(len(fx), n, replace=False)
            for i in indices:
                _eval(int(fx[i]), int(fy[i]))

    return best_dot


def choose_explore_delta(minimap_bgr, center, recent_angles, blocked_angles=None, heatmap=None, terrain_memory=None, sector_tracker=None):
    dot = minimap_explore_target(
        minimap_bgr, center, recent_angles, blocked_angles,
        heatmap=heatmap, terrain_memory=terrain_memory, sector_tracker=sector_tracker,
    )
    if dot is None:
        return random_explore_delta(), None
    dx, dy = path_delta_from_center(minimap_bgr, center, dot, terrain_memory=terrain_memory)
    walkable = buffered_walkable_mask(minimap_bgr, center)
    if terrain_memory is not None:
        walkable = terrain_memory.apply(walkable)
    dx, dy = steer_delta_away_from_walls(walkable, center, dx, dy)
    return (dx, dy), dot


def choose_recent_enemy_delta(minimap_bgr, center, recent_enemy_dot=None, recent_enemy_vec=None, terrain_memory=None):
    """Guide exploration toward the last seen enemy position or direction."""
    if minimap_bgr is None or minimap_bgr.size == 0:
        return None, None

    h, w = minimap_bgr.shape[:2]

    def _clamp(x, y):
        return int(max(0, min(w - 1, x))), int(max(0, min(h - 1, y)))

    candidates = []
    if recent_enemy_dot is not None:
        tx, ty = _clamp(*recent_enemy_dot)
        if dot_distance((tx, ty), center) >= LAST_SEEN_MONSTER_MIN_DIST:
            candidates.append((tx, ty))

    if recent_enemy_vec is not None:
        vx, vy = recent_enemy_vec
        vlen   = math.hypot(vx, vy)
        if vlen > 1e-6:
            d  = random.randint(LAST_SEEN_MONSTER_FALLBACK_DIST_MIN, LAST_SEEN_MONSTER_FALLBACK_DIST_MAX)
            tx = int(center[0] + (vx / vlen) * d)
            ty = int(center[1] + (vy / vlen) * d)
            tx, ty = _clamp(tx, ty)
            if dot_distance((tx, ty), center) >= LAST_SEEN_MONSTER_MIN_DIST:
                candidates.append((tx, ty))

    if not candidates:
        return None, None

    walkable   = buffered_walkable_mask(minimap_bgr, center)
    if terrain_memory is not None:
        walkable = terrain_memory.apply(walkable)
    best, best_score = None, -1.0
    vec_fallback = None  # direction-only fallback when ray is blocked by fog/walls

    for target in candidates:
        is_vec_candidate = (recent_enemy_dot is None or target != _clamp(*recent_enemy_dot))
        if walkable is not None:
            path_score = ray_walkability_score(walkable, center, target, samples=18)
            if path_score < 0.20:
                # Ray is mostly blocked — keep as fallback if it came from the vector
                if is_vec_candidate and vec_fallback is None:
                    vec_fallback = target
                continue
            score = (path_score * 100.0) + min(45.0, dot_distance(target, center) * 0.18)
        else:
            score = dot_distance(target, center)

        if score > best_score:
            best_score = score
            best       = target

    # If both scored candidates failed the threshold, still honour the combat direction
    # rather than returning None and letting the caller fall back to generic exploration.
    if best is None:
        best = vec_fallback
    if best is None:
        return None, None

    dx, dy = path_delta_from_center(minimap_bgr, center, best, terrain_memory=terrain_memory)
    dx, dy = steer_delta_away_from_walls(walkable, center, dx, dy)
    return (dx, dy), best


def choose_attack_move_delta(
    aim_dx, aim_dy, minimap_bgr, center, recent_angles, strafe_sign,
    enemy_dist=None, kite_committed_until=0.0, kite_cached_move=None,
    terrain_memory=None,
):
    """Pick movement while attacking: approach when far, orbit at range, flee when close.
    Returns (move_delta, strafe_sign, kite_committed_until, kite_cached_move)."""
    now = time.time()

    if enemy_dist is not None and enemy_dist < ATTACK_FLEE_MINIMAP_DIST and (abs(aim_dx) + abs(aim_dy)) > 10:
        flee_x = -aim_dx * ATTACK_BACKPEDAL_STRENGTH
        flee_y = -aim_dy * ATTACK_BACKPEDAL_STRENGTH
        return scale_vector(flee_x, flee_y, min_len=150, max_len=280), strafe_sign, 0.0, None

    if enemy_dist is not None and enemy_dist > ATTACK_CHASE_MINIMAP_DIST and (abs(aim_dx) + abs(aim_dy)) > 10:
        approach_strafe = strafe_sign if random.random() < 0.7 else -strafe_sign
        move_x = 0.88 * aim_dx + 0.32 * (-aim_dy * approach_strafe)
        move_y = 0.88 * aim_dy + 0.32 * ( aim_dx * approach_strafe)
        approach_delta = scale_vector(move_x, move_y, min_len=95, max_len=190)
        return approach_delta, approach_strafe, now + random.uniform(0.35, 0.60), approach_delta

    if now < kite_committed_until and kite_cached_move is not None:
        return kite_cached_move, strafe_sign, kite_committed_until, kite_cached_move

    if random.random() < ATTACK_KITE_CHANCE and (abs(aim_dx) + abs(aim_dy)) > 20:
        strafe_x  = -aim_dy * strafe_sign
        strafe_y  =  aim_dx * strafe_sign
        move_x    = strafe_x - ATTACK_ORBIT_BACKPEDAL_STRENGTH * aim_dx
        move_y    = strafe_y - ATTACK_ORBIT_BACKPEDAL_STRENGTH * aim_dy
        new_move  = scale_vector(move_x, move_y, min_len=90, max_len=210)
        new_strafe = -strafe_sign
    else:
        explore_move, _ = choose_explore_delta(
            minimap_bgr, center, recent_angles, None, terrain_memory=terrain_memory
        )
        new_move   = scale_vector(explore_move[0], explore_move[1], min_len=95, max_len=220)
        new_strafe = strafe_sign

    commit_until = now + random.uniform(KITE_COMMIT_MIN_SECONDS, KITE_COMMIT_MAX_SECONDS)
    return new_move, new_strafe, commit_until, new_move
