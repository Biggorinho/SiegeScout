import cv2
import numpy as np
import ctypes
import math
import time

from config import (
    SCREEN_W, SCREEN_H,
    MAP_DETECT_TIMEOUT_SECONDS, MAP_DETECT_POLL_SECONDS,
    MAP_INTERACT_RETRY_SECONDS, MAP_OPEN_DELAY_SECONDS,
    MAP_SELECT_SETTLE_SECONDS, MAP_NODE_CLICK_Y_OFFSET_FRAC,
    FALLBACK_NODE_X_FRAC, FALLBACK_NODE_Y_FRAC,
    SATANIC_NODE_CLICK_RING_RADIUS_PX,
    SATANIC_NODE_CLICK_X_OFFSET, SATANIC_NODE_CLICK_Y_OFFSET,
    SATANIC_NODE_MIN_SCORE, SATANIC_FIXED_NODES,
    VOTE_RESET_INTERACT_KEY, VOTE_RESET_ESC_HOLD_SECONDS,
    VOTE_RESET_MENU_OPEN_DELAY, VOTE_RESET_CLICK_X_FRAC,
    VOTE_RESET_CLICK_Y_FRAC, VOTE_RESET_CLICK_Y_SWEEP_FRAC,
    VOTE_RESET_CLICK_ATTEMPTS, VOTE_RESET_TO_INTERACT_DELAY,
    VOTE_RESET_SECOND_PASS_DELAY_SECONDS, VOTE_RESET_SECOND_PASS_ATTEMPTS,
    SATANIC_NODE_SCAN_ATTEMPTS, SATANIC_NODE_SCAN_RETRY_SECONDS,
    DEATH_RESPAWN_X_FRAC, DEATH_RESPAWN_Y_FRAC,
    DEATH_RESPAWN_CLICK_ATTEMPTS, DEATH_RESPAWN_RING_RADIUS_PX,
    DEATH_RESPAWN_POST_CLICK_CHECK_SECONDS,
    CORNER_ESCAPE_BURST_STEP_SECONDS,
)
from detectors import grab_frame_bgr, is_world_map_visible, is_death_screen, is_pause_menu_visible
from minimap import buffered_walkable_mask, random_explore_delta
from pathfinding import ray_walkability_score
from input_control import (
    tap_key, sleep_interruptible, left_click_at, release_all_inputs, wasd_move,
)


def wait_for_world_map(sct, full_mon, retry_interact_key=None):
    """Wait until world map is visible; optionally retry interact key while waiting."""
    deadline     = time.time() + MAP_DETECT_TIMEOUT_SECONDS
    next_retry_t = time.time() + MAP_INTERACT_RETRY_SECONDS
    last_frame   = None

    while time.time() < deadline:
        frame      = grab_frame_bgr(sct, full_mon)
        last_frame = frame
        if is_world_map_visible(frame):
            return True, frame

        # If pause menu is open, close it first or interact key cannot open map.
        if is_pause_menu_visible(frame):
            tap_key('esc', hold_time=0.08)
            if not sleep_interruptible(0.22):
                return False, last_frame
            if retry_interact_key:
                tap_key(retry_interact_key, hold_time=0.08)
                if not sleep_interruptible(0.22):
                    return False, last_frame
            continue

        now = time.time()
        if retry_interact_key and now >= next_retry_t:
            tap_key(retry_interact_key, hold_time=0.07)
            next_retry_t = now + MAP_INTERACT_RETRY_SECONDS

        if not sleep_interruptible(MAP_DETECT_POLL_SECONDS):
            return False, last_frame

    return False, last_frame


def attempt_respawn(sct, full_mon):
    """Click respawn button with retries around center until death screen disappears."""
    cx = int(SCREEN_W * DEATH_RESPAWN_X_FRAC)
    cy = int(SCREEN_H * DEATH_RESPAWN_Y_FRAC)
    r  = DEATH_RESPAWN_RING_RADIUS_PX
    offsets = [(0, 0), (0, -r), (r, 0), (0, r), (-r, 0)]

    for i in range(min(DEATH_RESPAWN_CLICK_ATTEMPTS, len(offsets))):
        dx, dy = offsets[i]
        tx = int(max(0, min(SCREEN_W - 1, cx + dx)))
        ty = int(max(0, min(SCREEN_H - 1, cy + dy)))
        left_click_at(tx, ty, hold_time=0.08)
        if not sleep_interruptible(DEATH_RESPAWN_POST_CLICK_CHECK_SECONDS):
            return False
        if not is_death_screen(grab_frame_bgr(sct, full_mon)):
            return True
    return False


def _find_red_hotspot(hsv_roi):
    if hsv_roi is None or hsv_roi.size == 0:
        return None

    red1 = cv2.inRange(hsv_roi, np.array([0, 80, 70]), np.array([18, 255, 255]))
    red2 = cv2.inRange(hsv_roi, np.array([160, 80, 70]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red1, red2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (25 <= area <= 20000):
            continue
        per = cv2.arcLength(cnt, True)
        if per <= 0:
            continue
        circularity = 4.0 * math.pi * area / (per * per)
        if circularity < 0.02:
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        bx, by, bw, bh = cv2.boundingRect(cnt)
        patch = hsv_roi[max(0, cy - bh):min(hsv_roi.shape[0], cy + bh + 1),
                        max(0, cx - bw):min(hsv_roi.shape[1], cx + bw + 1)]
        if patch.size == 0:
            continue
        red_pixels = np.logical_or(
            np.logical_and(patch[:, :, 0] <= 18, patch[:, :, 1] >= 80),
            np.logical_and(patch[:, :, 0] >= 160, patch[:, :, 1] >= 80),
        )
        red_ratio = float(np.count_nonzero(red_pixels)) / float(red_pixels.size)
        sat_mean = float(np.mean(patch[:, :, 1]))
        val_mean = float(np.mean(patch[:, :, 2]))
        score = area * (0.9 + red_ratio * 2.5) + circularity * 400.0 + sat_mean * 1.2 + val_mean * 0.4
        if score > best_score:
            best_score = score
            best = (cx, cy, area, red_ratio, score)

    return best


def _detect_blue_node_centers(frame_bgr):
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([92, 70, 70]), np.array([136, 255, 255]))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (8 <= area <= 700):
            continue
        per = cv2.arcLength(cnt, True)
        if per <= 0:
            continue
        circularity = 4.0 * math.pi * area / (per * per)
        if circularity < 0.18:
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))
    return centers


def _estimate_map_offset_from_nodes(frame_bgr):
    """Estimate map drag offset by matching detected blue nodes to fixed anchors."""
    detected = _detect_blue_node_centers(frame_bgr)
    if not detected:
        return 0, 0, 0

    deltas = []
    for _, (fx, fy) in SATANIC_FIXED_NODES.items():
        best = None
        best_d2 = None
        for dx, dy in detected:
            d2 = (dx - fx) ** 2 + (dy - fy) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = (dx, dy)
        if best is None:
            continue
        dist = math.sqrt(best_d2)
        if dist > 130:
            continue
        deltas.append((best[0] - fx, best[1] - fy))

    if len(deltas) < 4:
        return 0, 0, len(deltas)

    dxs = np.array([d[0] for d in deltas], dtype=np.float32)
    dys = np.array([d[1] for d in deltas], dtype=np.float32)
    mdx = float(np.median(dxs))
    mdy = float(np.median(dys))

    inliers = []
    for dx, dy in deltas:
        if abs(dx - mdx) <= 24 and abs(dy - mdy) <= 24:
            inliers.append((dx, dy))

    if len(inliers) >= 3:
        mdx = float(np.mean([d[0] for d in inliers]))
        mdy = float(np.mean([d[1] for d in inliers]))

    mdx = max(-300.0, min(300.0, mdx))
    mdy = max(-220.0, min(220.0, mdy))
    return int(round(mdx)), int(round(mdy)), len(inliers) if inliers else len(deltas)


def _score_satanic_fixed_node(hsv, x, y, core_radius=6, ring_radius=14):
    h, w = hsv.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None

    x0 = max(0, x - ring_radius)
    x1 = min(w, x + ring_radius + 1)
    y0 = max(0, y - ring_radius)
    y1 = min(h, y + ring_radius + 1)
    patch = hsv[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
    px, py = x - x0, y - y0
    dist2 = (xx - px) ** 2 + (yy - py) ** 2
    core_m = dist2 <= (core_radius ** 2)
    ring_m = np.logical_and(dist2 > (core_radius ** 2), dist2 <= (ring_radius ** 2))

    core = patch[core_m]
    ring = patch[ring_m]
    if core.size == 0 or ring.size == 0:
        return None

    core_red = np.logical_or(
        np.logical_and(core[:, 0] <= 22, core[:, 1] >= 70),
        np.logical_and(core[:, 0] >= 158, core[:, 1] >= 70),
    )
    ring_red = np.logical_or(
        np.logical_and(ring[:, 0] <= 22, ring[:, 1] >= 70),
        np.logical_and(ring[:, 0] >= 158, ring[:, 1] >= 70),
    )
    core_purple = np.logical_and(core[:, 0] >= 128, core[:, 0] <= 160)
    ring_purple = np.logical_and(ring[:, 0] >= 128, ring[:, 0] <= 160)
    core_blue = np.logical_and(core[:, 0] >= 90, core[:, 0] <= 132)
    ring_blue = np.logical_and(ring[:, 0] >= 90, ring[:, 0] <= 132)

    core_red_ratio = float(np.count_nonzero(core_red)) / float(core.shape[0])
    ring_red_ratio = float(np.count_nonzero(ring_red)) / float(ring.shape[0])
    core_purple_ratio = float(np.count_nonzero(core_purple)) / float(core.shape[0])
    ring_purple_ratio = float(np.count_nonzero(ring_purple)) / float(ring.shape[0])
    core_blue_ratio = float(np.count_nonzero(core_blue)) / float(core.shape[0])
    ring_blue_ratio = float(np.count_nonzero(ring_blue)) / float(ring.shape[0])

    core_sat = float(np.mean(core[:, 1]))
    ring_sat = float(np.mean(ring[:, 1]))
    core_val = float(np.mean(core[:, 2]))

    # Red highlight is the strongest satanic signal; purple is valid but weaker.
    hot_core_ratio = max(core_red_ratio, core_purple_ratio * 0.62)
    hot_ring_ratio = max(ring_red_ratio, ring_purple_ratio * 0.62)
    hot_dom_core = max(0.0, hot_core_ratio - core_blue_ratio)
    hot_dom_ring = max(0.0, hot_ring_ratio - ring_blue_ratio)

    score = (
        hot_dom_core * 260.0
        + hot_dom_ring * 90.0
        + core_red_ratio * 170.0
        + ring_red_ratio * 70.0
        + core_purple_ratio * 55.0
        + ring_purple_ratio * 20.0
        - core_blue_ratio * 150.0
        - ring_blue_ratio * 60.0
        + core_sat * 0.06
        + ring_sat * 0.03
        + core_val * 0.02
    )

    return (
        score,
        core_red_ratio,
        ring_red_ratio,
        core_purple_ratio,
        ring_purple_ratio,
        core_blue_ratio,
        ring_blue_ratio,
        hot_core_ratio,
        hot_ring_ratio,
    )


def _find_satanic_node_from_fixed_nodes(frame_bgr):
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    best = None
    best_score = -1.0
    best_core_hot = 0.0
    best_core_red = 0.0
    best_ring_red = 0.0
    second_score = -1.0

    for name, pos in SATANIC_FIXED_NODES.items():
        x, y = int(pos[0]), int(pos[1])
        scored = _score_satanic_fixed_node(hsv, x, y)
        if scored is None:
            continue
        score, core_red, ring_red, core_purple, ring_purple, core_blue, ring_blue, hot_core, hot_ring = scored
        if score > best_score:
            second_score = best_score
            best_score = score
            best_core_hot = hot_core
            best_core_red = core_red
            best_ring_red = ring_red
            best = (x, y, name)
        elif score > second_score:
            second_score = score

    if best is None:
        return None
    if best_score < SATANIC_NODE_MIN_SCORE:
        print(f"[SAT] fixed-node best score too low ({best_score:.1f} < {SATANIC_NODE_MIN_SCORE:.1f})")
        return None
    if best_core_hot < 0.20:
        print(f"[SAT] fixed-node best core hot-ratio too low ({best_core_hot:.2f})")
        return None
    margin = best_score - second_score if second_score > -1.0 else 999.0
    # With purple-capable pools, many nodes can score high. Use adaptive margin:
    # if top node is clearly red-highlighted, allow a smaller margin.
    required_margin = 10.0 if (best_core_red >= 0.45 or best_ring_red >= 0.28) else 24.0
    if second_score > -1.0 and margin < required_margin:
        print(
            f"[SAT] fixed-node best margin too low ({margin:.1f} < {required_margin:.1f})"
        )
        return None

    x, y, name = best
    print(
        f"[SAT] fixed-node picked {name} at ({x},{y}) score={best_score:.1f} "
        f"core_hot={best_core_hot:.2f} margin={best_score-second_score:.1f}"
    )
    return x, y


def _find_satanic_node(frame_bgr):
    """Find the highlighted satanic zone node on the world map.

    The map uses normal blue node markers that turn red when the satanic zone is
    active. Score compact node-shaped contours by how red their center patch is,
    rather than looking for a large red glow blob.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    fixed_node = _find_satanic_node_from_fixed_nodes(frame_bgr)
    if fixed_node is not None:
        return fixed_node

    h, w   = frame_bgr.shape[:2]
    x0, x1 = int(w * 0.05), int(w * 0.95)
    y0, y1 = int(h * 0.10), int(h * 0.82)
    roi    = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hotspot = _find_red_hotspot(hsv)
    node_mask = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([180, 255, 255]))
    node_mask = cv2.morphologyEx(node_mask, cv2.MORPH_OPEN,  np.ones((2, 2), dtype=np.uint8), iterations=1)
    node_mask = cv2.morphologyEx(node_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(node_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1.0

    roi_h, roi_w = hsv.shape[:2]
    edge_margin_x = int(roi_w * 0.06)
    edge_margin_y = int(roi_h * 0.04)

    if len(contours) > 120:
        return None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (8 <= area <= 800):
            continue

        per = cv2.arcLength(cnt, True)
        if per <= 0:
            continue

        circularity = 4.0 * math.pi * area / (per * per)
        if circularity < 0.05:
            continue

        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue

        cx_roi = int(M["m10"] / M["m00"])
        cy_roi = int(M["m01"] / M["m00"])

        if cx_roi < edge_margin_x or cx_roi > roi_w - edge_margin_x:
            continue
        if cy_roi < edge_margin_y or cy_roi > roi_h - edge_margin_y:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / float(max(1, bh))
        if aspect < 0.30 or aspect > 3.00:
            continue

        node_size = max(bw, bh)
        if node_size > 70:
            continue

        center_radius = max(2, node_size // 2)
        outer_radius = max(center_radius + 4, int(node_size * 1.7))

        cx0 = max(0, cx_roi - outer_radius)
        cx1 = min(hsv.shape[1], cx_roi + outer_radius + 1)
        cy0 = max(0, cy_roi - outer_radius)
        cy1 = min(hsv.shape[0], cy_roi + outer_radius + 1)
        roi_patch_hsv = hsv[cy0:cy1, cx0:cx1]
        roi_patch_bgr = roi[cy0:cy1, cx0:cx1]
        if roi_patch_bgr.size == 0:
            continue

        yy, xx = np.ogrid[:roi_patch_hsv.shape[0], :roi_patch_hsv.shape[1]]
        px = cx_roi - cx0
        py = cy_roi - cy0
        dist2 = (xx - px) ** 2 + (yy - py) ** 2
        inner2 = center_radius ** 2
        outer2 = outer_radius ** 2
        ring_mask = (dist2 >= inner2) & (dist2 <= outer2)
        core_mask = dist2 < inner2

        ring_pixels = roi_patch_hsv[ring_mask]
        core_pixels = roi_patch_hsv[core_mask]
        if ring_pixels.size == 0 or core_pixels.size == 0:
            continue

        ring_red = np.logical_or(
            np.logical_and(ring_pixels[:, 0] <= 22, ring_pixels[:, 1] >= 70),
            np.logical_and(ring_pixels[:, 0] >= 158, ring_pixels[:, 1] >= 70),
        )
        core_red = np.logical_or(
            np.logical_and(core_pixels[:, 0] <= 22, core_pixels[:, 1] >= 70),
            np.logical_and(core_pixels[:, 0] >= 158, core_pixels[:, 1] >= 70),
        )
        ring_blue = np.logical_and(ring_pixels[:, 0] >= 90, ring_pixels[:, 0] <= 132)
        core_blue = np.logical_and(core_pixels[:, 0] >= 90, core_pixels[:, 0] <= 132)

        ring_red_ratio = float(np.count_nonzero(ring_red)) / float(ring_pixels.shape[0])
        core_red_ratio = float(np.count_nonzero(core_red)) / float(core_pixels.shape[0])
        ring_blue_ratio = float(np.count_nonzero(ring_blue)) / float(ring_pixels.shape[0])
        core_blue_ratio = float(np.count_nonzero(core_blue)) / float(core_pixels.shape[0])

        ring_sat = float(np.mean(ring_pixels[:, 1]))
        ring_val = float(np.mean(ring_pixels[:, 2]))

        red_dominance = max(0.0, ring_red_ratio - ring_blue_ratio)
        core_redness = max(0.0, core_red_ratio - core_blue_ratio)

        if hotspot is not None:
            hx, hy = hotspot[0], hotspot[1]
            dist = math.hypot(cx_roi - hx, cy_roi - hy)
            score = (
                max(0.0, 260.0 - dist) * 7.5
                + red_dominance * 400.0
                + core_redness * 120.0
                + ring_red_ratio * 50.0
                + circularity * 45.0
                + ring_sat * 0.05
                - core_blue_ratio * 35.0
            )
        else:
            score = (
                red_dominance * 650.0
                + core_redness * 180.0
                + ring_red_ratio * 90.0
                + core_red_ratio * 35.0
                + circularity * 60.0
                + ring_sat * 0.07
                + ring_val * 0.03
                - core_blue_ratio * 50.0
            )

        cand_x = cx_roi + x0
        cand_y = cy_roi + y0
        print(
            f"[SAT]   candidate ({cand_x},{cand_y}) area={area:.0f} circ={circularity:.2f} "
            f"size={node_size} ring_red={ring_red_ratio:.2f} core_red={core_red_ratio:.2f} "
            f"ring_blue={ring_blue_ratio:.2f} score={score:.1f}"
        )

        if score > best_score:
            best_score = score
            best = (cand_x, cand_y)

    if best:
        print(f"[SAT] picked {best} score={best_score:.1f}")
    else:
        print("[SAT] no satanic node found – falling back")

    return best


def _click_node_and_choose(x, y):
    y_click = int(max(0, min(SCREEN_H - 1, y + SCREEN_H * MAP_NODE_CLICK_Y_OFFSET_FRAC)))
    left_click_at(x, y_click, hold_time=0.06)
    if not sleep_interruptible(0.16):
        return False
    tap_key(VOTE_RESET_INTERACT_KEY, hold_time=0.06)
    return sleep_interruptible(MAP_SELECT_SETTLE_SECONDS)


def _click_vote_reset_menu_button(vote_x, vote_y, attempts):
    for i in range(attempts):
        y_i = vote_y + int(i * SCREEN_H * VOTE_RESET_CLICK_Y_SWEEP_FRAC)
        left_click_at(vote_x, y_i, hold_time=0.08)
        if not sleep_interruptible(0.10):
            return False
    return True


def _try_select_node_with_offsets(sct, full_mon, x, y, offsets):
    """Try selecting a node with multiple click offsets; return True if map closes."""
    for dx, dy in offsets:
        tx = int(max(0, min(SCREEN_W - 1, x + dx)))
        ty = int(max(0, min(SCREEN_H - 1, y + dy)))
        if not _click_node_and_choose(tx, ty):
            return False
        if not is_world_map_visible(grab_frame_bgr(sct, full_mon)):
            return True
    return False


def _reopen_world_map_for_recenter(sct, full_mon):
    """Close and reopen world map once to recenter map position."""
    print("[SAT] recenter: closing and reopening world map")
    # Only close with Esc if the map is actually open right now.
    # Pressing Esc when no map is open would open the pause menu instead.
    current = grab_frame_bgr(sct, full_mon)
    if is_world_map_visible(current):
        tap_key('esc', hold_time=0.08)
        if not sleep_interruptible(0.45):
            return None
    elif is_pause_menu_visible(current):
        print("[SAT] recenter: closing pause menu before reopening map")
        tap_key('esc', hold_time=0.08)
        if not sleep_interruptible(0.32):
            return None
    else:
        if not sleep_interruptible(0.15):
            return None
    tap_key(VOTE_RESET_INTERACT_KEY, hold_time=0.08)
    if not sleep_interruptible(MAP_OPEN_DELAY_SECONDS + 0.25):
        return None
    map_visible, frame = wait_for_world_map(sct, full_mon, retry_interact_key=VOTE_RESET_INTERACT_KEY)
    if not map_visible:
        print("[SAT] recenter failed: world map not visible")
        return None
    return frame


def run_vote_reset_sequence(sct, full_mon):
    """Open ESC menu, vote reset, open map, pick red satanic node, fallback to fixed node."""
    release_all_inputs()

    # Ensure we start from clean gameplay state (no menus open).
    # Close world map if visible, or close any stray pause menu with a second check.
    frame0 = grab_frame_bgr(sct, full_mon)
    if is_world_map_visible(frame0):
        print("[RST] world map open -> closing before vote reset menu")
        tap_key('esc', hold_time=0.08)
        if not sleep_interruptible(0.45):
            return False
        # Verify world map actually closed; if Esc opened a different menu, close that too.
        frame_check = grab_frame_bgr(sct, full_mon)
        if is_world_map_visible(frame_check):
            print("[RST] map still visible after close attempt – retrying Esc")
            tap_key('esc', hold_time=0.08)
            if not sleep_interruptible(0.35):
                return False
    else:
        # If pause menu is already open, close it so the next ESC opens a fresh menu.
        if is_pause_menu_visible(frame0):
            print("[RST] pause menu open -> closing before vote reset menu")
            tap_key('esc', hold_time=0.08)
            if not sleep_interruptible(0.35):
                return False

    tap_key('esc', hold_time=VOTE_RESET_ESC_HOLD_SECONDS)
    if not sleep_interruptible(VOTE_RESET_MENU_OPEN_DELAY):
        return False

    # If ESC did not open pause menu (timing race), recover once.
    frame_menu_check = grab_frame_bgr(sct, full_mon)
    if not is_pause_menu_visible(frame_menu_check):
        tap_key('esc', hold_time=VOTE_RESET_ESC_HOLD_SECONDS)
        if not sleep_interruptible(VOTE_RESET_MENU_OPEN_DELAY):
            return False

    vote_x = int(SCREEN_W * VOTE_RESET_CLICK_X_FRAC)
    vote_y = int(SCREEN_H * VOTE_RESET_CLICK_Y_FRAC)
    if not _click_vote_reset_menu_button(vote_x, vote_y, VOTE_RESET_CLICK_ATTEMPTS):
        return False
    if not sleep_interruptible(VOTE_RESET_TO_INTERACT_DELAY):
        return False

    tap_key(VOTE_RESET_INTERACT_KEY, hold_time=0.06)
    if not sleep_interruptible(MAP_OPEN_DELAY_SECONDS):
        return False

    map_visible, frame = wait_for_world_map(sct, full_mon, retry_interact_key=VOTE_RESET_INTERACT_KEY)
    if not map_visible:
        if not sleep_interruptible(VOTE_RESET_SECOND_PASS_DELAY_SECONDS):
            return False
        print("[RST] second vote-reset click pass")
        if not _click_vote_reset_menu_button(vote_x, vote_y, VOTE_RESET_SECOND_PASS_ATTEMPTS):
            return False
        if not sleep_interruptible(0.20):
            return False
        tap_key(VOTE_RESET_INTERACT_KEY, hold_time=0.06)
        if not sleep_interruptible(MAP_OPEN_DELAY_SECONDS):
            return False
        map_visible, frame = wait_for_world_map(sct, full_mon, retry_interact_key=VOTE_RESET_INTERACT_KEY)
        if not map_visible:
            return False

    return pick_zone_from_open_map(sct, full_mon, frame)


def pick_zone_from_open_map(sct, full_mon, frame=None):
    """Find and click the satanic node (or fallback) on an already-open world map.

    `frame` is an optional pre-captured BGR image of the open map; if None a
    fresh frame is grabbed.  Returns True when the map closes (zone selected).
    """
    if frame is None:
        frame = grab_frame_bgr(sct, full_mon)

    # Use the same deterministic recenter flow as respawn handling before scans.
    recentered_frame = _reopen_world_map_for_recenter(sct, full_mon)
    red_node = None
    # If recenter failed, always grab a fresh frame — never use the pre-vote-reset
    # frame which shows the old zone with no red satanic node highlighted.
    scan_frame = recentered_frame if recentered_frame is not None else grab_frame_bgr(sct, full_mon)
    for attempt in range(1, SATANIC_NODE_SCAN_ATTEMPTS + 1):
        if scan_frame is None:
            scan_frame = grab_frame_bgr(sct, full_mon)
        print(f"[SAT] scan attempt {attempt}/{SATANIC_NODE_SCAN_ATTEMPTS}")
        red_node = _find_satanic_node(scan_frame)
        if red_node is not None:
            break
        if attempt < SATANIC_NODE_SCAN_ATTEMPTS:
            if not sleep_interruptible(SATANIC_NODE_SCAN_RETRY_SECONDS):
                return False
            scan_frame = grab_frame_bgr(sct, full_mon)

    if red_node is not None:
        print(f"[RST] satanic node picked at {red_node}")
        click_x = red_node[0] + SATANIC_NODE_CLICK_X_OFFSET
        click_y = red_node[1] + SATANIC_NODE_CLICK_Y_OFFSET
        r       = SATANIC_NODE_CLICK_RING_RADIUS_PX
        satanic_offsets = [
            (0, 0), (0, -r), (r, 0), (0, r), (-r, 0),
            (r // 2, -r // 2), (-r // 2, -r // 2),
        ]
        if not _try_select_node_with_offsets(sct, full_mon, click_x, click_y, satanic_offsets):
            return False
        if not is_world_map_visible(grab_frame_bgr(sct, full_mon)):
            return True

    fallback_x = int(SCREEN_W * FALLBACK_NODE_X_FRAC)
    fallback_y = int(SCREEN_H * FALLBACK_NODE_Y_FRAC)
    print(f"[RST] fallback node picked at ({fallback_x}, {fallback_y})")
    if not _click_node_and_choose(fallback_x, fallback_y):
        return False

    sleep_interruptible(0.20)
    frame_final = grab_frame_bgr(sct, full_mon)
    if is_world_map_visible(frame_final):
        print("[RST] map still open – ESC then F to retry fallback node")
        tap_key('esc', hold_time=0.10)
        sleep_interruptible(0.70)
        tap_key(VOTE_RESET_INTERACT_KEY, hold_time=0.08)
        sleep_interruptible(MAP_OPEN_DELAY_SECONDS + 0.40)
        frame_retry = grab_frame_bgr(sct, full_mon)
        if is_world_map_visible(frame_retry):
            sleep_interruptible(0.30)
            _click_node_and_choose(fallback_x, fallback_y)
    return True


def perform_corner_escape_burst(
    base_dx, base_dy,
    preferred_dx=None, preferred_dy=None,
    minimap_bgr=None, center=None,
):
    """Try short burst pattern, preferring recent enemy direction and walkable lanes."""

    def _valid(dx, dy):
        return dx is not None and dy is not None and (abs(dx) + abs(dy)) > 0

    if not _valid(base_dx, base_dy):
        base_dx, base_dy = random_explore_delta()

    seed_dx, seed_dy = (preferred_dx, preferred_dy) if _valid(preferred_dx, preferred_dy) else (base_dx, base_dy)

    if minimap_bgr is not None and minimap_bgr.size != 0 and center is not None:
        walk_mask = buffered_walkable_mask(minimap_bgr, center)
        if walk_mask is not None:
            candidates = [
                (seed_dx, seed_dy), (base_dx, base_dy),
                (-seed_dx, -seed_dy), (-base_dx, -base_dy),
            ]
            best_vec, best_score = None, -1.0
            for vx, vy in candidates:
                vlen = math.hypot(vx, vy)
                if vlen < 1e-6:
                    continue
                tx    = int(max(0, min(minimap_bgr.shape[1] - 1, center[0] + (vx / vlen) * 40)))
                ty    = int(max(0, min(minimap_bgr.shape[0] - 1, center[1] + (vy / vlen) * 40)))
                score = ray_walkability_score(walk_mask, center, (tx, ty), samples=14)
                if score > best_score:
                    best_score = score
                    best_vec   = (vx, vy)
            if best_vec is not None:
                seed_dx, seed_dy = best_vec

    sequence = [
        (seed_dx, seed_dy),
        (-seed_dy,  seed_dx),
        ( seed_dy, -seed_dx),
        (-seed_dx, -seed_dy),
    ]
    for dx, dy in sequence:
        wasd_move(dx, dy, duration=CORNER_ESCAPE_BURST_STEP_SECONDS)
