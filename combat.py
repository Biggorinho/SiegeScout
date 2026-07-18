import cv2
import numpy as np
import math
import time

from config import (
    SCREEN_W, SCREEN_H,
    MAP_DETECT_TIMEOUT_SECONDS, MAP_DETECT_POLL_SECONDS,
    VOTE_RESET_MAP_WAIT_SECONDS,
    MAP_INTERACT_RETRY_SECONDS, MAP_OPEN_DELAY_SECONDS,
    MAP_SELECT_SETTLE_SECONDS, MAP_NODE_CLICK_Y_OFFSET_FRAC,
    FALLBACK_NODE_X_FRAC, FALLBACK_NODE_Y_FRAC,
    SATANIC_NODE_CLICK_RING_RADIUS_PX,
    SATANIC_NODE_CLICK_X_OFFSET, SATANIC_NODE_CLICK_Y_OFFSET,
    SATANIC_NODE_MIN_SCORE, SATANIC_FIXED_NODES, SATANIC_NODE_CLICK_Y_CORRECTION,
    VOTE_RESET_INTERACT_KEY,
    VOTE_RESET_MENU_SETTLE_SECONDS, VOTE_RESET_CLICK_RETRY_DELAY,
    VOTE_RESET_CLICK_Y_SWEEP_FRAC,
    VOTE_RESET_FIXED_CLICK_X_PX, VOTE_RESET_FIXED_CLICK_Y_PX,
    VOTE_RESET_CLICK_ATTEMPTS,
    SATANIC_NODE_SCAN_ATTEMPTS, SATANIC_NODE_SCAN_RETRY_SECONDS,
    DEATH_RESPAWN_X_FRAC, DEATH_RESPAWN_Y_FRAC,
    DEATH_RESPAWN_CLICK_ATTEMPTS, DEATH_RESPAWN_RING_RADIUS_PX,
    DEATH_RESPAWN_POST_CLICK_CHECK_SECONDS,
    CORNER_ESCAPE_BURST_STEP_SECONDS,
)
from detectors import (
    grab_frame_bgr, is_world_map_visible, is_death_screen,
    is_pause_menu_visible, is_options_menu_visible,
)
from minimap import buffered_walkable_mask, random_explore_delta
from pathfinding import ray_walkability_score
from input_control import (
    tap_key, sleep_interruptible, left_click_at,
    release_all_inputs, wasd_move, move_cursor_to, mouse_scroll,
)


def wait_for_world_map(sct, full_mon, retry_interact_key=None, timeout_seconds=None):
    """Wait until world map is visible; optionally retry interact key while waiting.

    timeout_seconds overrides MAP_DETECT_TIMEOUT_SECONDS — the post-vote-click
    wait needs longer (the map auto-opens only once the vote passes)."""
    if timeout_seconds is None:
        timeout_seconds = MAP_DETECT_TIMEOUT_SECONDS
    deadline     = time.time() + timeout_seconds
    next_retry_t = time.time() + MAP_INTERACT_RETRY_SECONDS
    last_frame   = None

    while time.time() < deadline:
        frame      = grab_frame_bgr(sct, full_mon)
        last_frame = frame
        # Menu check must come FIRST: the pause menu renders over the map
        # without hiding it, so both detectors can be true at once — and in
        # that state the modal menu consumes all clicks, so reporting the map
        # as ready would make the caller click a map that has no focus.
        if is_pause_menu_visible(frame):
            tap_key('esc', hold_time=0.08)
            if not sleep_interruptible(0.22):
                return False, last_frame
            if retry_interact_key:
                tap_key(retry_interact_key, hold_time=0.08)
                if not sleep_interruptible(0.22):
                    return False, last_frame
            continue

        if is_world_map_visible(frame):
            return True, frame

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

    # Estimate how much the world map has been scrolled/dragged from the calibrated
    # positions by voting across all (detected_blue_node, fixed_node) offset pairs.
    # The offset with the most agreeing pairs is the map scroll correction.
    offset_x, offset_y = 0, 0
    detected_blue = _detect_blue_node_centers(frame_bgr)
    if len(detected_blue) >= 3:
        fixed_list = list(SATANIC_FIXED_NODES.values())
        _BUCKET = 25
        _votes: dict = {}
        for sx, sy in detected_blue:
            for fx, fy in fixed_list:
                bx = round((sx - fx) / _BUCKET) * _BUCKET
                by = round((sy - fy) / _BUCKET) * _BUCKET
                _votes[(bx, by)] = _votes.get((bx, by), 0) + 1
        if _votes:
            _best_key, _best_n = max(_votes.items(), key=lambda kv: kv[1])
            if _best_n >= 3:
                offset_x, offset_y = int(_best_key[0]), int(_best_key[1])
                if offset_x != 0 or offset_y != 0:
                    print(f"[SAT] map scroll offset: dx={offset_x} dy={offset_y} ({_best_n} votes)")

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    best = None
    best_score = -1.0
    best_core_hot = 0.0
    best_core_red = 0.0
    best_ring_red = 0.0
    second_score = -1.0

    for name, pos in SATANIC_FIXED_NODES.items():
        x, y = int(pos[0]) + offset_x, int(pos[1]) + offset_y
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
    if best_core_hot < 0.25:
        print(f"[SAT] fixed-node best core hot-ratio too low ({best_core_hot:.2f})")
        return None
    if best_core_red < 0.15:
        print(f"[SAT] fixed-node rejected: not enough true red in core ({best_core_red:.2f} < 0.15)")
        return None
    margin = best_score - second_score if second_score > -1.0 else 999.0
    required_margin = 10.0 if (best_core_red >= 0.45 or best_ring_red >= 0.28) else 28.0
    if second_score > -1.0 and margin < required_margin:
        print(f"[SAT] fixed-node best margin too low ({margin:.1f} < {required_margin:.1f})")
        return None

    x, y, name = best
    print(f"[SAT] fixed-node picked {name} at ({x},{y}) score={best_score:.1f} core_hot={best_core_hot:.2f} margin={best_score-second_score:.1f}")
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
    # Bottom row of zone nodes sits at ~82-85% screen height (Corrupted Cave
    # y=868/1080); the old 0.82 cut them out. 0.90 keeps the F/G hint bar
    # (~96%) excluded.
    y0, y1 = int(h * 0.10), int(h * 0.90)
    roi    = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hotspot = _find_red_hotspot(hsv)
    node_mask = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([180, 255, 255]))
    node_mask = cv2.morphologyEx(node_mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    node_mask = cv2.morphologyEx(node_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(node_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1.0

    roi_h, roi_w = hsv.shape[:2]
    edge_margin_x = int(roi_w * 0.04)
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
        if roi_patch_hsv.size == 0:
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


def _find_pause_menu_button_rows(frame_bgr):
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    h, w = frame_bgr.shape[:2]
    x0, x1 = int(w * 0.30), int(w * 0.70)
    y0, y1 = int(h * 0.30), int(h * 0.82)
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    def _rows_from_mask(mask, min_area=220):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rows = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            if bw < int(roi.shape[1] * 0.20):
                continue
            if bw > int(roi.shape[1] * 0.80):   # too wide = button fused with side clutter
                continue
            if not (12 <= bh <= 135):
                continue
            if (bw / max(1, bh)) < 1.2:
                continue
            # Menu buttons are horizontally centred; reject off-centre red clutter
            # (campfires, skill icons, world glows) that otherwise false-confirms.
            cx = x + bw / 2.0
            if abs(cx - roi.shape[1] / 2.0) > roi.shape[1] * 0.22:
                continue
            rows.append((y0 + y + bh // 2, area, bw))
        return rows

    r1 = cv2.inRange(hsv, np.array([0, 45, 38]), np.array([20, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([160, 45, 38]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(r1, r2)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)

    rows = _rows_from_mask(red)
    if len(rows) < 3:
        broad = cv2.inRange(hsv, np.array([0, 32, 32]), np.array([180, 255, 255]))
        broad = cv2.morphologyEx(broad, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
        broad = cv2.morphologyEx(broad, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)
        rows.extend(_rows_from_mask(broad, min_area=320))

    rows.sort(key=lambda item: item[0])

    # Each button renders as ~2 red bars at full resolution (border + body), so
    # merge bars belonging to the same button into a single centred row. The gap
    # is frame-relative: within-button gaps fall under it, between-button gaps
    # above it, at any resolution.
    merge_dist = max(18, int(h * 0.035))
    deduped = []
    group = []
    for row in rows:
        if group and (row[0] - group[-1][0]) <= merge_dist:
            group.append(row)
        else:
            if group:
                ys = [g[0] for g in group]
                deduped.append((sum(ys) // len(ys),
                                max(g[1] for g in group),
                                max(g[2] for g in group)))
            group = [row]
    if group:
        ys = [g[0] for g in group]
        deduped.append((sum(ys) // len(ys),
                        max(g[1] for g in group),
                        max(g[2] for g in group)))

    return deduped


def _select_vote_reset_row(rows, target_y):
    """Return the Vote Reset row.

    The pause menu is Resume / Vote Reset / Options / Save & Exit, so Vote Reset
    is the 2nd row of the evenly-spaced, similar-width button stack. Returning
    that 2nd row is position-independent — it works wherever the menu renders,
    unlike picking the row nearest a fixed target_y (which lands on Options when
    the menu sits lower in combat zones). Falls back to nearest-target_y only if
    no clean stack is found."""
    # Need the full 4-button stack to know which row is Vote Reset. Fewer rows =
    # can't tell -> return None so the caller uses the safe calibrated Y. NEVER
    # guess a nearest-target row: that can return Save & Exit and close the game.
    if not rows or len(rows) < 4:
        return None

    rows = sorted(rows, key=lambda item: item[0])

    best_window = None
    best_var = None
    for i in range(len(rows) - 3):
        window = rows[i:i + 4]
        y_vals = [row[0] for row in window]
        gaps = [y_vals[j + 1] - y_vals[j] for j in range(3)]
        avg_gap = sum(gaps) / 3.0
        if avg_gap < 18 or avg_gap > 80:
            continue
        widths = [row[2] for row in window]
        if max(widths) / float(max(1, min(widths))) > 1.5:   # menu buttons are uniform width
            continue
        gap_var = sum((gap - avg_gap) ** 2 for gap in gaps) / 3.0
        if best_var is None or gap_var < best_var:
            best_var = gap_var
            best_window = window

    if best_window is not None:
        return best_window[1]  # Resume[0] / >Vote Reset[1]< / Options[2] / Save & Exit[3]

    return None  # no clean stack -> don't guess


def pause_menu_stack_confirmed(frame):
    """True only when the frame shows the clean 4-button pause-menu stack
    (Resume / Vote Reset / Options / Save & Exit). Unlike the red-skin
    detector this cannot fire on a death screen, so it is the one signal
    safe to trust while HP reads critical — the pause menu's dim overlay is
    itself what hides the HP bar and makes HP read 0%."""
    if frame is None:
        return False
    rows = _find_pause_menu_button_rows(frame)
    return _select_vote_reset_row(rows, VOTE_RESET_FIXED_CLICK_Y_PX) is not None


def undo_esc_if_pause_menu_opened(sct, full_mon, settle=0.6):
    """Verify an automatic panel-closing ESC didn't OPEN the pause menu.

    ESC toggles the menu: when a panel detector false-positives on plain
    gameplay (MosArathim sand tripped the gold-legend check on 2026-07-18),
    the 'closing' ESC opens the pause menu on top of live play. The bot then
    keeps aiming/jumping — and a space tap on a hovered button activated
    Save & Exit and exited to the lobby. So after every auto-ESC: look again,
    and if the pause menu is now up, toggle it straight back closed.
    (For a real Options close this also handles the pause menu that Options
    returns to.) Returns True when an undo ESC was sent."""
    if not sleep_interruptible(settle):
        return False
    frame = grab_frame_bgr(sct, full_mon)
    if is_pause_menu_visible(frame) or len(_find_pause_menu_button_rows(frame)) >= 2:
        print("[MENU] auto-ESC left the pause menu up (false-positive panel?) — ESC to close")
        tap_key('esc', hold_time=0.12)
        sleep_interruptible(0.35)
        return True
    return False


_DEBUG_DUMP_MAX = 10
_debug_dump_count = 0


def _dump_debug_frame(frame, prefix):
    """Save a frame to debug/menu_dumps/ for diagnosis. Capped per session; in a
    subdir so the debug recorder's *.png rotation doesn't delete it."""
    global _debug_dump_count
    if _debug_dump_count >= _DEBUG_DUMP_MAX or frame is None:
        return
    try:
        import os
        import time as _time
        d = os.path.join("debug", "menu_dumps")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{prefix}_{int(_time.time())}.png")
        if cv2.imwrite(path, frame):
            _debug_dump_count += 1
            print(f"[RST] saved debug frame -> {path}")
    except Exception:
        pass


def _vote_reset_row_is_active(frame, click_y):
    """True when the Vote Reset bar at click_y renders its normal red skin.

    The game disables Vote Reset while enemies are nearby (dark skin, red
    coverage ~0.05 vs ~0.66 active, measured on 2026-07-11 dumps) and
    silently ignores clicks on it — clicking anyway just burns the fail
    streak into lockouts. A disabled bar therefore means our own enemy
    detection missed something; callers skip and go find it instead."""
    if frame is None or frame.size == 0:
        return True  # can't judge — don't block the click path
    h, w = frame.shape[:2]
    half = max(10, int(h * 0.019))
    y0, y1 = max(0, int(click_y) - half), min(h, int(click_y) + half)
    x0, x1 = int(w * 0.44), int(w * 0.56)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return True
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    r1 = cv2.inRange(hsv, np.array([0, 45, 38]), np.array([20, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([160, 45, 38]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(r1, r2)
    return (float(np.count_nonzero(red)) / red.size) >= 0.25


def _click_vote_reset_menu_button(sct, full_mon, attempts):
    frame  = grab_frame_bgr(sct, full_mon)
    rows   = _find_pause_menu_button_rows(frame)
    vote_x = VOTE_RESET_FIXED_CLICK_X_PX
    vote_y = VOTE_RESET_FIXED_CLICK_Y_PX

    # Click the confidently-detected Vote Reset row (2nd of the 4-button stack);
    # otherwise the calibrated fixed Y — which is the known Vote Reset position, so
    # it is safe (never Save & Exit, the bottom button). Save & Exit can only be
    # reached by a guessed row, which _select_vote_reset_row no longer returns.
    selected = _select_vote_reset_row(rows, vote_y) if rows else None
    if selected is not None:
        click_y = int(selected[0])
        print(f"[RST] vote reset = stack 2nd row y={click_y} ({len(rows)} rows)")
    else:
        click_y = vote_y
        print(f"[RST] vote reset fallback fixed y={vote_y} ({len(rows)} rows)")
        _dump_debug_frame(frame, f"pausemenu_{len(rows)}rows")

    if not _vote_reset_row_is_active(frame, click_y):
        print(f"[RST] vote reset button disabled (enemies near / safe zone / cooldown) at y={click_y} -> unavailable")
        _dump_debug_frame(frame, "votereset_disabled")
        return "unavailable"

    # settle_time lets the menu register the hover before the press — a
    # mouse-down in the same frame as the cursor teleport can be dropped.
    left_click_at(vote_x, click_y, hold_time=0.08, settle_time=0.15)
    if not sleep_interruptible(0.12):
        return False
    post_frame = grab_frame_bgr(sct, full_mon)
    if is_options_menu_visible(post_frame):
        print("[RST] click landed on Options — ESC back to pause menu")
        tap_key('esc', hold_time=0.12)
        sleep_interruptible(0.55)
        return False
    # Verify the menu reacted: a successful vote closes it. If it is still up
    # with the button still active, the click was lost — retry once.
    if not sleep_interruptible(0.40):
        return False
    post_frame = grab_frame_bgr(sct, full_mon)
    if is_pause_menu_visible(post_frame) and _vote_reset_row_is_active(post_frame, click_y):
        print("[RST] vote reset click had no effect -> hover-settle retry")
        left_click_at(vote_x, click_y, hold_time=0.10, settle_time=0.25)
        if not sleep_interruptible(0.30):
            return False
    return True


def _force_vote_reset_click_sequence(sct, full_mon, attempts, esc_already_sent=False):
    """Blind-click the vote-reset button when the pause detector can't confirm.

    Clicks the fixed button positions + interact even without a confirmed menu
    skin, verifying success by whether the world map opens.

    `esc_already_sent`: when the caller (`_ensure_pause_menu_open`) has already
    pressed the single menu-opening ESC, do NOT press another one here — ESC
    toggles the menu, so a second press would close the menu we're about to
    click.
    """
    print("[RST] force vote-reset click fallback")
    frame = grab_frame_bgr(sct, full_mon)
    if is_world_map_visible(frame):
        tap_key('esc', hold_time=0.10)
        if not sleep_interruptible(0.45):
            return False
        frame = grab_frame_bgr(sct, full_mon)

    if not esc_already_sent and not is_pause_menu_visible(frame):
        tap_key('esc', hold_time=0.12)
        if not sleep_interruptible(0.65):
            return False

    base_x = VOTE_RESET_FIXED_CLICK_X_PX
    base_y = VOTE_RESET_FIXED_CLICK_Y_PX
    dx = int(SCREEN_W * 0.03)
    dy = int(SCREEN_H * 0.022)
    click_points = [
        (base_x, base_y), (base_x, base_y + dy), (base_x, base_y - dy),
        (base_x - dx, base_y), (base_x + dx, base_y),
        (base_x - dx, base_y + dy), (base_x + dx, base_y + dy),
        (base_x - dx, base_y - dy), (base_x + dx, base_y - dy),
    ]

    total_clicks = max(1, attempts) * 2
    for i in range(min(total_clicks, len(click_points))):
        cx, cy = click_points[i]
        cx = int(max(0, min(SCREEN_W - 1, cx)))
        cy = int(max(0, min(SCREEN_H - 1, cy)))
        print(f"[RST] fallback blind click {i+1}/{min(total_clicks, len(click_points))} at ({cx},{cy})")
        left_click_at(cx, cy, hold_time=0.08)
        if not sleep_interruptible(0.12):
            return False
        post_frame = grab_frame_bgr(sct, full_mon)
        if is_options_menu_visible(post_frame):
            print("[RST] fallback click landed on Options — ESC back to pause menu")
            tap_key('esc', hold_time=0.12)
            if not sleep_interruptible(0.55):
                return False
            continue
        tap_key(VOTE_RESET_INTERACT_KEY, hold_time=0.06)
        if not sleep_interruptible(MAP_OPEN_DELAY_SECONDS):
            return False
        map_visible, frame = wait_for_world_map(sct, full_mon, retry_interact_key=VOTE_RESET_INTERACT_KEY)
        if map_visible:
            return pick_zone_from_open_map(sct, full_mon, frame)

        retry_frame = grab_frame_bgr(sct, full_mon)
        if not is_pause_menu_visible(retry_frame) and not is_world_map_visible(retry_frame):
            tap_key('esc', hold_time=0.12)
            if not sleep_interruptible(0.55):
                return False

    return False


def _ensure_pause_menu_open(sct, full_mon):
    """Open the pause menu with a SINGLE ESC (ESC toggles it).

    Root cause of the old ~9x/run "could not open pause menu" failures: this
    used to press ESC up to twice. When the red-bar detector missed a menu
    that had just opened after the first ESC (0.65s is too early for the skin
    to fade in, and a lava-red zone fights a red detector), the second ESC
    TOGGLED the menu closed — so it always ended closed, and the confirm-fail
    dump captured that closed frame, hiding the real state. The blind-click
    fallback then reopened it with its own (odd) ESC and succeeded, proving a
    single ESC from gameplay reliably opens the menu.

    So: press ESC once, poll while the skin fades in, and NEVER press a second
    (closing) ESC. If detection still misses it, the menu is open anyway — the
    caller blind-clicks it without re-pressing ESC.
    """
    frame = grab_frame_bgr(sct, full_mon)

    if is_pause_menu_visible(frame) or len(_find_pause_menu_button_rows(frame)) >= 2:
        print("[RST] pause menu already open")
        return True

    if is_world_map_visible(frame):
        tap_key('esc', hold_time=0.10)  # close the map back to gameplay first
        sleep_interruptible(0.45)

    print("[RST] pressing ESC to open pause menu")
    tap_key('esc', hold_time=0.12)

    # The menu skin fades in — poll instead of a single early grab.
    for _ in range(4):
        sleep_interruptible(0.30)
        frame = grab_frame_bgr(sct, full_mon)
        if is_pause_menu_visible(frame) or len(_find_pause_menu_button_rows(frame)) >= 2:
            print("[RST] pause menu confirmed")
            return True

    # Menu is open (the single ESC opened it) but the detector missed it. Dump
    # for detection tuning — this frame now shows the OPEN menu — and let the
    # caller click without another ESC that would toggle it closed.
    _dump_debug_frame(frame, "pausemenu_open_undetected")
    print("[RST] menu opened but unconfirmed -> blind-click without re-ESC")
    return False


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
    current = grab_frame_bgr(sct, full_mon)
    # Clear a pause menu FIRST: it can sit on top of the open map (both
    # detectors true), and ESC always closes the top-most UI — the old
    # map-first branch spent its only ESC on the menu, then pressed the
    # interact key ('f' = Choose) on a map that was never closed/recentered.
    if is_pause_menu_visible(current):
        tap_key('esc', hold_time=0.08)
        if not sleep_interruptible(0.32):
            return None
        current = grab_frame_bgr(sct, full_mon)
    if is_world_map_visible(current):
        tap_key('esc', hold_time=0.08)
        if not sleep_interruptible(0.45):
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
    # The map pans horizontally with the mouse wheel and the scroll position
    # survives close/reopen — the calibrated node coordinates assume the
    # leftmost position. Hover the map and scroll hard left to normalize.
    move_cursor_to(SCREEN_W // 2, SCREEN_H // 2)
    mouse_scroll(12)
    if not sleep_interruptible(0.35):
        return None
    return grab_frame_bgr(sct, full_mon)


_SAFE_ZONE_WALK_CYCLE = ((1, 0), (0, 1), (-1, 0), (0, -1))


def safe_zone_walk_delta(attempt_index):
    """Rotating short-walk direction between failed waypoint-F attempts.

    F only opens the zone map while standing ON the waypoint; after chasing
    enemies the player often ends a few steps off it (2026-07-18: 60+ F
    presses in place, 10min idle). Walking a nudge in a cycling direction
    before each retry sweeps the area until the waypoint is back in range."""
    return _SAFE_ZONE_WALK_CYCLE[attempt_index % len(_SAFE_ZONE_WALK_CYCLE)]


def handle_stray_world_map(sct, full_mon, target_zone=None):
    """Use a world map found open during gameplay instead of dismissing it.

    An unprompted world map is (virtually always) the zone-select map the game
    auto-opens once a vote reset passes — the old ESC-it-shut net threw that
    map away 2s after a reset was mis-reported as failed, stranding the bot in
    the safe zone for 10min (2026-07-18 22:06). Picking the zone consumes the
    map AND resumes the farm loop; ESC only when the pick fails.
    Returns True when the map was consumed by a successful zone pick."""
    frame = grab_frame_bgr(sct, full_mon)
    if pick_zone_from_open_map(sct, full_mon, frame, target_zone=target_zone) is True:
        return True
    tap_key('esc', hold_time=0.12)
    sleep_interruptible(0.35)
    return False


def run_vote_reset_sequence(sct, full_mon, target_zone=None):
    """ESC menu vote reset (Shift+R removed — never reliably works)."""
    release_all_inputs()
    sleep_interruptible(0.50)

    if not _ensure_pause_menu_open(sct, full_mon):
        print("[RST] menu open but unconfirmed -> blind-click (no re-ESC)")
        return _force_vote_reset_click_sequence(
            sct, full_mon, VOTE_RESET_CLICK_ATTEMPTS, esc_already_sent=True)

    # Park the cursor away from the button stack before scanning: the cursor
    # sprite + hover highlight left on a row by an earlier click corrupt its
    # red bar, and the resulting bad row counts caused wrong-button clicks.
    move_cursor_to(int(SCREEN_W * 0.30), int(SCREEN_H * 0.45))
    sleep_interruptible(VOTE_RESET_MENU_SETTLE_SECONDS + 0.3)

    clicked = _click_vote_reset_menu_button(sct, full_mon, VOTE_RESET_CLICK_ATTEMPTS)
    if clicked == "unavailable":
        # In-game reset cooldown — close the menu we opened and report
        # distinctly so the caller backs off instead of burning fail streaks.
        tap_key('esc', hold_time=0.10)
        sleep_interruptible(0.35)
        return "unavailable"
    if not clicked:
        return False

    tap_key(VOTE_RESET_INTERACT_KEY, hold_time=0.08)
    sleep_interruptible(MAP_OPEN_DELAY_SECONDS + 0.4)

    # The click is verified — the map now only appears once the VOTE PASSES,
    # which takes seconds (other players / auto-resolve). The default 3.2s
    # timeout declared failure 2s before the map opened (2026-07-18 22:05).
    map_visible, frame = wait_for_world_map(
        sct, full_mon, retry_interact_key=VOTE_RESET_INTERACT_KEY,
        timeout_seconds=VOTE_RESET_MAP_WAIT_SECONDS)
    if map_visible:
        return pick_zone_from_open_map(sct, full_mon, frame, target_zone=target_zone)

    print("[RST] vote reset failed after all retries")
    return False


def pick_zone_from_open_map(sct, full_mon, frame=None, target_zone=None):
    """Find and click a zone node on an already-open world map.

    target_zone=None  → auto-detect the satanic (red-highlighted) node.
    target_zone="Name" → skip CV scan, click the known fixed coordinate directly.
    Returns True when the map closes (zone selected).
    """
    if frame is None:
        frame = grab_frame_bgr(sct, full_mon)

    _reopen_world_map_for_recenter(sct, full_mon)
    scan_frame = grab_frame_bgr(sct, full_mon)

    # ── fixed zone: click known coordinate directly, no CV scan ──────────────
    if target_zone is not None:
        coords = SATANIC_FIXED_NODES.get(target_zone)
        if coords is not None:
            fx, fy = coords
            click_y = fy + SATANIC_NODE_CLICK_Y_OFFSET
            print(f"[RST] fixed zone '{target_zone}' at ({fx}, {fy})")
            r = SATANIC_NODE_CLICK_RING_RADIUS_PX
            offsets = [(0, 0), (0, -r), (r, 0), (0, r), (-r, 0),
                       (r // 2, -r // 2), (-r // 2, -r // 2)]
            result = _try_select_node_with_offsets(sct, full_mon, fx, click_y, offsets)
            if not result:
                # interrupted by stop signal — bail out entirely
                if not is_world_map_visible(grab_frame_bgr(sct, full_mon)):
                    return True
                return False
            if not is_world_map_visible(grab_frame_bgr(sct, full_mon)):
                return True
            # All offset clicks processed but map still open → fall through to fallback
            print("[RST] fixed zone clicks missed – trying fallback coordinate")
        else:
            print(f"[RST] unknown zone '{target_zone}' – falling back to satanic scan")
            target_zone = None  # treat as satanic mode

    # ── satanic auto-detect (when target_zone is None) ────────────────────────
    if target_zone is None:
        red_node = None
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
            click_y = red_node[1] + SATANIC_NODE_CLICK_Y_OFFSET + SATANIC_NODE_CLICK_Y_CORRECTION
            r = SATANIC_NODE_CLICK_RING_RADIUS_PX
            satanic_offsets = [(0, 0), (0, -r), (r, 0), (0, r), (-r, 0),
                               (r // 2, -r // 2), (-r // 2, -r // 2)]
            if not _try_select_node_with_offsets(sct, full_mon, click_x, click_y, satanic_offsets):
                return False
            if not is_world_map_visible(grab_frame_bgr(sct, full_mon)):
                return True

    # ── fallback coordinate ───────────────────────────────────────────────────
    # Reaching here means the satanic node scan failed or its click didn't close
    # the map — capture the world map for diagnosis.
    _dump_debug_frame(scan_frame, "worldmap_fail")
    fallback_x = int(SCREEN_W * FALLBACK_NODE_X_FRAC)
    fallback_y = int(SCREEN_H * FALLBACK_NODE_Y_FRAC)
    print(f"[RST] fallback node at ({fallback_x}, {fallback_y})")
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
                tx = int(max(0, min(minimap_bgr.shape[1] - 1, center[0] + (vx / vlen) * 40)))
                ty = int(max(0, min(minimap_bgr.shape[0] - 1, center[1] + (vy / vlen) * 40)))
                score = ray_walkability_score(walk_mask, center, (tx, ty), samples=14)
                if score > best_score:
                    best_score = score
                    best_vec = (vx, vy)
            if best_vec is not None:
                seed_dx, seed_dy = best_vec

    sequence = [
        (seed_dx, seed_dy),
        (-seed_dy, seed_dx),
        (seed_dy, -seed_dx),
        (-seed_dx, -seed_dy),
    ]
    for dx, dy in sequence:
        wasd_move(dx, dy, duration=CORNER_ESCAPE_BURST_STEP_SECONDS)
