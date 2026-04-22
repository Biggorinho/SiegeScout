import cv2
import numpy as np
import math

from config import (
    HP_BAR, PLAYER_EXCLUSION_RADIUS, PLAYER_EXCLUSION_RADIUS_FRAC,
    ITEM_MATCH_THRESHOLD, SCREEN_W, SCREEN_H,
    COMPANION_WHITE_SUPPRESS_RADIUS, COMPANION_WHITE_MIN_V, COMPANION_WHITE_MAX_S,
)


ENEMY_TIER_SKULL = 4
ENEMY_TIER_YELLOW = 3
ENEMY_TIER_PURPLE = 2
ENEMY_TIER_RED = 1


def _contour_shape_stats(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area <= 0 or perimeter <= 0:
        return None

    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect = float(bw) / float(max(1, bh))
    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / float(max(1.0, hull_area))
    extent = float(area) / float(max(1, bw * bh))

    defects_count = 0
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is not None and len(hull_idx) >= 4:
        try:
            defects = cv2.convexityDefects(cnt, hull_idx)
            if defects is not None:
                defects_count = int(defects.shape[0])
        except cv2.error:
            # Contour has self-intersections; defects cannot be reliably computed.
            defects_count = 0

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "aspect": float(aspect),
        "circularity": float(circularity),
        "solidity": float(solidity),
        "extent": float(extent),
        "defects": defects_count,
    }


def _is_pickaxe_like(stats):
    if stats is None:
        return False
    return (
        6 <= stats["area"] <= 260
        and 0.18 <= stats["circularity"] <= 0.76
        and 0.35 <= stats["solidity"] <= 0.88
        and 0.20 <= stats["extent"] <= 0.86
        and 0.45 <= stats["aspect"] <= 2.20
        and stats["defects"] <= 7
    )


def _is_star_like(stats):
    if stats is None:
        return False
    return (
        6 <= stats["area"] <= 280
        and 0.10 <= stats["circularity"] <= 0.65
        and stats["solidity"] <= 0.78
        and 0.18 <= stats["extent"] <= 0.84
        and 0.50 <= stats["aspect"] <= 2.00
        and stats["defects"] >= 4
    )


def _is_enemy_like(stats, kind):
    if stats is None:
        return False
    if kind == "skull":
        return (
            8 <= stats["area"] <= 220
            and 0.55 <= stats["aspect"] <= 1.75
            and stats["circularity"] >= 0.20
            and stats["solidity"] >= 0.60
            and stats["extent"] >= 0.26
            and stats["defects"] <= 6
        )
    return stats["area"] >= 3 and stats["circularity"] >= 0.42 and stats["solidity"] >= 0.72 and stats["defects"] <= 3


def _append_enemy_dot(dots, cx, cy, tier, kind, cx0, cy0, exclusion_radius):
    if ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5 < exclusion_radius:
        return
    for item in dots:
        px, py = item[0], item[1]
        if (px - cx) ** 2 + (py - cy) ** 2 <= 16:
            if tier > item[2]:
                item[2] = tier
                item[3] = kind
            return
    dots.append([cx, cy, tier, kind])


def classify_enemy_dots(minimap_bgr):
    """Return enemy dots as [x, y, strength_tier, kind] entries."""
    if minimap_bgr is None or minimap_bgr.size == 0:
        return []
    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)

    r1    = cv2.inRange(hsv, np.array([0,   120, 110]), np.array([10,  255, 255]))
    r2    = cv2.inRange(hsv, np.array([170, 120, 110]), np.array([180, 255, 255]))
    p     = cv2.inRange(hsv, np.array([128,  80,  90]), np.array([160, 255, 255]))
    y     = cv2.inRange(hsv, np.array([10,   90, 100]), np.array([38,  255, 255]))
    skull = cv2.inRange(hsv, np.array([0, 0, 148]),     np.array([180, 55, 245]))

    h, w = minimap_bgr.shape[:2]
    cx0  = w // 2
    cy0  = h // 2
    exclusion_radius = max(6, min(PLAYER_EXCLUSION_RADIUS, int(min(w, h) * PLAYER_EXCLUSION_RADIUS_FRAC)))
    # Keep only a tiny center exclusion for skulls; large values suppress real nearby skulls.
    skull_exclusion_radius = max(exclusion_radius + 2, int(min(w, h) * 0.045))
    white_suppress_radius = max(COMPANION_WHITE_SUPPRESS_RADIUS, exclusion_radius + 12)
    dots = []

    def _looks_like_white_companion(cnt, near_center=False):
        if not near_center:
            return False
        x, y0, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            return False
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y0 + bh + pad)
        patch = hsv[y0:y1, x0:x1]
        if patch.size == 0:
            return False
        sat_mean = float(np.mean(patch[:, :, 1]))
        val_mean = float(np.mean(patch[:, :, 2]))
        return val_mean >= COMPANION_WHITE_MIN_V and sat_mean <= COMPANION_WHITE_MAX_S

    def _looks_like_orange_companion(cnt, near_center=False):
        """Detect orange/yellow companion duck in skull mask; reject if found."""
        x, y0, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            return False
        pad = 3
        x0 = max(0, x - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y0 + bh + pad)
        patch = hsv[y0:y1, x0:x1]
        if patch.size == 0:
            return False
        # Check if patch contains substantial orange-yellow hues (like duck beak/body)
        orange_yellow = cv2.inRange(patch, np.array([5, 50, 70]), np.array([35, 255, 255]))
        orange_yellow_ratio = float(np.count_nonzero(orange_yellow)) / float(max(1, orange_yellow.size))
        sat_mean = float(np.mean(patch[:, :, 1]))
        val_mean = float(np.mean(patch[:, :, 2]))
        return orange_yellow_ratio >= 0.08 and sat_mean >= 70.0 and val_mean >= 85.0

    def _looks_like_large_skull_blob(cnt, stats):
        if stats is None:
            return False
        if not (
            8 <= stats["area"] <= 220
            and 0.72 <= stats["aspect"] <= 1.35
            and stats["circularity"] >= 0.22
            and stats["solidity"] >= 0.78
            and 0.22 <= stats["extent"] <= 0.75
            and 0 <= stats["defects"] <= 5
        ):
            return False

        x, y0, bw, bh = cv2.boundingRect(cnt)
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y0 + bh + pad)
        patch = hsv[y0:y1, x0:x1]
        if patch.size == 0:
            return False

        sat_mean = float(np.mean(patch[:, :, 1]))
        val_std = float(np.std(patch[:, :, 2]))
        if sat_mean > 65.0:
            return False
        if val_std < 12.0:
            return False

        return not _looks_like_orange_companion(cnt, True)

    def _looks_like_real_skull(cnt, stats, cx, cy):
        if stats is None:
            return False
        if not (
            8 <= stats["area"] <= 120
            and 0.85 <= stats["aspect"] <= 1.25
            and stats["circularity"] >= 0.42
            and stats["solidity"] >= 0.82
            and 0.26 <= stats["extent"] <= 0.66
            and 0 <= stats["defects"] <= 3
        ):
            return False

        x, y0, bw, bh = cv2.boundingRect(cnt)
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y0 + bh + pad)
        patch = hsv[y0:y1, x0:x1]
        if patch.size == 0:
            return False

        val = patch[:, :, 2].astype(np.float32)
        sat = patch[:, :, 1].astype(np.float32)
        if float(np.mean(sat)) > 55.0:
            return False
        if float(np.std(val)) < 15.0:
            return False

        ph, pw = val.shape[:2]
        if ph < 5 or pw < 5:
            return False

        inner_y0 = ph // 4
        inner_y1 = ph - inner_y0
        inner_x0 = pw // 4
        inner_x1 = pw - inner_x0
        inner = val[inner_y0:inner_y1, inner_x0:inner_x1]
        if inner.size == 0:
            return False

        border_mask = np.ones((ph, pw), dtype=bool)
        border_mask[inner_y0:inner_y1, inner_x0:inner_x1] = False
        border = val[border_mask]
        if border.size == 0:
            return False

        inner_mean = float(np.mean(inner))
        border_mean = float(np.mean(border))
        if border_mean - inner_mean < 22.0:
            return False

        center_y0 = max(0, ph // 2 - 1)
        center_y1 = min(ph, ph // 2 + 2)
        center_x0 = max(0, pw // 2 - 1)
        center_x1 = min(pw, pw // 2 + 2)
        center = val[center_y0:center_y1, center_x0:center_x1]
        if center.size == 0:
            return False
        center_mean = float(np.mean(center))
        if center_mean > 145.0:
            return False

        ring_mask = np.zeros((ph, pw), dtype=bool)
        ring_mask[max(0, ph // 2 - 3):min(ph, ph // 2 + 4), max(0, pw // 2 - 3):min(pw, pw // 2 + 4)] = True
        ring_mask[center_y0:center_y1, center_x0:center_x1] = False
        ring = val[ring_mask]
        if ring.size == 0:
            return False

        bright_ring_ratio = float(np.count_nonzero(ring > 180)) / float(max(1, ring.size))
        if bright_ring_ratio < 0.28:
            return False

        inner_dark_ratio = float(np.count_nonzero(inner < 115)) / float(max(1, inner.size))
        if inner_dark_ratio < 0.22:
            return False

        return True

    # Blobs larger than this (px²) are treated as merged-enemy groups and split
    # via distance-transform peak detection rather than a single centre-of-mass.
    _SINGLE_DOT_MAX_AREA = 65

    def _peaks_in_blob(mask, bx, by, bw, bh):
        """Return (cx, cy) list of distance-transform local maxima inside a bbox."""
        crop = mask[by:by + bh, bx:bx + bw]
        dist = cv2.distanceTransform(crop, cv2.DIST_L2, 3)
        # Suppress non-maxima with a window slightly larger than one dot radius
        kernel = np.ones((9, 9), np.uint8)
        dilated = cv2.dilate(dist, kernel)
        peak_px = np.uint8((dist == dilated) & (dist > 1.0)) * 255
        n, _, _, centroids = cv2.connectedComponentsWithStats(peak_px, connectivity=8)
        return [(int(centroids[i][0]) + bx, int(centroids[i][1]) + by)
                for i in range(1, n)]

    def _append_from_mask(mask, tier, kind, area_min, circularity_min):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            stats = _contour_shape_stats(cnt)
            if stats is None or stats["area"] < area_min:
                continue
            # Hard cap — anything larger than ~50 merged dots is noise / UI element
            if stats["area"] > 4000:
                continue

            if stats["area"] <= _SINGLE_DOT_MAX_AREA:
                # Small blob: use centre-of-mass with circularity guard
                if stats["circularity"] < circularity_min:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] <= 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if kind == "skull":
                    # Real skulls are compact, low-saturation blobs with a dark outline
                    # and bright core. Text fragments and HUD specks usually fail the
                    # local contrast check below.
                    if not _looks_like_real_skull(cnt, stats, cx, cy):
                        continue
                    # Town labels / player-adjacent UI often create white skull-like specks near center.
                    if ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5 <= skull_exclusion_radius:
                        continue
                if _looks_like_white_companion(cnt, True):
                    continue
                if _looks_like_orange_companion(cnt, True):
                    continue
                if not _is_enemy_like(stats, kind):
                    continue
                _append_enemy_dot(dots, cx, cy, tier, kind, cx0, cy0, exclusion_radius)
            else:
                # Large merged blob: split via distance-transform peaks
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if kind == "skull" and not _looks_like_large_skull_blob(cnt, stats):
                    continue
                for cx, cy in _peaks_in_blob(mask, bx, by, bw, bh):
                    _append_enemy_dot(dots, cx, cy, tier, kind, cx0, cy0, exclusion_radius)

    _append_from_mask(skull, ENEMY_TIER_SKULL, "skull", 8, 0.20)
    _append_from_mask(y, ENEMY_TIER_YELLOW, "yellow", 3, 0.45)
    _append_from_mask(p, ENEMY_TIER_PURPLE, "purple", 3, 0.45)
    _append_from_mask(cv2.bitwise_or(r1, r2), ENEMY_TIER_RED, "red", 3, 0.45)

    # Mining veins / ore dots on the minimap can match the yellow enemy hue.
    # Remove anything that sits on top of an ore candidate before the caller
    # turns it into ATTACK state.
    ore_dots = classify_ore_dots(minimap_bgr)
    if ore_dots:
        filtered = []
        for ex, ey, tier, kind in dots:
            keep = True
            for ox, oy, _, _ in ore_dots:
                if (ex - ox) ** 2 + (ey - oy) ** 2 <= 12 * 12:
                    keep = False
                    break
            if keep:
                filtered.append([ex, ey, tier, kind])
        dots = filtered

    item_dots = classify_item_dots(minimap_bgr)
    if item_dots:
        filtered = []
        for ex, ey, tier, kind in dots:
            keep = True
            for ix, iy, _, _ in item_dots:
                if (ex - ix) ** 2 + (ey - iy) ** 2 <= 12 * 12:
                    keep = False
                    break
            if keep:
                filtered.append([ex, ey, tier, kind])
        dots = filtered

    return dots


def find_enemy_dots(minimap_bgr):
    """Return list of (x, y) dot centers in minimap image coordinates."""
    return [(x, y) for x, y, _, _ in classify_enemy_dots(minimap_bgr)]


def get_hp_percent(frame_bgr):
    """Return HP as 0.0–1.0, or None if the HP bar is not visible."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    bar = frame_bgr[HP_BAR["y"]:HP_BAR["y"] + HP_BAR["h"],
                    HP_BAR["x"]:HP_BAR["x"] + HP_BAR["w"]]
    if bar.size == 0:
        return None
    hsv     = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    visible = cv2.inRange(hsv, np.array([0, 15, 15]), np.array([180, 255, 255]))
    if int(np.count_nonzero(visible)) < max(6, int(bar.shape[0] * bar.shape[1] * 0.05)):
        return None
    r1   = cv2.inRange(hsv, np.array([0,   55, 55]), np.array([12,  255, 255]))
    r2   = cv2.inRange(hsv, np.array([168, 55, 55]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(r1, r2)
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return 0.0
    return float(cols[-1] + 1) / HP_BAR["w"]


def find_items_on_screen(frame_bgr, templates):
    """Return (cx, cy) of best item match or None."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h, w = frame_bgr.shape[:2]
    x0, x1 = int(w * 0.03), int(w * 0.97)
    y0, y1 = int(h * 0.08), int(h * 0.92)
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 25, 100]), np.array([180, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)

    best_pt, best_score = None, -1.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        stats = _contour_shape_stats(cnt)
        if stats is None or stats["area"] < 6 or stats["area"] > 360:
            continue
        if not _is_star_like(stats):
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        score = stats["area"] * (1.0 + (1.0 - stats["solidity"])) + stats["defects"] * 18.0
        if score > best_score:
            best_score = score
            best_pt = (cx + x0, cy + y0)

    if best_pt is not None:
        return best_pt

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    for tpl in templates:
        if tpl is None:
            continue
        tw, th = tpl.shape[::-1]
        res  = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score >= ITEM_MATCH_THRESHOLD and score > best_score:
            best_score = score
            best_pt    = (loc[0] + tw // 2, loc[1] + th // 2)
    return best_pt


def grab_frame_bgr(sct, full_mon):
    return cv2.cvtColor(np.array(sct.grab(full_mon)), cv2.COLOR_BGRA2BGR)


def is_world_map_visible(frame_bgr):
    if frame_bgr is None or frame_bgr.size == 0:
        return False
    hsv        = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    blue_nodes = cv2.inRange(hsv, np.array([90,  80, 100]), np.array([132, 255, 255]))
    purple_bg  = cv2.inRange(hsv, np.array([125, 35,  35]), np.array([170, 255, 220]))
    blue_ratio   = float(np.count_nonzero(blue_nodes)) / blue_nodes.size
    purple_ratio = float(np.count_nonzero(purple_bg))  / purple_bg.size
    return blue_ratio > 0.0008 and purple_ratio > 0.10


def is_pause_menu_visible(frame_bgr):
    """Detect ESC pause menu by stacked red button rows in the center area."""
    if frame_bgr is None or frame_bgr.size == 0:
        return False

    h, w = frame_bgr.shape[:2]
    x0, x1 = int(w * 0.28), int(w * 0.72)
    y0, y1 = int(h * 0.28), int(h * 0.78)
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Keep the hue gate but allow darker/desaturated red menu skins.
    r1 = cv2.inRange(hsv, np.array([0, 45, 38]), np.array([20, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([160, 45, 38]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(r1, r2)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bar_count = 0
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < 420:
            continue
        if bw < int((x1 - x0) * 0.16):
            continue
        if not (18 <= bh <= 90):
            continue
        if (bw / max(1, bh)) < 1.8:
            continue
        # Pause menu buttons are mostly centered horizontally.
        cx = x + bw * 0.5
        if abs(cx - (roi.shape[1] * 0.5)) > roi.shape[1] * 0.22:
            continue
        bar_count += 1

    return bar_count >= 2


def is_death_screen(frame_bgr):
    """Detect death overlay by 'You Died' red text and respawn button area."""
    if frame_bgr is None or frame_bgr.size == 0:
        return False

    h, w = frame_bgr.shape[:2]
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    ty0, ty1 = int(h * 0.16), int(h * 0.40)
    tx0, tx1 = int(w * 0.32), int(w * 0.68)
    text_roi  = hsv[ty0:ty1, tx0:tx1]
    if text_roi.size == 0:
        return False
    tr1      = cv2.inRange(text_roi, np.array([0,   130, 105]), np.array([16,  255, 255]))
    tr2      = cv2.inRange(text_roi, np.array([166, 130, 105]), np.array([180, 255, 255]))
    text_red_pixels = int(np.count_nonzero(cv2.bitwise_or(tr1, tr2)))

    by0, by1 = int(h * 0.73), int(h * 0.93)
    bx0, bx1 = int(w * 0.36), int(w * 0.64)
    btn_roi   = hsv[by0:by1, bx0:bx1]
    if btn_roi.size == 0:
        return False
    br1      = cv2.inRange(btn_roi, np.array([0,   100, 70]), np.array([22,  255, 255]))
    br2      = cv2.inRange(btn_roi, np.array([168, 100, 70]), np.array([180, 255, 255]))
    btn_pixels = int(np.count_nonzero(cv2.bitwise_or(br1, br2)))

    return text_red_pixels >= 140 and btn_pixels >= 260


def classify_ore_dots(minimap_bgr):
    """Return ore dots on minimap as [x, y, tier, kind] (pickaxe-like markers)."""
    if minimap_bgr is None or minimap_bgr.size == 0:
        return []

    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    h, w = minimap_bgr.shape[:2]
    cx0, cy0 = w // 2, h // 2
    exclusion_radius = max(8, int(min(w, h) * 0.06))

    # A broad high-saturation mask catches the ore icon first, then shape filters
    # separate pickaxe-like markers from enemy circles/skulls.
    mask = cv2.inRange(hsv, np.array([0, 35, 70]), np.array([180, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)

    dots = []

    def _append_dot(x, y, tier, kind):
        if ((x - cx0) ** 2 + (y - cy0) ** 2) ** 0.5 < exclusion_radius:
            return
        for item in dots:
            px, py = item[0], item[1]
            if (px - x) ** 2 + (py - y) ** 2 <= 20:
                if tier > item[2]:
                    item[2] = tier
                    item[3] = kind
                return
        dots.append([x, y, tier, kind])

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        stats = _contour_shape_stats(cnt)
        if stats is None or stats["area"] < 6 or stats["area"] > 260:
            continue
        if not _is_pickaxe_like(stats):
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        _append_dot(x, y, 3, "pickaxe")

    return dots


def find_ore_nodes_on_screen(frame_bgr):
    """Detect visible ore clusters in world view and return [(x,y,kind,score), ...]."""
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    h, w = frame_bgr.shape[:2]
    x0, x1 = int(w * 0.03), int(w * 0.86)
    y0, y1 = int(h * 0.08), int(h * 0.86)
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 35, 70]), np.array([180, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)

    hits = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        stats = _contour_shape_stats(cnt)
        if stats is None or stats["area"] < 14 or stats["area"] > 900:
            continue
        if not _is_pickaxe_like(stats):
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"]) + x0
        cy = int(M["m01"] / M["m00"]) + y0
        score = float(stats["area"]) * (1.0 + stats["circularity"])
        hits.append((cx, cy, "pickaxe", score))

    dedup = []
    for node in sorted(hits, key=lambda x: x[3], reverse=True):
        x, y, kind, score = node
        keep = True
        for px, py, _, _ in dedup:
            if (px - x) ** 2 + (py - y) ** 2 <= 18 * 18:
                keep = False
                break
        if keep:
            dedup.append(node)
    return dedup


def classify_item_dots(minimap_bgr):
    """Return star-like item markers on the minimap as [x, y, tier, kind]."""
    if minimap_bgr is None or minimap_bgr.size == 0:
        return []

    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    h, w = minimap_bgr.shape[:2]
    cx0, cy0 = w // 2, h // 2
    exclusion_radius = max(8, int(min(w, h) * 0.06))

    mask = cv2.inRange(hsv, np.array([0, 25, 100]), np.array([180, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)

    dots = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        stats = _contour_shape_stats(cnt)
        if stats is None or stats["area"] < 6 or stats["area"] > 280:
            continue
        if not _is_star_like(stats):
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        if ((x - cx0) ** 2 + (y - cy0) ** 2) ** 0.5 < exclusion_radius:
            continue
        dots.append([x, y, 1, "star"])

    return dots


def detect_mining_qte_window(frame_bgr):
    """Detect mining QTE rings and whether white ring is aligned with green timing ring."""
    if frame_bgr is None or frame_bgr.size == 0:
        return {"active": False, "in_window": False, "delta": None}

    h, w = frame_bgr.shape[:2]
    x0, x1 = int(w * 0.34), int(w * 0.66)
    y0, y1 = int(h * 0.32), int(h * 0.76)
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return {"active": False, "in_window": False, "delta": None}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([90, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 52, 255]))

    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)

    def _best_circle(mask, rmin, rmax):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if r < rmin or r > rmax:
                continue
            per = cv2.arcLength(cnt, True)
            if per <= 0:
                continue
            circ = 4.0 * math.pi * area / (per * per)
            score = area * max(0.25, circ)
            if score > best_score:
                best_score = score
                best = (cx, cy, r)
        return best

    g = _best_circle(green, 18, 120)
    w_c = _best_circle(white, 10, 120)
    if g is None or w_c is None:
        return {"active": False, "in_window": False, "delta": None}

    gcx, gcy, gr = g
    wcx, wcy, wr = w_c
    center_delta = math.hypot(gcx - wcx, gcy - wcy)
    if center_delta > 18:
        return {"active": False, "in_window": False, "delta": None}

    radius_delta = abs(float(wr) - float(gr))
    in_window = radius_delta <= 5.0
    return {
        "active": True,
        "in_window": in_window,
        "delta": radius_delta,
        "green_radius": float(gr),
        "white_radius": float(wr),
    }


def find_item_tags_on_screen(frame_bgr):
    """Detect item/loot tags on screen (red/green/cyan colored text labels).
    Return list of (x, y, score) sorted by score descending, deduplicated.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    h, w = frame_bgr.shape[:2]
    x0, x1 = int(w * 0.03), int(w * 0.97)
    y0, y1 = int(h * 0.08), int(h * 0.92)
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Detect red text (item tags can be red)
    red_lower = cv2.inRange(hsv, np.array([0, 80, 100]), np.array([12, 255, 255]))
    red_upper = cv2.inRange(hsv, np.array([168, 80, 100]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(red_lower, red_upper)
    
    # Detect green text
    green = cv2.inRange(hsv, np.array([35, 80, 100]), np.array([85, 255, 255]))
    
    # Detect cyan text
    cyan = cv2.inRange(hsv, np.array([85, 80, 100]), np.array([105, 255, 255]))
    
    mask = cv2.bitwise_or(cv2.bitwise_or(red, green), cyan)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hits = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10 or area > 2000:
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"]) + x0
        cy = int(M["m01"] / M["m00"]) + y0
        score = float(area)
        hits.append((cx, cy, score))
    
    # Deduplicate: keep only hits that are far enough apart
    dedup = []
    for node in sorted(hits, key=lambda x: x[2], reverse=True):
        x, y, score = node
        keep = True
        for px, py, _ in dedup:
            if (px - x) ** 2 + (py - y) ** 2 <= 20 * 20:
                keep = False
                break
        if keep:
            dedup.append(node)
    
    return dedup
