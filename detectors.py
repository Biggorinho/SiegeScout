import cv2
import numpy as np
import math

from config import (
    HP_BAR, PLAYER_EXCLUSION_RADIUS, PLAYER_EXCLUSION_RADIUS_FRAC,
    ITEM_MATCH_THRESHOLD, SCREEN_W, SCREEN_H,
    COMPANION_WHITE_SUPPRESS_RADIUS, COMPANION_WHITE_MIN_V, COMPANION_WHITE_MAX_S,
    COMPANION_TRACK_RADIUS,
    ITEM_STAR_MIN_SAT, ITEM_STAR_MIN_VAL,
    ITEM_STAR_MAX_CIRCULARITY, ITEM_STAR_MAX_SOLIDITY, ITEM_STAR_MIN_EXTENT,
    ITEM_STAR_MIN_ASPECT, ITEM_STAR_MAX_ASPECT, ITEM_STAR_TELEPORT_CYAN_MAX,
    ROBOFLOW_API_KEY, ROBOFLOW_HEALTHBAR_ENABLED, ROBOFLOW_HEALTHBAR_CLASSES,
    ROBOFLOW_ENEMY_DUNGEON_ENABLED, ROBOFLOW_ENEMY_DUNGEON_CLASSES,
    ROBOFLOW_MINIMAP_VISION_ENABLED, ROBOFLOW_MINIMAP_VISION_CLASSES,
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
            8 <= stats["area"] <= 260
            and 0.45 <= stats["aspect"] <= 1.90
            and stats["circularity"] >= 0.16
            and stats["solidity"] >= 0.52
            and stats["extent"] >= 0.22
            and stats["defects"] <= 12
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
    skull = cv2.inRange(hsv, np.array([0, 0, 138]),     np.array([180, 65, 255]))

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

    def _looks_like_large_skull_blob(cnt, stats, reject_reason=None):
        def _rej(reason):
            if reject_reason is not None:
                reject_reason[0] = reason
            return False

        if stats is None:
            return _rej("missing_stats")
        if not (
            8 <= stats["area"] <= 320
            and 0.62 <= stats["aspect"] <= 1.55
            and stats["circularity"] >= 0.18
            and stats["solidity"] >= 0.68
            and 0.18 <= stats["extent"] <= 0.84
            and 0 <= stats["defects"] <= 10
        ):
            return _rej("large_shape_gate")

        x, y0, bw, bh = cv2.boundingRect(cnt)
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y0 + bh + pad)
        patch = hsv[y0:y1, x0:x1]
        if patch.size == 0:
            return _rej("empty_patch")

        sat_mean = float(np.mean(patch[:, :, 1]))
        val_std = float(np.std(patch[:, :, 2]))
        if sat_mean > 75.0:
            return _rej("large_sat_high")
        if val_std < 8.0:
            return _rej("large_val_std_low")

        if _looks_like_orange_companion(cnt, True):
            return _rej("orange_companion")
        return True

    def _looks_like_real_skull(cnt, stats, cx, cy, reject_reason=None):
        def _rej(reason):
            if reject_reason is not None:
                reject_reason[0] = reason
            return False

        if stats is None:
            return _rej("missing_stats")
        if not (
            8 <= stats["area"] <= 220
            and 0.70 <= stats["aspect"] <= 1.45
            and stats["circularity"] >= 0.24
            and stats["solidity"] >= 0.70
            and 0.20 <= stats["extent"] <= 0.78
            and 0 <= stats["defects"] <= 10
        ):
            return _rej("small_shape_gate")

        x, y0, bw, bh = cv2.boundingRect(cnt)
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y0 + bh + pad)
        patch = hsv[y0:y1, x0:x1]
        if patch.size == 0:
            return _rej("empty_patch")

        val = patch[:, :, 2].astype(np.float32)
        sat = patch[:, :, 1].astype(np.float32)
        if float(np.mean(sat)) > 68.0:
            return _rej("sat_mean_high")
        if float(np.std(val)) < 8.0:
            return _rej("val_std_low")

        ph, pw = val.shape[:2]
        if ph < 5 or pw < 5:
            return _rej("patch_too_small")

        inner_y0 = ph // 4
        inner_y1 = ph - inner_y0
        inner_x0 = pw // 4
        inner_x1 = pw - inner_x0
        inner = val[inner_y0:inner_y1, inner_x0:inner_x1]
        if inner.size == 0:
            return _rej("inner_empty")

        border_mask = np.ones((ph, pw), dtype=bool)
        border_mask[inner_y0:inner_y1, inner_x0:inner_x1] = False
        border = val[border_mask]
        if border.size == 0:
            return _rej("border_empty")

        inner_mean = float(np.mean(inner))
        border_mean = float(np.mean(border))
        if border_mean - inner_mean < 10.0:
            return _rej("border_inner_contrast_low")

        center_y0 = max(0, ph // 2 - 1)
        center_y1 = min(ph, ph // 2 + 2)
        center_x0 = max(0, pw // 2 - 1)
        center_x1 = min(pw, pw // 2 + 2)
        center = val[center_y0:center_y1, center_x0:center_x1]
        if center.size == 0:
            return _rej("center_empty")
        center_mean = float(np.mean(center))
        if center_mean > 180.0:
            return _rej("center_too_bright")

        ring_mask = np.zeros((ph, pw), dtype=bool)
        ring_mask[max(0, ph // 2 - 3):min(ph, ph // 2 + 4), max(0, pw // 2 - 3):min(pw, pw // 2 + 4)] = True
        ring_mask[center_y0:center_y1, center_x0:center_x1] = False
        ring = val[ring_mask]
        if ring.size == 0:
            return _rej("ring_empty")

        bright_ring_ratio = float(np.count_nonzero(ring > 180)) / float(max(1, ring.size))
        if bright_ring_ratio < 0.16:
            return _rej("bright_ring_ratio_low")

        inner_dark_ratio = float(np.count_nonzero(inner < 115)) / float(max(1, inner.size))
        if inner_dark_ratio < 0.12:
            return _rej("inner_dark_ratio_low")

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
                    reject_reason = ["real_gate"]
                    if not _looks_like_real_skull(cnt, stats, cx, cy, reject_reason=reject_reason):
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
                if kind == "skull":
                    reject_reason = ["large_blob_gate"]
                    if not _looks_like_large_skull_blob(cnt, stats, reject_reason=reject_reason):
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

    # Always-on duck/companion safety net: yellow dots near the player center
    # with orange-yellow pixels in their patch are almost certainly the
    # loot-goblin duck or another orange companion sprite. Drop them even when
    # CompanionSuppressor hasn't confirmed yet (it needs N frames; the duck
    # damages the bot in the meantime if not suppressed immediately).
    if dots:
        track_radius_sq = COMPANION_TRACK_RADIUS * COMPANION_TRACK_RADIUS

        def _patch_has_orange(cx, cy, patch_radius=6, min_ratio=0.05):
            x0 = max(0, cx - patch_radius)
            y0 = max(0, cy - patch_radius)
            x1 = min(w, cx + patch_radius)
            y1 = min(h, cy + patch_radius)
            patch = hsv[y0:y1, x0:x1]
            if patch.size == 0:
                return False
            orange = cv2.inRange(patch, np.array([5, 40, 60]), np.array([35, 255, 255]))
            ratio = float(np.count_nonzero(orange)) / float(max(1, orange.size))
            return ratio >= min_ratio

        filtered = []
        for ex, ey, tier, kind in dots:
            if kind == "yellow" and (ex - cx0) ** 2 + (ey - cy0) ** 2 <= track_radius_sq:
                if _patch_has_orange(ex, ey):
                    continue
            filtered.append([ex, ey, tier, kind])
        dots = filtered

    return dots


def find_enemy_dots(minimap_bgr):
    """Return list of (x, y) dot centers in minimap image coordinates."""
    return [(x, y) for x, y, _, _ in classify_enemy_dots(minimap_bgr)]


def build_enemy_color_mask(minimap_bgr):
    """Return a single-channel mask = bitwise-OR of all enemy-tier HSV masks.

    Used by the debug recorder to show what the enemy color detector "sees" —
    skull (low-sat bright), yellow, purple, and red ranges combined."""
    if minimap_bgr is None or minimap_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    r1    = cv2.inRange(hsv, np.array([0,   120, 110]), np.array([10,  255, 255]))
    r2    = cv2.inRange(hsv, np.array([170, 120, 110]), np.array([180, 255, 255]))
    p     = cv2.inRange(hsv, np.array([128,  80,  90]), np.array([160, 255, 255]))
    y     = cv2.inRange(hsv, np.array([10,   90, 100]), np.array([38,  255, 255]))
    skull = cv2.inRange(hsv, np.array([0, 0, 138]),     np.array([180, 65, 255]))
    return cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(r1, r2), cv2.bitwise_or(p, y)), skull)


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


def is_options_menu_visible(frame_bgr):
    """Detect the in-game Options menu by its row of red horizontal tabs near the top.

    The Options screen has six landscape tabs (VIDEO / AUDIO / GAMEPLAY /
    CONTROLS / RESET / BACK) sitting in a single row across the upper edge of
    the screen. Detection requires at least four tabs aligned on the same row
    so that a partially occluded skin or a slightly recoloured active tab
    still triggers, while pause-menu vertical button stacks do not.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return False

    h, w = frame_bgr.shape[:2]
    x0, x1 = int(w * 0.10), int(w * 0.95)
    y0, y1 = int(h * 0.02), int(h * 0.12)
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    r1 = cv2.inRange(hsv, np.array([0, 45, 38]), np.array([20, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([160, 45, 38]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(r1, r2)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tabs = []
    roi_w = roi.shape[1]
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        if bw < int(roi_w * 0.08):
            continue
        if not (18 <= bh <= 60):
            continue
        if (bw / max(1, bh)) < 2.0:
            continue
        tabs.append((y + bh // 2, x + bw // 2))

    if len(tabs) < 4:
        return False

    # Cluster by row centre y (within ±10 px). Largest cluster must have ≥ 4 tabs.
    tabs.sort(key=lambda t: t[0])
    best_cluster = 1
    i = 0
    while i < len(tabs):
        j = i
        while j + 1 < len(tabs) and (tabs[j + 1][0] - tabs[i][0]) <= 10:
            j += 1
        best_cluster = max(best_cluster, j - i + 1)
        i = j + 1

    return best_cluster >= 4


def is_inventory_panel_visible(frame_bgr):
    """Detect any in-game UI panel that exposes the right-side inventory.

    Crafting bench, merchant, stash, plain inventory hotkey — they all open
    the same inventory panel on the right of the screen and show the yellow
    keybinding legend across the very bottom (LMB / RMB / SHIFT / etc.).
    That gold-text band is rarely present during normal gameplay (the
    minimap occupies the top-right and the bottom-right shows the game
    world), so it's a reliable signature. ESC closes any of these panels.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return False

    h, w = frame_bgr.shape[:2]
    # Bottom band where the keybinding legend prints in gold text.
    x0, x1 = int(w * 0.55), w
    y0, y1 = int(h * 0.94), int(h * 0.995)
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Gold/yellow text — narrow hue, mid-to-high saturation, bright.
    yellow = cv2.inRange(hsv, np.array([18, 80, 130]), np.array([45, 255, 255]))
    yellow_ratio = float(np.count_nonzero(yellow)) / yellow.size
    return yellow_ratio > 0.015


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


def _is_item_star_shape(stats):
    """Shape gate for a minimap item star.

    Separates the 5-point star from look-alikes by shape: round dots have high
    circularity/solidity, ore pickaxes high solidity, thin map slivers low
    extent. Deliberately drops the old hard ``defects >= 4`` requirement, which
    silently dropped small/anti-aliased real stars (the recall bug)."""
    if stats is None:
        return False
    return (
        8 <= stats["area"] <= 300
        and stats["circularity"] <= ITEM_STAR_MAX_CIRCULARITY
        and stats["solidity"] <= ITEM_STAR_MAX_SOLIDITY
        and stats["extent"] >= ITEM_STAR_MIN_EXTENT
        and ITEM_STAR_MIN_ASPECT <= stats["aspect"] <= ITEM_STAR_MAX_ASPECT
    )


def _patch_teleport_cyan_ratio(hsv, cx, cy, r=6):
    """Fraction of bright-cyan pixels around (cx, cy) — high for teleport crosses."""
    x0, y0 = max(0, cx - r), max(0, cy - r)
    x1, y1 = cx + r + 1, cy + r + 1
    patch = hsv[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    cyan = cv2.inRange(patch, np.array([85, 90, 120]), np.array([103, 255, 255]))
    return float(np.count_nonzero(cyan)) / float(max(1, cyan.size))


def classify_item_dots(minimap_bgr):
    """Return star-like item markers on the minimap as [x, y, tier, kind].

    Colour-agnostic: rarity recolours the same star glyph, so any saturated
    bright hue is a candidate. Look-alikes are rejected by shape and a
    teleport-cyan colour check (see config ITEM_STAR_* thresholds)."""
    if minimap_bgr is None or minimap_bgr.size == 0:
        return []

    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    h, w = minimap_bgr.shape[:2]
    cx0, cy0 = w // 2, h // 2
    exclusion_radius = max(8, int(min(w, h) * 0.06))

    # Saturated + bright across all hues catches every rarity colour while
    # excluding grey/white map linework and structures.
    mask = cv2.inRange(hsv, np.array([0, ITEM_STAR_MIN_SAT, ITEM_STAR_MIN_VAL]),
                       np.array([180, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)

    dots = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        stats = _contour_shape_stats(cnt)
        if not _is_item_star_shape(stats):
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        if ((x - cx0) ** 2 + (y - cy0) ** 2) ** 0.5 < exclusion_radius:
            continue
        # Teleport crosses are bright cyan and otherwise pass the shape gate.
        if _patch_teleport_cyan_ratio(hsv, x, y) > ITEM_STAR_TELEPORT_CYAN_MAX:
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


# Roboflow Health Bar Detection
_healthbar_client = None
_healthbar_client_init_tried = False


def init_healthbar_detector():
    """Initialize Roboflow health bar detector client."""
    global _healthbar_client, _healthbar_client_init_tried
    
    if _healthbar_client_init_tried:
        return _healthbar_client
    
    _healthbar_client_init_tried = True
    
    if not ROBOFLOW_HEALTHBAR_ENABLED or not ROBOFLOW_API_KEY:
        return None
    
    try:
        from inference_sdk import InferenceHTTPClient
        _healthbar_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
        print("[HEALTHBAR] Roboflow client initialized successfully")
        return _healthbar_client
    except ImportError:
        print("[HEALTHBAR] Warning: inference-sdk not installed. Install with: pip install inference-sdk")
        return None
    except Exception as e:
        print(f"[HEALTHBAR] Failed to initialize Roboflow client: {e}")
        return None


def detect_healthbars(frame_bgr):
    """
    Detect health bars in current frame using Roboflow model.
    
    Returns:
        List of detections, each with:
        - 'x', 'y': center coordinates
        - 'width', 'height': bounding box dimensions
        - 'confidence': detection confidence (0-1)
        - 'class': should be 'healthbars'
    """
    global _healthbar_client
    
    if not ROBOFLOW_HEALTHBAR_ENABLED:
        return []
    
    # Lazy initialization
    if _healthbar_client is None and not _healthbar_client_init_tried:
        _healthbar_client = init_healthbar_detector()
    
    if _healthbar_client is None:
        return []
    
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    
    try:
        # Convert BGR to RGB for Roboflow API
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        result = _healthbar_client.run_workflow(
            workspace_name="discordturtle",
            workflow_id="hero-siege-healthbars-8gez6",
            images={"image": frame_rgb},
            parameters={"classes": "healthbars"},
            use_cache=True
        )
        
        # Extract predictions from result
        predictions = result.get("predictions", {})
        if isinstance(predictions, dict):
            detections = predictions.get("healthbars", [])
        else:
            detections = predictions if isinstance(predictions, list) else []
        
        return detections
    except Exception as e:
        print(f"[HEALTHBAR] Detection error: {e}")
        return []


# Roboflow Enemy & Dungeon Detection
_enemy_dungeon_client = None
_enemy_dungeon_client_init_tried = False


def init_enemy_dungeon_detector():
    """Initialize Roboflow enemy/dungeon detector client."""
    global _enemy_dungeon_client, _enemy_dungeon_client_init_tried
    
    if _enemy_dungeon_client_init_tried:
        return _enemy_dungeon_client
    
    _enemy_dungeon_client_init_tried = True
    
    if not ROBOFLOW_ENEMY_DUNGEON_ENABLED or not ROBOFLOW_API_KEY:
        return None
    
    try:
        from inference_sdk import InferenceHTTPClient
        _enemy_dungeon_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
        print("[ENEMY_DUNGEON] Roboflow client initialized successfully")
        return _enemy_dungeon_client
    except ImportError:
        print("[ENEMY_DUNGEON] Warning: inference-sdk not installed. Install with: pip install inference-sdk")
        return None
    except Exception as e:
        print(f"[ENEMY_DUNGEON] Failed to initialize Roboflow client: {e}")
        return None


def detect_enemy_dungeon_features(frame_bgr):
    """
    Detect enemy, dungeon, and teleport features using Roboflow model.
    
    Returns:
        Dict with keys for each enabled class:
        - 'enemy': list of enemy detections
        - 'teleport': list of teleport detections
        Each detection includes 'x', 'y', 'width', 'height', 'confidence', 'class'
    """
    global _enemy_dungeon_client
    
    if not ROBOFLOW_ENEMY_DUNGEON_ENABLED:
        return {}
    
    # Lazy initialization
    if _enemy_dungeon_client is None and not _enemy_dungeon_client_init_tried:
        _enemy_dungeon_client = init_enemy_dungeon_detector()
    
    if _enemy_dungeon_client is None:
        return {}
    
    if frame_bgr is None or frame_bgr.size == 0:
        return {}
    
    try:
        # Convert BGR to RGB for Roboflow API
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Build parameters string from configured classes
        classes_param = ", ".join(ROBOFLOW_ENEMY_DUNGEON_CLASSES)
        
        result = _enemy_dungeon_client.run_workflow(
            workspace_name="deniss-workspace-ekltv",
            workflow_id="my-first-project-3hra4",
            images={"image": frame_rgb},
            parameters={"classes": classes_param},
            use_cache=True
        )
        
        # Extract and organize predictions by class
        predictions = result.get("predictions", {})
        detections = {}
        
        if isinstance(predictions, dict):
            for class_name in ROBOFLOW_ENEMY_DUNGEON_CLASSES:
                detections[class_name] = predictions.get(class_name, [])
        else:
            # If predictions is a list, organize by class field
            detections = {class_name: [] for class_name in ROBOFLOW_ENEMY_DUNGEON_CLASSES}
            if isinstance(predictions, list):
                for pred in predictions:
                    class_name = pred.get("class", "").lower()
                    if class_name in detections:
                        detections[class_name].append(pred)
        
        return detections
    except Exception as e:
        print(f"[ENEMY_DUNGEON] Detection error: {e}")
        return {}


# Roboflow Minimap Vision Detection
_minimap_vision_client = None
_minimap_vision_client_init_tried = False


def init_minimap_vision_detector():
    """Initialize Roboflow minimap vision detector client."""
    global _minimap_vision_client, _minimap_vision_client_init_tried
    
    if _minimap_vision_client_init_tried:
        return _minimap_vision_client
    
    _minimap_vision_client_init_tried = True
    
    if not ROBOFLOW_MINIMAP_VISION_ENABLED or not ROBOFLOW_API_KEY:
        return None
    
    try:
        from inference_sdk import InferenceHTTPClient
        _minimap_vision_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=ROBOFLOW_API_KEY
        )
        print("[MINIMAP_VISION] Roboflow client initialized successfully")
        return _minimap_vision_client
    except ImportError:
        print("[MINIMAP_VISION] Warning: inference-sdk not installed. Install with: pip install inference-sdk")
        return None
    except Exception as e:
        print(f"[MINIMAP_VISION] Failed to initialize Roboflow client: {e}")
        return None


def detect_minimap_features(minimap_bgr):
    """
    Detect features on the minimap using Roboflow minimapVision model.
    Detects: player, dungeon, items, portals, ores, etc.
    
    Returns:
        Dict with keys for each enabled class:
        - 'player': list of player detections
        - 'dungeon': list of dungeon/zone detections
        - etc. for other enabled classes
        Each detection includes 'x', 'y', 'width', 'height', 'confidence', 'class'
    """
    global _minimap_vision_client
    
    if not ROBOFLOW_MINIMAP_VISION_ENABLED:
        return {}
    
    # Lazy initialization
    if _minimap_vision_client is None and not _minimap_vision_client_init_tried:
        _minimap_vision_client = init_minimap_vision_detector()
    
    if _minimap_vision_client is None:
        return {}
    
    if minimap_bgr is None or minimap_bgr.size == 0:
        return {}
    
    try:
        # Convert BGR to RGB for Roboflow API
        minimap_rgb = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2RGB)
        
        # Build parameters string from configured classes
        classes_param = ", ".join(ROBOFLOW_MINIMAP_VISION_CLASSES)
        
        result = _minimap_vision_client.run_workflow(
            workspace_name="test-jmbou",
            workflow_id="minimapvision",
            images={"image": minimap_rgb},
            parameters={"classes": classes_param},
            use_cache=True
        )
        
        # Extract and organize predictions by class
        predictions = result.get("predictions", {})
        detections = {}
        
        if isinstance(predictions, dict):
            for class_name in ROBOFLOW_MINIMAP_VISION_CLASSES:
                detections[class_name] = predictions.get(class_name, [])
        else:
            # If predictions is a list, organize by class field
            detections = {class_name: [] for class_name in ROBOFLOW_MINIMAP_VISION_CLASSES}
            if isinstance(predictions, list):
                for pred in predictions:
                    class_name = pred.get("class", "").lower()
                    if class_name in detections:
                        detections[class_name].append(pred)
        
        return detections
    except Exception as e:
        print(f"[MINIMAP_VISION] Detection error: {e}")
        return {}
