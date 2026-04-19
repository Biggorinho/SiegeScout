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
    skull = cv2.inRange(hsv, np.array([0, 0, 175]),     np.array([180, 45,  255]))

    h, w = minimap_bgr.shape[:2]
    cx0  = w // 2
    cy0  = h // 2
    exclusion_radius = max(6, min(PLAYER_EXCLUSION_RADIUS, int(min(w, h) * PLAYER_EXCLUSION_RADIUS_FRAC)))
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

    def _append_from_mask(mask, tier, kind, area_min, area_max, circularity_min):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (area_min < area < area_max):
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter ** 2)
            if circularity < circularity_min:
                continue
            M = cv2.moments(cnt)
            if M["m00"] <= 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if _looks_like_white_companion(cnt, ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5 <= white_suppress_radius):
                continue
            _append_enemy_dot(dots, cx, cy, tier, kind, cx0, cy0, exclusion_radius)

    _append_from_mask(skull, ENEMY_TIER_SKULL, "skull", 10, 360, 0.18)
    _append_from_mask(y, ENEMY_TIER_YELLOW, "yellow", 3, 320, 0.55)
    _append_from_mask(p, ENEMY_TIER_PURPLE, "purple", 3, 320, 0.55)
    _append_from_mask(cv2.bitwise_or(r1, r2), ENEMY_TIER_RED, "red", 3, 320, 0.55)

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
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    best_pt, best_score = None, -1
    for tpl in templates:
        if tpl is None:
            continue
        w, h = tpl.shape[::-1]
        res  = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score >= ITEM_MATCH_THRESHOLD and score > best_score:
            best_score = score
            best_pt    = (loc[0] + w // 2, loc[1] + h // 2)
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
