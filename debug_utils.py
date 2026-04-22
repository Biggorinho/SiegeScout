import cv2
import numpy as np
import os
import time

from config import DEBUG_DIR


def save_hard_stuck_snapshot(frame_bgr, minimap_bgr, nearest, center, hard_stuck_age):
    """Write one full-frame + minimap snapshot when hard-stuck is reached."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    stamp = int(time.time() * 1000)

    if frame_bgr is not None and frame_bgr.size != 0:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"hard_stuck_full_{stamp}.png"), frame_bgr)

    if minimap_bgr is not None and minimap_bgr.size != 0:
        mini = minimap_bgr.copy()
        if center is not None:
            cv2.drawMarker(mini, (int(center[0]), int(center[1])), (255, 120, 0), cv2.MARKER_CROSS, 12, 2)
        if nearest is not None:
            cv2.circle(mini, (int(nearest[0]), int(nearest[1])), 8, (0, 0, 255), 2)
            cv2.putText(mini, "TARGET", (int(nearest[0]) + 4, int(nearest[1]) - 6),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), 1)
        cv2.putText(mini, f"HARD_STUCK {hard_stuck_age:.1f}s", (8, 18),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 220, 255), 1)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"hard_stuck_minimap_{stamp}.png"), mini)


def save_minimap_frame(
    minimap_bgr, dots_raw, companion_dot, filtered_dots, nearest, center, frame_idx,
    state=None, walkable_mask=None, explore_target=None,
):
    """Save 3-panel annotated minimap debug frame.

    Panels (left to right):
      1. Annotated minimap  — player cross, raw dots, companion, filtered dots, target, explore target
      2. Enemy colour mask  — what the dot detector sees (red/purple/yellow pixels)
      3. Walkable mask      — what the pathfinder sees (white=walkable) with the same overlays
    """
    os.makedirs(DEBUG_DIR, exist_ok=True)
    img    = minimap_bgr.copy()
    cx, cy = center

    # ── panel 1: annotated minimap ─────────────────────────────────────────────
    # raw candidate dots (yellow outline)
    for d in dots_raw:
        cv2.circle(img, d, 5, (255, 255, 0), 1)

    # companion (magenta)
    if companion_dot is not None:
        cv2.circle(img, companion_dot, 8, (255, 0, 255), 2)
        cv2.putText(img, "COMP", (companion_dot[0] + 4, companion_dot[1] - 4),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 255), 1)

    # confirmed enemy dots (green)
    for d in filtered_dots:
        cv2.circle(img, d, 6, (0, 255, 80), 2)

    # nearest enemy (red)
    if nearest:
        cv2.circle(img, nearest, 9, (0, 0, 255), 2)
        cv2.putText(img, "ENEMY", (nearest[0] + 4, nearest[1] - 6),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), 1)

    # explore / path target (cyan)
    if explore_target is not None:
        tx, ty = int(explore_target[0]), int(explore_target[1])
        cv2.drawMarker(img, (tx, ty), (255, 220, 0), cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.line(img, (cx, cy), (tx, ty), (255, 220, 0), 1)

    # player center (orange cross)
    cv2.drawMarker(img, (cx, cy), (255, 100, 0), cv2.MARKER_CROSS, 14, 2)

    # state label
    label = str(state) if state is not None else ""
    if label:
        cv2.putText(img, label, (4, img.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 220, 255), 1)

    # ── panel 2: enemy colour mask ─────────────────────────────────────────────
    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    r1  = cv2.inRange(hsv, np.array([0,   120, 110]), np.array([10,  255, 255]))
    r2  = cv2.inRange(hsv, np.array([170, 120, 110]), np.array([180, 255, 255]))
    p   = cv2.inRange(hsv, np.array([128,  80,  90]), np.array([160, 255, 255]))
    y   = cv2.inRange(hsv, np.array([10,   90, 100]), np.array([38,  255, 255]))
    color_mask = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(r1, r2), p), y)
    panel2 = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)

    # ── panel 3: walkable mask ─────────────────────────────────────────────────
    if walkable_mask is not None and walkable_mask.size != 0:
        panel3 = cv2.cvtColor(walkable_mask, cv2.COLOR_GRAY2BGR)
    else:
        panel3 = np.zeros_like(img)
        cv2.putText(panel3, "no mask", (4, 14), cv2.FONT_HERSHEY_PLAIN, 0.9, (80, 80, 80), 1)

    # draw overlays on walkable panel too so it's easy to cross-reference
    cv2.drawMarker(panel3, (cx, cy), (255, 100, 0), cv2.MARKER_CROSS, 14, 2)
    if nearest:
        cv2.circle(panel3, nearest, 9, (0, 0, 255), 2)
    if explore_target is not None:
        tx, ty = int(explore_target[0]), int(explore_target[1])
        cv2.drawMarker(panel3, (tx, ty), (255, 220, 0), cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.line(panel3, (cx, cy), (tx, ty), (255, 220, 0), 1)

    combined = np.hstack([img, panel2, panel3])
    cv2.imwrite(os.path.join(DEBUG_DIR, f"dbg_{frame_idx:04d}.png"), combined)


def save_skull_detection_snapshot(minimap_bgr, enemy_dots, center, stamp=None, state_label=None, zone_age_s=None):
    """Write a focused minimap snapshot whenever skull dots are detected.

    Panels (left to right):
      1. Minimap with skull markers and center
      2. Skull HSV mask used by detector
      3. Generic enemy color mask for cross-check
    """
    if minimap_bgr is None or minimap_bgr.size == 0:
        return

    os.makedirs(DEBUG_DIR, exist_ok=True)
    if stamp is None:
        stamp = int(time.time() * 1000)

    label = str(state_label).lower() if state_label is not None else "unknown"
    label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label) or "unknown"

    skull_dots = [(x, y) for x, y, _, kind in enemy_dots if kind == "skull"]
    if not skull_dots:
        return

    panel1 = minimap_bgr.copy()
    if center is not None:
        cv2.drawMarker(panel1, (int(center[0]), int(center[1])), (255, 100, 0), cv2.MARKER_CROSS, 14, 2)
    for x, y in skull_dots:
        cv2.circle(panel1, (int(x), int(y)), 8, (255, 255, 255), 2)
        cv2.putText(panel1, "SKULL", (int(x) + 4, int(y) - 6),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)

    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    skull_mask = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([180, 75, 255]))
    panel2 = cv2.cvtColor(skull_mask, cv2.COLOR_GRAY2BGR)

    r1 = cv2.inRange(hsv, np.array([0, 120, 110]), np.array([10, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([170, 120, 110]), np.array([180, 255, 255]))
    p = cv2.inRange(hsv, np.array([128, 80, 90]), np.array([160, 255, 255]))
    y = cv2.inRange(hsv, np.array([10, 90, 100]), np.array([38, 255, 255]))
    enemy_mask = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(r1, r2), p), y)
    panel3 = cv2.cvtColor(enemy_mask, cv2.COLOR_GRAY2BGR)

    info = f"skulls={len(skull_dots)} state={label}"
    if zone_age_s is not None:
        info += f" zone={int(zone_age_s)}s"
    cv2.putText(panel1, info, (5, 14), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 220, 255), 1)
    cv2.putText(panel2, "skull_mask", (5, 14), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 220, 255), 1)
    cv2.putText(panel3, "enemy_mask", (5, 14), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 220, 255), 1)

    combined = np.hstack([panel1, panel2, panel3])
    cv2.imwrite(os.path.join(DEBUG_DIR, f"skull_dbg_{label}_z{int(zone_age_s) if zone_age_s is not None else 0}_{stamp}.png"), combined)


def save_skull_mask_probe(minimap_bgr, center=None, stamp=None, state_label=None, zone_age_s=None):
    """Save skull mask probe even when no skull was confirmed, for miss diagnosis."""
    if minimap_bgr is None or minimap_bgr.size == 0:
        return

    os.makedirs(DEBUG_DIR, exist_ok=True)
    if stamp is None:
        stamp = int(time.time() * 1000)

    label = str(state_label).lower() if state_label is not None else "unknown"
    label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label) or "unknown"

    panel1 = minimap_bgr.copy()
    if center is not None:
        cv2.drawMarker(panel1, (int(center[0]), int(center[1])), (255, 100, 0), cv2.MARKER_CROSS, 14, 2)

    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
    skull_mask = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([180, 75, 255]))
    panel2 = cv2.cvtColor(skull_mask, cv2.COLOR_GRAY2BGR)

    cv2.putText(panel1, f"state={label} zone={int(zone_age_s) if zone_age_s is not None else 0}s", (5, 14),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 220, 255), 1)
    cv2.putText(panel2, "skull_mask_probe", (5, 14), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 220, 255), 1)

    combined = np.hstack([panel1, panel2])
    cv2.imwrite(os.path.join(DEBUG_DIR, f"skull_probe_{label}_z{int(zone_age_s) if zone_age_s is not None else 0}_{stamp}.png"), combined)
