"""Capture the world map at every horizontal scroll position, for calibration.

The zone-select map is wider than the screen: the old purple region sits at the
left, the new sepia/nautical region to its right. Wheel-up pans left, wheel-down
pans right, and the scroll position survives closing the map -- so the bot
normalizes hard-left before using any calibrated coordinate.

This tool walks that axis and saves a full-resolution frame at each notch, plus
a manifest recording the capture size and how far each frame is scrolled. Those
frames are the source for the zone coordinate table; hand-typed coordinates are
what produced the "recalibrated from a live rotation-failure frame" comments
already in config.py.

It NEVER clicks -- it only scrolls and screenshots, so it cannot select a zone
or change your game state.

Usage (dev-only, from the project root, with the game running):

    1. Stand on a waypoint and open the zone map yourself (press F).
    2. python tools/calibrate_map.py
    3. Alt-tab back to the game before the countdown ends.

Abort at any time with F1 or P (the project-wide stop keys).
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mss  # noqa: E402
from config import SCREEN_W, SCREEN_H  # noqa: E402
from detectors import grab_frame_bgr, is_world_map_visible  # noqa: E402
from input_control import (  # noqa: E402
    mouse_scroll, move_cursor_to, sleep_interruptible,
)

# Consecutive near-identical frames that mean "the map stopped panning".
_STABLE_REPEATS = 3
# Mean absolute per-pixel difference below which two frames count as identical.
# Measured margins are wide: a 4px pan already scores ~13.7, while a static
# frame with +/-1 of capture noise scores ~0.67. 3.0 sits between them, so
# animated map art (torch flicker, shimmer) cannot masquerade as panning and a
# real pan can never be mistaken for the edge.
_STABLE_MAD = 3.0


def _mad(a, b):
    """Mean absolute difference between two frames, or inf if incomparable."""
    if a is None or b is None or a.shape != b.shape:
        return float("inf")
    return float(np.mean(cv2.absdiff(a, b)))


def capture_scroll_axis(sct, mon, out_dir, max_notches, step, settle,
                        normalize_notches):
    os.makedirs(out_dir, exist_ok=True)

    print(f"[cal] normalizing: scrolling up {normalize_notches} notches (hard left)")
    move_cursor_to(SCREEN_W // 2, SCREEN_H // 2)
    mouse_scroll(normalize_notches)
    if not sleep_interruptible(max(settle, 0.5)):
        return None

    frames = []
    prev = None
    stable = 0
    notch = 0

    while notch <= max_notches:
        frame = grab_frame_bgr(sct, mon)
        diff = _mad(prev, frame)
        name = f"map_stop_{notch:02d}.png"
        cv2.imwrite(os.path.join(out_dir, name), frame)
        frames.append({
            "file": name,
            "notches_right": notch,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "diff_from_prev": None if prev is None else round(diff, 3),
        })
        print(f"[cal] {name}  {frame.shape[1]}x{frame.shape[0]}  "
              f"diff={'-' if prev is None else f'{diff:.2f}'}")

        # The right edge announces itself: further scrolling stops changing the
        # image. Require several in a row so one dropped frame can't end it.
        if diff <= _STABLE_MAD:
            stable += 1
            if stable >= _STABLE_REPEATS:
                print(f"[cal] right edge reached after {notch} notches "
                      f"({stable} identical frames)")
                break
        else:
            stable = 0

        prev = frame
        notch += step
        mouse_scroll(-step)          # negative = wheel down = pan right
        if not sleep_interruptible(settle):
            print("[cal] aborted by stop key")
            return frames
    else:
        print(f"[cal] hit --max-notches ({max_notches}) before the map stopped "
              f"panning; re-run with a higher limit if the last frames differ")

    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="calibration",
                    help="directory for captured frames (default: calibration/)")
    ap.add_argument("--max-notches", type=int, default=40,
                    help="safety cap on how far right to scroll")
    ap.add_argument("--step", type=int, default=1,
                    help="notches between captures (default 1 = finest)")
    ap.add_argument("--settle", type=float, default=0.35,
                    help="seconds to wait after each scroll before capturing")
    ap.add_argument("--normalize-notches", type=int, default=18,
                    help="wheel-up notches used to reach the hard-left edge")
    ap.add_argument("--countdown", type=int, default=5,
                    help="seconds to alt-tab into the game before capture")
    args = ap.parse_args()

    print(f"[cal] screen {SCREEN_W}x{SCREEN_H}")
    print(f"[cal] focus the GAME window now — starting in {args.countdown}s "
          f"(F1 or P aborts)")
    for i in range(args.countdown, 0, -1):
        print(f"      {i}...")
        if not sleep_interruptible(1.0):
            print("[cal] aborted before capture")
            return 1

    mon = {"top": 0, "left": 0, "width": SCREEN_W, "height": SCREEN_H}
    with mss.mss() as sct:
        first = grab_frame_bgr(sct, mon)
        if not is_world_map_visible(first):
            # Only the purple region is detectable today, and that is exactly
            # where a normalized map starts -- so a miss here almost always
            # means the map simply is not open.
            print("[cal] WARNING: no world map detected. Open the zone map "
                  "(stand on a waypoint, press F) and re-run.")
            print("[cal] continuing anyway in case the detector missed it.")

        frames = capture_scroll_axis(
            sct, mon, args.out, args.max_notches, args.step, args.settle,
            args.normalize_notches)

    if not frames:
        print("[cal] nothing captured")
        return 1

    manifest = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "screen_w": SCREEN_W,
        "screen_h": SCREEN_H,
        "step_notches": args.step,
        "normalize_notches": args.normalize_notches,
        "settle_seconds": args.settle,
        "frames": frames,
    }
    path = os.path.join(args.out, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[cal] saved {len(frames)} frames + manifest to {args.out}/")
    print("[cal] the map is still open — press ESC in game to close it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
