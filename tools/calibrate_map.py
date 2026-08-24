"""Capture the world map across its full width, for zone calibration.

The zone-select map is far wider than the screen: the old purple region sits
at the left, the new sepia/nautical region to its right, on one continuous
horizontal scroll axis. Wheel-up pans left, wheel-down pans right, at a
measured 30px per notch.

This tool walks that axis end to end and saves a frame at every step, plus a
manifest recording capture size and scroll distance. Those frames are the
source for the zone coordinate table; the hand-typed coordinates already in
config.py carry comments like "recalibrated from a live rotation-failure
frame", which is what guessing costs.

Both edges are DETECTED rather than assumed -- the map stops changing when it
is clamped -- so this keeps working when the next season widens the map again.

It NEVER clicks, so it cannot select a zone or change your game state.

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mss  # noqa: E402
from config import SCREEN_W, SCREEN_H  # noqa: E402
from detectors import grab_frame_bgr, is_world_map_visible  # noqa: E402
from input_control import (  # noqa: E402
    mouse_scroll, move_cursor_to, sleep_interruptible,
)
from map_scroll import PX_PER_NOTCH, scroll_to_edge  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="calibration",
                    help="directory for captured frames (default: calibration/)")
    ap.add_argument("--step", type=int, default=4,
                    help="notches between captures (4 = 120px, ~94%% overlap)")
    ap.add_argument("--max-steps", type=int, default=250,
                    help="safety cap on captures while walking right")
    ap.add_argument("--normalize-chunk", type=int, default=8,
                    help="notches per iteration while seeking the left edge")
    ap.add_argument("--normalize-max", type=int, default=80,
                    help="cap on left-edge seek iterations (80x8 = 19200px)")
    ap.add_argument("--settle", type=float, default=0.35,
                    help="seconds to wait after each scroll before capturing")
    ap.add_argument("--countdown", type=int, default=5,
                    help="seconds to alt-tab into the game before capture")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # Clear frames from an earlier run: a partial old capture mixed with a new
    # one silently yields a coordinate set stitched from two scroll origins.
    stale = [f for f in os.listdir(args.out) if f.startswith("map_stop_")]
    for f in stale:
        os.remove(os.path.join(args.out, f))
    if stale:
        print(f"[cal] cleared {len(stale)} frames from a previous run")

    print(f"[cal] screen {SCREEN_W}x{SCREEN_H}  ({PX_PER_NOTCH}px per notch)")
    print(f"[cal] focus the GAME window now — starting in {args.countdown}s "
          f"(F1 or P aborts)")
    for i in range(args.countdown, 0, -1):
        print(f"      {i}...")
        if not sleep_interruptible(1.0):
            print("[cal] aborted before capture")
            return 1

    mon = {"top": 0, "left": 0, "width": SCREEN_W, "height": SCREEN_H}
    with mss.mss() as sct:
        def grab():
            return grab_frame_bgr(sct, mon)

        def settle():
            sleep_interruptible(args.settle)

        alive = {"ok": True}

        def should_continue():
            return alive["ok"]

        if not is_world_map_visible(grab()):
            # Only the purple region is detectable today, and a normalized map
            # starts there -- so a miss here usually means the map is not open.
            print("[cal] WARNING: no world map detected. Open the zone map "
                  "(stand on a waypoint, press F) and re-run.")
            print("[cal] continuing anyway in case the detector missed it.")

        move_cursor_to(SCREEN_W // 2, SCREEN_H // 2)

        # ── seek the left edge ────────────────────────────────────────────
        print("[cal] seeking left edge...")
        _, reason = scroll_to_edge(
            grab, mouse_scroll, step=args.normalize_chunk,
            max_steps=args.normalize_max, settle=settle)
        if reason != "edge":
            print(f"[cal] ABORT: left edge not reached ({reason}). Every "
                  f"coordinate is relative to it, so capturing now would "
                  f"produce an unusable set. Raise --normalize-max.")
            return 1
        print("[cal] left edge confirmed")

        # ── walk right, saving every frame ────────────────────────────────
        frames_meta = []

        def on_capture(i, frame):
            name = f"map_stop_{i:03d}.png"
            cv2.imwrite(os.path.join(args.out, name), frame)
            frames_meta.append({
                "file": name,
                "index": i,
                "notches_right": i * args.step,
                "pixels_right": i * args.step * PX_PER_NOTCH,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            })
            if i % 10 == 0:
                print(f"[cal] {name}  (+{i * args.step * PX_PER_NOTCH}px)")

        print("[cal] walking right...")
        frames, reason = scroll_to_edge(
            grab, mouse_scroll, step=-args.step, max_steps=args.max_steps,
            on_capture=on_capture, should_continue=should_continue,
            settle=settle)

    if reason == "edge":
        span = (len(frames) - 1) * args.step * PX_PER_NOTCH
        print(f"[cal] right edge reached: {len(frames)} frames, "
              f"map spans ~{span}px beyond the first screen")
    elif reason == "max_steps":
        print(f"[cal] WARNING: hit --max-steps ({args.max_steps}) before the "
              f"right edge. Re-run with a higher cap; the capture is partial.")
    else:
        print("[cal] aborted by stop key; capture is partial")

    manifest = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "screen_w": SCREEN_W,
        "screen_h": SCREEN_H,
        "px_per_notch": PX_PER_NOTCH,
        "step_notches": args.step,
        "reason": reason,
        "reached_right_edge": reason == "edge",
        "frames": frames_meta,
    }
    with open(os.path.join(args.out, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[cal] saved {len(frames_meta)} frames + manifest to {args.out}/")
    print("[cal] the map is still open — press ESC in game to close it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
