"""Visit every zone repeatedly and capture its spawn area, for layout analysis.

The question this answers: how much of a zone's SPAWN area is the same every
time you enter it? Whole-map wall structure across resets correlates only
0.87-0.94 -- similar, never identical -- so a persisted blocked-map applied as
hard walls would route the bot around walls that are not there and into walls it
thinks are clear. But a near-spawn comparison on 2026-08-25 hit 0.999 on one
pair, i.e. pixel-identical, which is what makes a SPAWN-AREA prior worth
measuring properly.

Existing debug snapshots cannot settle it: they fire when combat starts, which
is anywhere from instantly to several seconds after entry, by which point the
player has walked off. Frames captured at a fixed offset from entry, tagged with
the zone, are directly comparable. That is all this tool does.

Each visit captures the minimap at a fixed settle time, then walks a short
scripted box to lift fog of war around the spawn and captures again after each
leg. Run several passes so the same zone is seen repeatedly -- comparing passes
is the whole point.

Analyse the result with:  python tools/survey_zone_spawns.py --analyse

Usage (dev-only, from the project root, with the game running):

    1. Be in-game, standing somewhere a vote reset is allowed (NOT a safe zone).
    2. python tools/survey_zone_spawns.py --passes 3
    3. Alt-tab back to the game before the countdown ends.

It drives vote resets and walks a few steps; it does not touch your inventory,
skills or menus beyond the normal reset flow. Abort at any time with F1 or P.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mss  # noqa: E402
import combat  # noqa: E402
from config import SCREEN_W, SCREEN_H, ZONE_NODES_WORLD  # noqa: E402
from input_control import (  # noqa: E402
    release_all_inputs, sleep_interruptible, wasd_move,
)
from minimap import get_minimap_region  # noqa: E402

# Fixed offset from zone entry for the first capture. Long enough for the zone
# to finish loading and the minimap to populate, short enough that the bot has
# not wandered off the spawn point -- which is exactly what made the existing
# enter_attack snapshots unusable for this.
SETTLE_SECONDS = 1.5

# Short scripted legs to lift fog of war around spawn, as (dx, dy, seconds).
# Deliberately a there-and-back box so the player ends near where it started
# and the next capture is still spawn-relative.
SCAN_LEGS = [
    (0, -260, 0.55), (0, 260, 0.55),
    (260, 0, 0.55), (-260, 0, 0.55),
    (0, 260, 0.55), (0, -260, 0.55),
    (-260, 0, 0.55), (260, 0, 0.55),
]


def _grab_minimap(sct, mode="small"):
    region = get_minimap_region(mode)
    return cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)


def _safe_name(zone):
    return re.sub(r"[^A-Za-z0-9]+", "_", zone).strip("_")


def survey(out_dir, zones, passes, minimap_mode):
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        try:
            manifest = json.load(open(manifest_path))
            print(f"[survey] appending to {len(manifest)} existing records")
        except Exception:
            manifest = []

    with mss.mss() as sct:
        full_mon = {"left": 0, "top": 0, "width": SCREEN_W, "height": SCREEN_H}
        for p in range(passes):
            for zone in zones:
                print(f"[survey] pass {p + 1}/{passes}  ->  {zone}")
                try:
                    result = combat.run_vote_reset_sequence(
                        sct, full_mon, target_zone=zone)
                except Exception as exc:
                    print(f"[survey]   reset raised {type(exc).__name__}: {exc} -- skipping")
                    continue
                if result == "unavailable":
                    print("[survey]   vote reset unavailable here -- skipping")
                    continue
                if result is False:
                    print("[survey]   reset failed -- skipping")
                    continue

                if not sleep_interruptible(SETTLE_SECONDS):
                    return manifest, True
                stamp = int(time.time() * 1000)
                zdir = os.path.join(out_dir, _safe_name(zone))
                os.makedirs(zdir, exist_ok=True)

                shots = []
                for leg_i in range(len(SCAN_LEGS) + 1):
                    name = f"{stamp}_leg{leg_i:02d}.png"
                    cv2.imwrite(os.path.join(zdir, name), _grab_minimap(sct, minimap_mode))
                    shots.append(name)
                    if leg_i == len(SCAN_LEGS):
                        break
                    dx, dy, secs = SCAN_LEGS[leg_i]
                    wasd_move(dx, dy, secs)
                    if not sleep_interruptible(0.25):
                        release_all_inputs()
                        return manifest, True

                release_all_inputs()
                manifest.append({
                    "zone": zone, "pass": p, "stamp": stamp,
                    "dir": _safe_name(zone), "shots": shots,
                    "settle_seconds": SETTLE_SECONDS,
                })
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)
                print(f"[survey]   captured {len(shots)} frames")
    return manifest, False


# ── analysis ────────────────────────────────────────────────────────────────

def _wall_sig(path, frac):
    img = cv2.imread(path)
    if img is None or img.size == 0:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    wall = cv2.inRange(hsv, np.array([143, 210, 60]), np.array([162, 255, 255]))
    h, w = wall.shape
    r = int(min(h, w) * frac)
    crop = wall[h // 2 - r:h // 2 + r, w // 2 - r:w // 2 + r]
    if crop.size == 0 or crop.sum() // 255 < 40:
        return None
    s = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (s - s.mean()) / (s.std() + 1e-6)


def analyse(out_dir):
    manifest_path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"no manifest at {manifest_path} -- run a survey first")
        return
    manifest = json.load(open(manifest_path))
    by_zone = defaultdict(list)
    for rec in manifest:
        by_zone[rec["zone"]].append(rec)

    print(f"{len(manifest)} visits across {len(by_zone)} zones\n")
    print(f"{'zone':<28}{'visits':>7}{'radius':>9}{'same-zone corr':>18}")
    print("-" * 62)
    summary = []
    for zone, recs in sorted(by_zone.items()):
        if len(recs) < 2:
            print(f"{zone[:27]:<28}{len(recs):>7}{'':>9}{'(need 2+ visits)':>18}")
            continue
        for label, frac in (("spawn .25", 0.25), ("wide .49", 0.49)):
            sigs = []
            for rec in recs:
                # leg00 = the fixed-offset capture, before any scripted walking
                p = os.path.join(out_dir, rec["dir"], rec["shots"][0])
                s = _wall_sig(p, frac)
                if s is not None:
                    sigs.append(s)
            if len(sigs) < 2:
                continue
            cors = [float((sigs[i] * sigs[j]).mean())
                    for i in range(len(sigs)) for j in range(i + 1, len(sigs))]
            med = float(np.median(cors))
            print(f"{zone[:27]:<28}{len(recs):>7}{label:>9}{med:>18.3f}")
            summary.append((zone, label, med))

    if summary:
        for label in ("spawn .25", "wide .49"):
            vals = [m for _, l, m in summary if l == label]
            if vals:
                print(f"\nmedian across zones, {label}: {np.median(vals):.3f}")
        print("\nReading it: high at spawn .25 and lower at wide .49 means the")
        print("spawn area repeats while the wider map does not -- a spawn-area")
        print("prior is worth building, bounded by roughly that radius. Similar")
        print("at both, or low at both, means persistence is not worth it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="spawn_survey", help="output dir (default: spawn_survey/)")
    ap.add_argument("--passes", type=int, default=3, help="visits per zone (default: 3)")
    ap.add_argument("--zones", default="", help="comma-separated subset (default: all 46)")
    ap.add_argument("--minimap", default="small", choices=("small", "large"))
    ap.add_argument("--countdown", type=int, default=6, help="seconds to alt-tab back")
    ap.add_argument("--analyse", action="store_true", help="analyse an existing survey and exit")
    args = ap.parse_args()

    if args.analyse:
        analyse(args.out)
        return

    zones = [z.strip() for z in args.zones.split(",") if z.strip()] or list(ZONE_NODES_WORLD)
    unknown = [z for z in zones if z not in ZONE_NODES_WORLD]
    if unknown:
        print(f"unknown zone(s): {unknown}")
        print(f"known: {sorted(ZONE_NODES_WORLD)}")
        return

    est_min = len(zones) * args.passes * 35 / 60.0
    print(f"[survey] {len(zones)} zones x {args.passes} passes "
          f"~= {est_min:.0f} min. Abort any time with F1 or P.")
    for i in range(args.countdown, 0, -1):
        print(f"  starting in {i}...")
        if not sleep_interruptible(1.0):
            print("[survey] aborted")
            return

    aborted = False
    try:
        manifest, aborted = survey(args.out, zones, args.passes, args.minimap)
    finally:
        release_all_inputs()
    print(f"[survey] {'aborted after' if aborted else 'done —'} "
          f"{len(manifest)} visits saved to {args.out}/")
    print(f"[survey] analyse with: python tools/survey_zone_spawns.py --analyse --out {args.out}")


if __name__ == "__main__":
    main()
