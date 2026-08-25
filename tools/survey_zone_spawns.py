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
import hud_ocr  # noqa: E402
from config import SCREEN_W, SCREEN_H, ZONE_NODES_WORLD  # noqa: E402
from detectors import grab_frame_bgr, is_death_screen  # noqa: E402
from input_control import (  # noqa: E402
    attack_key_down, attack_key_up, release_all_inputs, sleep_interruptible,
    wasd_move, wasd_move_and_attack,
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


# The game disables Vote Reset while enemies are near, which is the single
# biggest threat to a long survey: the bot spawns into a pack, cannot reset, and
# every remaining zone is skipped. Clearing the spawn area first is what makes an
# unattended full run viable -- a first smoke run lost 3 of 4 visits to exactly
# this. Sweeping the aim around while holding attack kills or scatters whatever
# is adjacent without needing enemy detection here.
CLEAR_SWEEP_DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def _grab_minimap(sct, mode="small"):
    region = get_minimap_region(mode)
    return cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)


def _clear_spawn_area(seconds):
    """Hold attack and sweep around, to unblock a Vote Reset. True unless aborted."""
    if seconds <= 0:
        return True
    per_dir = max(0.12, seconds / float(len(CLEAR_SWEEP_DIRS)))
    attack_key_down()
    try:
        for dx, dy in CLEAR_SWEEP_DIRS:
            # Small steps: the point is to face each direction and swing, not to
            # walk away from the spawn point the captures are relative to.
            wasd_move_and_attack(dx * 200, dy * 200, duration=per_dir,
                                 aim_dx=dx * 200, aim_dy=dy * 200)
            if not sleep_interruptible(0.05):
                return False
    finally:
        attack_key_up()
        release_all_inputs()
    return True


def _safe_name(zone):
    return re.sub(r"[^A-Za-z0-9]+", "_", zone).strip("_")


def _handle_death(sct, full_mon):
    """Respawn if the character is dead. Returns True if a death was handled.

    Without this the survey does not stop when the character dies -- it keeps
    calling the reset flow, which sees no pause menu and fires blind clicks at
    the death screen. A live run ended exactly that way, burning the rest of
    the zone list on a corpse.
    """
    try:
        frame = grab_frame_bgr(sct, full_mon)
    except Exception:
        return False
    if not is_death_screen(frame):
        return False
    print("[survey]   DEAD -- respawning")
    combat.attempt_respawn(sct, full_mon)
    sleep_interruptible(3.0)
    return True


def _verify_zone(sct, full_mon, expected):
    """Where did we actually land? Returns (matched, seen_name).

    `matched` is None when the HUD could not be read. This is not paranoia:
    run_vote_reset_sequence falls back to the satanic colour scan when a zone
    click fails, which SILENTLY changes the destination. Without this check a
    failed click files another zone's spawn under `expected`, and per-zone
    correlation — the entire output of this tool — is quietly wrong.
    """
    try:
        frame = grab_frame_bgr(sct, full_mon)
        name, _ratio, _raw = hud_ocr.read_current_zone(frame, list(ZONE_NODES_WORLD))
    except Exception as exc:
        print(f"[survey]   zone check failed ({type(exc).__name__}: {exc})")
        return None, None
    if name is None:
        return None, None
    return name == expected, name


def _travel_to(sct, full_mon, zone, clear_seconds, retries):
    """Vote-reset into `zone`. Returns (True | reason, aborted).

    A blocked reset is the expected failure, not an exceptional one: the game
    disables Vote Reset while enemies are near, so the fix is to clear them and
    ask again rather than to give up on the zone.
    """
    reason = "unknown"
    for attempt in range(retries + 1):
        # Check FIRST, not just between retries: the character may have died
        # during the previous zone's spawn-clearing sweep.
        if _handle_death(sct, full_mon):
            reason = "died"
        if attempt > 0:
            print(f"[survey]   blocked ({reason}) -- clearing spawn area, "
                  f"retry {attempt}/{retries}")
            if not _clear_spawn_area(clear_seconds):
                return "aborted", True
            if _handle_death(sct, full_mon):
                # Clearing killed us; skip this zone rather than blind-clicking
                # a death screen for the rest of the run.
                return "died while clearing", False
        try:
            result = combat.run_vote_reset_sequence(sct, full_mon, target_zone=zone)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}", False
        if result == "unavailable":
            reason = "enemies near / safe zone / cooldown"
            continue
        if result is False:
            reason = "reset failed"
            continue
        return True, False
    return reason, False


def survey(out_dir, zones, passes, minimap_mode, clear_seconds, reset_retries):
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
                result, aborted = _travel_to(sct, full_mon, zone, clear_seconds,
                                             reset_retries)
                if aborted:
                    return manifest, True
                if result is not True:
                    print(f"[survey]   could not reach {zone} ({result}) -- skipping")
                    continue

                if not sleep_interruptible(SETTLE_SECONDS):
                    return manifest, True

                matched, seen = _verify_zone(sct, full_mon, zone)
                if matched is False:
                    print(f"[survey]   LANDED IN '{seen}', not '{zone}' "
                          f"-- discarding (reset fell back to the satanic scan)")
                    continue
                if matched is None:
                    print("[survey]   zone unreadable -- capturing as UNVERIFIED")

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
                    "verified": bool(matched),
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

    # Records from before zone verification existed have no "verified" key.
    # Treat them as unverified rather than as trusted: they predate the check
    # that catches the reset falling back to the satanic scan, so any of them
    # could be another zone's spawn filed under this name.
    legacy = [r for r in manifest if "verified" not in r]
    unverified = [r for r in manifest if r.get("verified") is False]
    usable = [r for r in manifest if r.get("verified") is True]
    if legacy or unverified:
        print(f"excluded: {len(legacy)} pre-verification, "
              f"{len(unverified)} unverified (zone could not be read)")
    if not usable:
        print("no verified visits yet — re-run the survey to collect some")
        return

    by_zone = defaultdict(list)
    for rec in usable:
        by_zone[rec["zone"]].append(rec)

    print(f"{len(usable)} verified visits across {len(by_zone)} zones\n")
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
            print(f"{zone[:27]:<28}{len(recs):>7}  {label:>9}{med:>16.3f}")
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
    ap.add_argument("--clear-seconds", type=float, default=3.0,
                    help="attack sweep to clear enemies before retrying a blocked "
                         "reset (default: 3.0, 0 disables)")
    ap.add_argument("--reset-retries", type=int, default=2,
                    help="clear-and-retry attempts per zone (default: 2)")
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
        manifest, aborted = survey(args.out, zones, args.passes, args.minimap,
                                   args.clear_seconds, args.reset_retries)
    finally:
        release_all_inputs()
    print(f"[survey] {'aborted after' if aborted else 'done —'} "
          f"{len(manifest)} visits saved to {args.out}/")
    print(f"[survey] analyse with: python tools/survey_zone_spawns.py --analyse --out {args.out}")


if __name__ == "__main__":
    main()
