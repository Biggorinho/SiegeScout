"""What the pause-menu detector actually gets right and wrong.

This detector gates every vote reset -- `combat.wait_for_world_map` checks it
before anything else and taps ESC when it fires -- so both of its failure
directions are expensive. A miss costs ~19s of blind clicking; a false
positive presses ESC during gameplay, which OPENS the menu it thought was
already open.

Both directions are real, and both are measured here. The xfail tests are
not aspirational: they are a scoreboard for the next attempt at this
detector, written after one attempt failed (see below). They are `strict`,
so if someone fixes the detector these tests fail loudly and get promoted
rather than sitting green and forgotten.

Diagnosis, 2026-08-27, measured on the frames below:

* The mask is a red hue gate, `H<=20 or H>=160` with `S>=45, V>=38`. In a
  sepia zone the BACKGROUND sits at H=10.8, S=201, V=95 -- inside that gate.
  So the mask floods, the buttons stop being separable contours, and the
  count drops to zero. Loosening S/V makes it strictly worse, which is the
  giveaway: this is not a darkness problem, it is a hue-collision problem.
  The buttons are separable by VALUE (V=32-62 against a V=91-101 ground),
  not by hue.
* Capping V at 80 recovers 2 of the 4 captured sepia misses, and never
  yields more bars than the shipping mask on any fixture -- but 2 of 4 is
  not a fix.
* Requiring the bars to form a vertical stack (aligned centres, similar
  widths, actually separated) does NOT separate menus from gameplay:
  `hud_forgotten_caves` scores 6 stacked bars, more than the real
  `purple_town_pause_menu` at 2. That idea is dead; do not retry it.
"""

import os
import sys

import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detectors import is_pause_menu_visible

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _frame(name):
    path = os.path.join(_FIXTURES, name)
    frame = cv2.imread(path)
    assert frame is not None, f"missing fixture {path}"
    return frame


@pytest.mark.parametrize("name", [
    "pause_menu.png",
    "pause_menu_combat.png",
    "purple_town_pause_menu.png",
    "votereset_active.png",
    "votereset_disabled.png",
])
def test_real_pause_menus_are_detected(name):
    """The cases that work today. Any change must keep these green."""
    assert is_pause_menu_visible(_frame(name)) is True


@pytest.mark.parametrize("name", [
    "map_selection.png",
    "safe_zone.png",
    "purple_zone_combat.png",
    "map_sepia_right.png",
    "mining_prompt.png",
    "wilderness_ping_only.png",
    "worldmap_satanic_forest.png",
])
def test_non_menu_screens_are_not_detected(name):
    """The negatives that already hold. Any change must keep these green."""
    assert is_pause_menu_visible(_frame(name)) is False


@pytest.mark.xfail(strict=True, reason=(
    "Known false positive: ordinary gameplay in these zones reads as a pause "
    "menu. Nothing covered it before 2026-08-27. Costly because "
    "wait_for_world_map taps ESC on a positive, which opens the real menu -- "
    "a plausible source of the ESC-wedge this project keeps hitting."))
@pytest.mark.parametrize("name", [
    "hud_forgotten_caves.png",
    "hud_prison_dungeon.png",
    "hud_glacial_trail_half.png",
])
def test_gameplay_frames_are_not_mistaken_for_the_menu(name):
    assert is_pause_menu_visible(_frame(name)) is False


@pytest.mark.xfail(strict=True, reason=(
    "Known miss: in a sepia zone the background hue lands inside the red "
    "gate, the mask floods, and the buttons stop being separable contours. "
    "Captured live 2026-08-27 in Dry Hills -- the menu is plainly open "
    "(RESUME / VOTE RESET / OPTIONS / SAVE & EXIT) and the bot fell through "
    "to ~19s of blind clicking, four times in one run."))
def test_a_pause_menu_in_a_sepia_zone_is_detected():
    assert is_pause_menu_visible(_frame("pause_menu_sepia_dark.png")) is True
