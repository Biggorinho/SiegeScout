"""Reading the satanic zone off a real HUD.

The HUD prints four lines top-right; only the satanic one is magenta:

    Dawn's Chapel        cream  - the zone you are in
    (skull)Prison Dungeon  MAGENTA - the satanic zone
    Hell (2)             cream  - difficulty
    Safe Zone            green

Picking the magenta line is what stops us reading the zone we are standing in
as the satanic target. Fixtures are unmodified frames the bot itself dumped.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hud_ocr
import stats_tracker as st

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
ZONES = [z for act in st._ZONE_NAMES.values() for z in act]

needs_ocr = pytest.mark.skipif(
    not hud_ocr.ocr_available(), reason="rapidocr-onnxruntime not installed")


def _fixture(name):
    img = cv2.imread(os.path.join(FIX, name))
    assert img is not None, name
    return img


def test_hud_strip_is_the_top_right_corner():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    strip = hud_ocr.hud_strip(frame)
    h, w = strip.shape[:2]
    assert 0 < h < 1080 * 0.30, "strip should be a shallow band"
    assert 0 < w < 1920 * 0.60, "strip should not span the whole width"


def test_hud_strip_scales_with_resolution():
    small = hud_ocr.hud_strip(np.zeros((540, 960, 3), dtype=np.uint8))
    large = hud_ocr.hud_strip(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert large.shape[0] == pytest.approx(small.shape[0] * 2, abs=2)
    assert large.shape[1] == pytest.approx(small.shape[1] * 2, abs=2)


@needs_ocr
def test_reads_prison_dungeon_from_four_line_hud():
    name, ratio, raw = hud_ocr.read_satanic_zone(
        _fixture("hud_prison_dungeon.png"), ZONES)
    assert name == "Prison Dungeon", f"raw={raw!r}"


@needs_ocr
def test_reads_forgotten_caves_an_act_8_zone():
    name, ratio, raw = hud_ocr.read_satanic_zone(
        _fixture("hud_forgotten_caves.png"), ZONES)
    assert name == "Forgotten Caves", f"raw={raw!r}"


@needs_ocr
def test_reads_at_half_resolution():
    # Same pipeline must work whatever the capture size is.
    name, ratio, raw = hud_ocr.read_satanic_zone(
        _fixture("hud_glacial_trail_half.png"), ZONES)
    assert name == "The Glacial Trail", f"raw={raw!r}"


@needs_ocr
def test_does_not_return_the_zone_the_player_is_standing_in():
    # hud_prison_dungeon.png shows "Dawn's Chapel" as the current zone in
    # cream directly above the magenta satanic line.
    name, _, _ = hud_ocr.read_satanic_zone(
        _fixture("hud_prison_dungeon.png"), ZONES + ["Dawn's Chapel"])
    assert name != "Dawn's Chapel"


@needs_ocr
def test_reads_the_cream_satanic_line_when_standing_in_that_zone():
    """The satanic line is NOT always magenta.

    Captured live 2026-08-24: after a reset lands the player INSIDE the
    satanic zone, the HUD shows it as the ordinary current-zone line -- cream,
    hue 20-29 -- prefixed with the word "Satanic":

        Satanic The Highland Mines
        Nightmare
        Zone Level 210

    Requiring magenta rejected every line here, so the reader returned None on
    every reset of that session and the bot fell back to the colour scan. The
    word "Satanic" is the signal colour cannot provide."""
    name, ratio, raw = hud_ocr.read_satanic_zone(
        _fixture("hud_satanic_cream_in_zone.png"), ZONES)
    assert name == "The Highland Mines", f"raw={raw!r}"


def test_frame_without_magenta_returns_none():
    black = np.zeros((1080, 1920, 3), dtype=np.uint8)
    name, ratio, _ = hud_ocr.read_satanic_zone(black, ZONES)
    assert name is None
    assert ratio == 0.0


def test_injected_ocr_is_used():
    """A caller-supplied engine keeps the module testable without rapidocr."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Paint a magenta bar inside the HUD strip so the colour test passes.
    # The strip stops short of the minimap, so its origin has to come from
    # the module rather than being assumed flush with the frame edge.
    y0, _, x0, _ = hud_ocr.hud_strip_bounds(frame)
    frame[y0 + 5:y0 + 20, x0 + 10:x0 + 200] = (200, 0, 200)   # BGR magenta

    def fake_ocr(img):
        box = [[10, 5], [200, 5], [200, 20], [10, 20]]
        return ([[box, "Steam Train", 0.91]], 0.01)

    name, ratio, raw = hud_ocr.read_satanic_zone(frame, ZONES, ocr=fake_ocr)
    assert name == "Steam Train"
    assert raw == "Steam Train"


def test_none_frame_is_fail_soft():
    assert hud_ocr.read_satanic_zone(None, ZONES)[0] is None


def test_missing_engine_is_fail_soft():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def broken_ocr(img):
        raise RuntimeError("engine exploded")

    name, ratio, _ = hud_ocr.read_satanic_zone(frame, ZONES, ocr=broken_ocr)
    assert name is None and ratio == 0.0
