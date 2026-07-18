"""Regression tests for the 2026-07-18 19:44 Save & Exit incident.

Chain of failure (all links must stay broken):
1. is_inventory_panel_visible false-positived on MosArathim sand while the
   player idled post-kill -> the auto-ESC OPENED the pause menu (ESC toggles).
2. The menu's dim overlay made HP read 0% -> _hp_critical disabled the
   stuck-in-menu safety net, so the menu stayed up.
3. The cursor was parked on Save & Exit from earlier aiming; the jump branch
   tapped SPACE with that stale cursor -> the hovered Save & Exit activated
   and the game exited to the lobby.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import combat
import input_control
from config import SCREEN_W, SCREEN_H, JUMP_AIM_DISTANCE

_PAUSE = Path(__file__).resolve().parent / "fixtures" / "pause_menu.png"

# Pause-menu button stack measured on debug/menu_dumps/pausemenu_3rows_1784394059.png
# and debug/periodic_1784396674576.png (1920x1080): buttons span x 830-1090,
# y 395-675 (Resume y~425 ... Save & Exit y~642).
_MENU_RECT = (830, 395, 1090, 675)


# ── Fix 1: auto-ESC must verify it didn't OPEN the pause menu ────────────────

def _patch_undo_env(monkeypatch, menu_visible_after_esc, taps):
    monkeypatch.setattr(combat, "grab_frame_bgr",
                        lambda sct, mon: np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8))
    monkeypatch.setattr(combat, "is_pause_menu_visible",
                        lambda frame: menu_visible_after_esc)
    monkeypatch.setattr(combat, "_find_pause_menu_button_rows", lambda frame: [])
    monkeypatch.setattr(combat, "tap_key",
                        lambda key, hold_time=0.05: taps.append(key))
    monkeypatch.setattr(combat, "sleep_interruptible", lambda s: True)


def test_undo_esc_closes_pause_menu_opened_by_false_positive(monkeypatch):
    """If the pause menu is visible right after an auto-ESC, that ESC opened it
    (the panel detection was a false positive) — it must be toggled closed."""
    taps = []
    _patch_undo_env(monkeypatch, menu_visible_after_esc=True, taps=taps)
    assert combat.undo_esc_if_pause_menu_opened(None, None) is True
    assert taps == ["esc"]


def test_undo_esc_noop_when_no_menu_appeared(monkeypatch):
    """A legit panel close (no pause menu afterwards) must not send extra ESCs
    — a blind second ESC would itself open the pause menu."""
    taps = []
    _patch_undo_env(monkeypatch, menu_visible_after_esc=False, taps=taps)
    assert combat.undo_esc_if_pause_menu_opened(None, None) is False
    assert taps == []


def test_undo_esc_trusts_button_rows_when_skin_detector_misses(monkeypatch):
    """The red-skin detector often misses a fresh menu (fade-in); >=2 detected
    button rows must count as 'menu open' too, same as _ensure_pause_menu_open."""
    taps = []
    _patch_undo_env(monkeypatch, menu_visible_after_esc=False, taps=taps)
    monkeypatch.setattr(combat, "_find_pause_menu_button_rows",
                        lambda frame: [(425, 5000, 259), (497, 5000, 259)])
    assert combat.undo_esc_if_pause_menu_opened(None, None) is True
    assert taps == ["esc"]


# ── Fix 2: HP-critical must not blind the bot to a confirmed menu stack ──────

def test_pause_menu_stack_confirmed_on_real_menu():
    """The clean 4-button stack is the one pause-menu signal a death screen
    cannot fake — it must be recognized so the HP-critical gate can't hide an
    open menu (HP reads 0% BECAUSE the menu dim hides the bar)."""
    if not _PAUSE.exists():
        pytest.skip("pause_menu fixture missing")
    img = cv2.imread(str(_PAUSE))
    assert img is not None
    assert combat.pause_menu_stack_confirmed(img) is True


def test_pause_menu_stack_not_confirmed_on_blank_frame():
    blank = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
    assert combat.pause_menu_stack_confirmed(blank) is False


# ── Fix 3: never tap SPACE with a stale cursor ───────────────────────────────

class _FakePoint:
    def __init__(self):
        self.x = 0
        self.y = 0


def _fake_ctypes(cursor_xy):
    class _User32:
        @staticmethod
        def GetCursorPos(pt):
            pt.x, pt.y = cursor_xy
            return 1

    class _Windll:
        user32 = _User32

    class _Wintypes:
        POINT = _FakePoint

    class _Ctypes:
        wintypes = _Wintypes
        windll = _Windll

        @staticmethod
        def byref(obj):
            return obj

    return _Ctypes


def test_jump_reaims_cursor_before_space_tap(monkeypatch):
    """2026-07-18 incident: cursor was parked ON Save & Exit (963,642) when the
    aligned-jump branch tapped SPACE, activating the hovered button. The jump
    must re-aim the cursor (onto the 240px ring, provably outside the menu)
    BEFORE tapping space — never tap with a stale cursor."""
    calls = []
    monkeypatch.setattr(input_control, "ctypes", _fake_ctypes((963, 642)))
    monkeypatch.setattr(input_control, "_aim_cursor_toward_vector",
                        lambda dx, dy, distance=JUMP_AIM_DISTANCE: calls.append("aim") or True)
    monkeypatch.setattr(input_control, "tap_space",
                        lambda hold_time=0.04: calls.append("space"))

    # movement straight down, cursor 102px below center -> alignment ~0.999
    input_control.maybe_jump_with_movement(0.0, 1.0, [0.0])

    assert "space" in calls, "aligned jump should still tap space"
    assert calls.index("aim") < calls.index("space"), \
        f"cursor must be re-aimed before space, got order {calls}"


def test_jump_aim_ring_never_touches_pause_menu_buttons(monkeypatch):
    """Invariant behind fix 3: every point on the jump-aim ring lies outside
    the pause-menu button rect (farthest rect corner is ~191px from screen
    center; the ring radius is 240). Guards against config regressions."""
    positions = []
    monkeypatch.setattr(
        input_control.ctypes.windll.user32, "SetCursorPos",
        lambda x, y: positions.append((x, y)) or 1,
    )
    x0, y0, x1, y1 = _MENU_RECT
    for deg in range(0, 360, 5):
        rad = np.deg2rad(deg)
        input_control._aim_cursor_toward_vector(np.cos(rad), np.sin(rad))
    assert positions, "aim should have moved the cursor"
    for (px, py) in positions:
        assert not (x0 <= px <= x1 and y0 <= py <= y1), \
            f"aim ring point ({px},{py}) lands inside the menu button rect"
