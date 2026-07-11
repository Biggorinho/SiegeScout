"""The pause menu renders ON TOP of the world map without hiding it, so both
detectors can be true at once. Map-flow helpers must clear the menu BEFORE
acting on the map: a menu-over-map state consumes clicks (modal), and an ESC
aimed at the map closes the menu instead — the pre-fix order made
wait_for_world_map declare success while the menu still had focus."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import combat


class _FakeUi:
    """ESC closes the top-most UI (menu over map); 'f' opens the map."""

    def __init__(self, menu=True, map_open=True):
        self.menu = menu
        self.map_open = map_open
        self.keys = []

    def tap_key(self, key, hold_time=0.0):
        self.keys.append(key)
        if key == "esc":
            if self.menu:
                self.menu = False
            elif self.map_open:
                self.map_open = False
        elif key == combat.VOTE_RESET_INTERACT_KEY:
            self.map_open = True

    def install(self, monkeypatch):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        monkeypatch.setattr(combat, "grab_frame_bgr", lambda sct, mon: frame)
        monkeypatch.setattr(combat, "is_pause_menu_visible", lambda f: self.menu)
        monkeypatch.setattr(combat, "is_world_map_visible", lambda f: self.map_open)
        monkeypatch.setattr(combat, "tap_key", self.tap_key)
        monkeypatch.setattr(combat, "sleep_interruptible", lambda s: True)


def test_wait_for_world_map_clears_menu_before_reporting_map(monkeypatch):
    ui = _FakeUi(menu=True, map_open=True)
    ui.install(monkeypatch)
    visible, _ = combat.wait_for_world_map(
        None, None, retry_interact_key=combat.VOTE_RESET_INTERACT_KEY)
    assert visible is True
    assert "esc" in ui.keys, "menu-over-map must be ESC-cleared, not reported as map"
    assert ui.menu is False


def test_recenter_closes_menu_then_map_then_reopens(monkeypatch):
    ui = _FakeUi(menu=True, map_open=True)
    ui.install(monkeypatch)
    frame = combat._reopen_world_map_for_recenter(None, None)
    assert frame is not None
    # esc (menu) -> esc (map) -> 'f' reopen; pre-fix produced ['esc', 'f'],
    # leaving the map that was never closed un-recentered.
    assert ui.keys[:3] == ["esc", "esc", combat.VOTE_RESET_INTERACT_KEY]
    assert ui.map_open is True


def test_recenter_map_only_unchanged(monkeypatch):
    ui = _FakeUi(menu=False, map_open=True)
    ui.install(monkeypatch)
    frame = combat._reopen_world_map_for_recenter(None, None)
    assert frame is not None
    assert ui.keys[:2] == ["esc", combat.VOTE_RESET_INTERACT_KEY]
