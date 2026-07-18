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


# ── 2026-07-18 22:05 incident: vote reset succeeded but was reported failed ──
# The vote takes seconds to pass (other players / auto-resolve), the world map
# auto-opens only AFTER it passes — but wait_for_world_map's default timeout is
# 3.2s, so the sequence declared failure, and the stray-map net then ESC'd away
# the auto-opened map instead of using it. The bot idled 10min in the safe zone.


def test_wait_for_world_map_honors_timeout_override(monkeypatch):
    import time as _time
    ui = _FakeUi(menu=False, map_open=False)
    ui.install(monkeypatch)
    t0 = _time.time()
    visible, _ = combat.wait_for_world_map(None, None, timeout_seconds=0.2)
    assert visible is False
    assert (_time.time() - t0) < 1.5, "short override must cut the wait short"


def test_vote_reset_sequence_waits_longer_for_vote_to_pass(monkeypatch):
    """After a verified Vote Reset click the map only appears once the vote
    passes — the sequence must wait VOTE_RESET_MAP_WAIT_SECONDS (>= 8s), not
    the default 3.2s map timeout."""
    from config import VOTE_RESET_MAP_WAIT_SECONDS
    assert VOTE_RESET_MAP_WAIT_SECONDS >= 8.0

    captured = {}

    def fake_wait(sct, mon, retry_interact_key=None, timeout_seconds=None):
        captured["timeout"] = timeout_seconds
        return False, None

    monkeypatch.setattr(combat, "_ensure_pause_menu_open", lambda s, m: True)
    monkeypatch.setattr(combat, "_click_vote_reset_menu_button",
                        lambda s, m, a: True)
    monkeypatch.setattr(combat, "release_all_inputs", lambda: None)
    monkeypatch.setattr(combat, "move_cursor_to", lambda x, y: None)
    monkeypatch.setattr(combat, "tap_key", lambda k, hold_time=0.0: None)
    monkeypatch.setattr(combat, "sleep_interruptible", lambda s: True)
    monkeypatch.setattr(combat, "wait_for_world_map", fake_wait)

    combat.run_vote_reset_sequence(None, None)
    assert captured["timeout"] == VOTE_RESET_MAP_WAIT_SECONDS


def test_handle_stray_world_map_picks_zone_instead_of_esc(monkeypatch):
    """A world map open during gameplay is (virtually always) the post-vote
    zone-select map — consume it by picking the zone, don't ESC it away."""
    ui = _FakeUi(menu=False, map_open=True)
    ui.install(monkeypatch)
    monkeypatch.setattr(combat, "pick_zone_from_open_map",
                        lambda s, m, f, target_zone=None: True)
    assert combat.handle_stray_world_map(None, None) is True
    assert "esc" not in ui.keys, "usable map must not be ESC'd away"


def test_handle_stray_world_map_escs_when_pick_fails(monkeypatch):
    ui = _FakeUi(menu=False, map_open=True)
    ui.install(monkeypatch)
    monkeypatch.setattr(combat, "pick_zone_from_open_map",
                        lambda s, m, f, target_zone=None: False)
    assert combat.handle_stray_world_map(None, None) is False
    assert ui.keys.count("esc") == 1, "unusable map falls back to a single ESC"


def test_safe_zone_walk_delta_cycles_directions():
    """F only works ON the waypoint; after a failed F the bot must walk a short
    nudge in a rotating direction (2026-07-18: 60+ F presses in place, 10min).
    The cycle must cover 4 distinct directions and wrap."""
    deltas = [combat.safe_zone_walk_delta(i) for i in range(4)]
    assert len(set(deltas)) == 4
    for dx, dy in deltas:
        assert (dx, dy) != (0, 0)
    assert combat.safe_zone_walk_delta(4) == deltas[0]
