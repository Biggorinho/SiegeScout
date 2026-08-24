"""Wiring the HUD-read satanic zone through to a calibrated map click.

The ordering here is a real constraint, not a style choice: the world map is
fullscreen and hides the HUD, so the satanic zone name is unreadable by the
time a zone must be picked. It has to be read during gameplay and carried in.
Verified against captured frames -- the reader returns None on every one.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import combat
import config
from map_scroll import plan_zone_click

FRAME = np.zeros((1080, 1920, 3), dtype=np.uint8)


def _quiet(monkeypatch):
    monkeypatch.setattr(combat, "release_all_inputs", lambda: None)
    monkeypatch.setattr(combat, "sleep_interruptible", lambda s: True)
    monkeypatch.setattr(combat, "tap_key", lambda k, hold_time=0.0: None)
    monkeypatch.setattr(combat, "move_cursor_to", lambda x, y: None)
    monkeypatch.setattr(combat, "_ensure_pause_menu_open", lambda s, m: True)
    monkeypatch.setattr(combat, "_click_vote_reset_menu_button",
                        lambda s, m, a: True)


def test_satanic_name_is_read_before_the_map_opens(monkeypatch):
    _quiet(monkeypatch)
    order = []
    monkeypatch.setattr(combat, "read_satanic_zone_name",
                        lambda s, m: order.append("read") or "Prison Dungeon")
    monkeypatch.setattr(combat, "wait_for_world_map",
                        lambda *a, **k: (order.append("map"), (True, FRAME))[1])
    seen = {}

    def fake_pick(sct, mon, frame=None, target_zone=None, satanic_hint=None):
        seen["hint"] = satanic_hint
        return True

    monkeypatch.setattr(combat, "pick_zone_from_open_map", fake_pick)
    combat.run_vote_reset_sequence(None, None)

    assert seen["hint"] == "Prison Dungeon"
    assert order.index("read") < order.index("map"), \
        "the HUD must be read before the map covers it"


def test_explicit_zone_skips_the_hud_read(monkeypatch):
    """A user-chosen zone needs no OCR — don't pay for it."""
    _quiet(monkeypatch)
    calls = []
    monkeypatch.setattr(combat, "read_satanic_zone_name",
                        lambda s, m: calls.append(1))
    monkeypatch.setattr(combat, "wait_for_world_map", lambda *a, **k: (True, FRAME))
    monkeypatch.setattr(combat, "pick_zone_from_open_map",
                        lambda *a, **k: True)
    combat.run_vote_reset_sequence(None, None, target_zone="Chilling Lake")
    assert calls == []


def test_hint_drives_the_scroll_and_click_position(monkeypatch):
    """A far-right zone must be scrolled into view and clicked where the plan
    says -- not at its world x, which is off screen."""
    monkeypatch.setattr(combat, "sleep_interruptible", lambda s: True)
    monkeypatch.setattr(combat, "move_cursor_to", lambda x, y: None)
    monkeypatch.setattr(combat, "_reopen_world_map_for_recenter",
                        lambda s, m: FRAME)
    monkeypatch.setattr(combat, "grab_frame_bgr", lambda s, m: FRAME)
    # Map closes after the click => selection succeeded.
    monkeypatch.setattr(combat, "is_world_map_visible", lambda f: False)
    scrolled, clicked = [], []
    monkeypatch.setattr(combat, "mouse_scroll", lambda n: scrolled.append(n))

    def fake_select(sct, mon, x, y, offsets):
        clicked.append((x, y))
        return True

    monkeypatch.setattr(combat, "_try_select_node_with_offsets", fake_select)

    assert combat.pick_zone_from_open_map(
        None, None, FRAME, satanic_hint="Abyss Realm") is True

    expected = plan_zone_click(
        "Abyss Realm", config.ZONE_NODES_WORLD, config.WORLD_MAP_MAX_SCROLL_PX,
        screen_w=config.SCREEN_W, px_per_notch=config.WORLD_MAP_PX_PER_NOTCH,
        click_y_bias=-int(config.SCREEN_H * config.MAP_NODE_CLICK_Y_OFFSET_FRAC))
    notches, cx, cy = expected
    assert scrolled == [-notches], "must wheel DOWN (negative) to pan right"
    assert clicked == [(cx, cy)]
    assert 0 <= cx <= config.SCREEN_W, "click must land on screen"


def test_left_edge_zone_is_not_scrolled(monkeypatch):
    monkeypatch.setattr(combat, "sleep_interruptible", lambda s: True)
    monkeypatch.setattr(combat, "move_cursor_to", lambda x, y: None)
    monkeypatch.setattr(combat, "_reopen_world_map_for_recenter", lambda s, m: FRAME)
    monkeypatch.setattr(combat, "grab_frame_bgr", lambda s, m: FRAME)
    monkeypatch.setattr(combat, "is_world_map_visible", lambda f: False)
    scrolled = []
    monkeypatch.setattr(combat, "mouse_scroll", lambda n: scrolled.append(n))
    monkeypatch.setattr(combat, "_try_select_node_with_offsets",
                        lambda *a: True)
    combat.pick_zone_from_open_map(None, None, FRAME, target_zone="Chilling Lake")
    assert scrolled == []


def test_unnormalized_map_does_not_click_a_guessed_coordinate(monkeypatch):
    """If the left edge was never reached every coordinate is at an unknown
    offset, so the calibrated path must be skipped entirely."""
    monkeypatch.setattr(combat, "sleep_interruptible", lambda s: True)
    monkeypatch.setattr(combat, "move_cursor_to", lambda x, y: None)
    monkeypatch.setattr(combat, "_reopen_world_map_for_recenter", lambda s, m: None)
    monkeypatch.setattr(combat, "grab_frame_bgr", lambda s, m: FRAME)
    monkeypatch.setattr(combat, "is_world_map_visible", lambda f: True)
    monkeypatch.setattr(combat, "_find_satanic_node", lambda f: None)
    monkeypatch.setattr(combat, "_dump_debug_frame", lambda f, p: None)
    clicked = []
    monkeypatch.setattr(combat, "_try_select_node_with_offsets",
                        lambda *a: clicked.append(a) or True)
    monkeypatch.setattr(combat, "_click_node_and_choose",
                        lambda x, y: clicked.append((x, y)) or True)
    combat.pick_zone_from_open_map(None, None, FRAME, satanic_hint="Abyss Realm")
    # It may fall back to the generic fallback coordinate, but must never have
    # used the calibrated plan for Abyss Realm.
    plan = plan_zone_click("Abyss Realm", config.ZONE_NODES_WORLD,
                           config.WORLD_MAP_MAX_SCROLL_PX)
    assert all(c[:2] != (plan[1], plan[2]) for c in clicked if len(c) >= 2)
