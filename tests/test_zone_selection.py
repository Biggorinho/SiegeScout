"""Wiring the HUD-read satanic zone through to a calibrated map click.

Two constraints drive the shape of this, and neither is a style choice.

The world map is fullscreen and hides the HUD, so the satanic zone name is
unreadable by the time a zone must be picked -- it has to be read during
gameplay and carried in. Verified against captured frames: the reader returns
None on every one.

And the map's scroll position is not something the bot can count out in wheel
notches and trust afterwards, because the map moves for reasons of its own.
The bot therefore only ever parks it hard against an edge, which is a position
it can verify, and every zone is clicked from the edge that shows it.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import combat
import config
from map_scroll import MAP_EDGE_LEFT, MAP_EDGE_RIGHT, plan_edge_click

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
    """A user-chosen zone needs no OCR - don't pay for it."""
    _quiet(monkeypatch)
    calls = []
    monkeypatch.setattr(combat, "read_satanic_zone_name",
                        lambda s, m: calls.append(1))
    monkeypatch.setattr(combat, "wait_for_world_map", lambda *a, **k: (True, FRAME))
    monkeypatch.setattr(combat, "pick_zone_from_open_map",
                        lambda *a, **k: True)
    combat.run_vote_reset_sequence(None, None, target_zone="Chilling Lake")
    assert calls == []


def _edge_env(monkeypatch):
    """Fake a map that can be parked at either edge, recording the trips."""
    monkeypatch.setattr(combat, "sleep_interruptible", lambda s: True)
    monkeypatch.setattr(combat, "move_cursor_to", lambda x, y: None)
    monkeypatch.setattr(combat, "grab_frame_bgr", lambda s, m: FRAME)
    # Map closes after the click => selection succeeded.
    monkeypatch.setattr(combat, "is_world_map_visible", lambda f: False)
    trips = []

    def fake_reopen(sct, mon, edge=MAP_EDGE_LEFT):
        trips.append(("reopen", edge))
        return FRAME

    def fake_scroll(sct, mon, edge):
        trips.append(("scroll", edge))
        return FRAME

    monkeypatch.setattr(combat, "_reopen_world_map_for_recenter", fake_reopen)
    monkeypatch.setattr(combat, "_scroll_map_to_edge", fake_scroll)
    return trips


def _click_plan(zone):
    return plan_edge_click(
        zone, combat.ZONE_EDGE_POSITIONS,
        click_y_bias=-int(config.SCREEN_H * config.MAP_NODE_CLICK_Y_OFFSET_FRAC))


def test_a_right_edge_zone_is_clicked_from_the_right_edge(monkeypatch):
    """An abyss zone is off screen at the left edge, so the map has to be
    parked right -- and clicked where it lands there, not at its world x."""
    trips = _edge_env(monkeypatch)
    clicked = []
    monkeypatch.setattr(combat, "_try_select_node_with_offsets",
                        lambda s, m, x, y, o: clicked.append((x, y)) or True)

    assert combat.pick_zone_from_open_map(
        None, None, FRAME, satanic_hint="Abyss Realm") is True

    edge, cx, cy = _click_plan("Abyss Realm")
    assert edge == MAP_EDGE_RIGHT
    assert clicked == [(cx, cy)]
    assert 0 <= cx <= config.SCREEN_W, "click must land on screen"


def test_a_known_zone_parks_at_its_own_edge_directly(monkeypatch):
    """Knowing the name means knowing the edge, so there is no reason to
    normalize left first and then travel back across the map."""
    trips = _edge_env(monkeypatch)
    monkeypatch.setattr(combat, "_try_select_node_with_offsets", lambda *a: True)
    combat.pick_zone_from_open_map(None, None, FRAME, satanic_hint="Abyss Realm")
    assert trips == [("reopen", MAP_EDGE_RIGHT)], trips


def test_a_left_edge_zone_is_never_panned_to_the_other_side(monkeypatch):
    trips = _edge_env(monkeypatch)
    monkeypatch.setattr(combat, "_try_select_node_with_offsets", lambda *a: True)
    combat.pick_zone_from_open_map(None, None, FRAME, target_zone="Chilling Lake")
    assert trips == [("reopen", MAP_EDGE_LEFT)], trips


def test_a_map_that_never_reached_its_edge_clicks_nothing_calibrated(monkeypatch):
    """If the pan did not arrive, the map sits at an offset nobody knows and
    every calibrated coordinate is fiction -- so the calibrated path must be
    skipped entirely rather than clicked hopefully."""
    monkeypatch.setattr(combat, "sleep_interruptible", lambda s: True)
    monkeypatch.setattr(combat, "move_cursor_to", lambda x, y: None)
    monkeypatch.setattr(combat, "_reopen_world_map_for_recenter",
                        lambda s, m, edge=MAP_EDGE_LEFT: None)
    monkeypatch.setattr(combat, "_scroll_map_to_edge", lambda s, m, e: None)
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
    # used the calibrated position for Abyss Realm.
    _edge, cx, cy = _click_plan("Abyss Realm")
    assert all(c[:2] != (cx, cy) for c in clicked if len(c) >= 2)


def test_the_stop_key_cuts_a_traverse_short(monkeypatch):
    """A traverse is seconds of wheel. Pressing stop must not have to wait it
    out, and an aborted pan must not be reported as an arrival."""
    monkeypatch.setattr(combat, "move_cursor_to", lambda x, y: None)
    monkeypatch.setattr(combat, "grab_frame_bgr",
                        lambda s, m: np.random.default_rng().integers(
                            0, 255, (8, 8, 3), dtype=np.uint8))
    scrolled = []
    monkeypatch.setattr(combat, "mouse_scroll", lambda n: scrolled.append(n))
    monkeypatch.setattr(combat, "sleep_interruptible", lambda s: False)

    assert combat._scroll_map_to_edge(None, None, MAP_EDGE_LEFT) is None
    assert len(scrolled) == 1, f"kept scrolling after the stop key: {scrolled}"


# -- scanning the map edges for the satanic node ----------------------------

def _scan_env(monkeypatch, per_offset):
    """Drive the scan with canned scores keyed by the edge's scroll offset."""
    monkeypatch.setattr(combat, "sleep_interruptible", lambda s: True)
    monkeypatch.setattr(combat, "move_cursor_to", lambda x, y: None)
    monkeypatch.setattr(combat, "grab_frame_bgr", lambda s, m: FRAME)
    monkeypatch.setattr(combat, "_scroll_map_to_edge", lambda s, m, e: FRAME)
    monkeypatch.setattr(combat, "score_zones_in_frame",
                        lambda f, off, margin=60: per_offset(off))


def test_scan_finds_a_zone_only_visible_at_the_right_edge(monkeypatch):
    """The whole point: a satanic zone the opening view cannot show."""
    def scores(off):
        if off > 1500:
            return [(615.0, "Abyss Realm", 100, 200)]
        return [(305.0, "Parallel Dimension", 400, 300)]
    _scan_env(monkeypatch, scores)
    name, score, edge = combat.scan_map_for_satanic_zone(None, None)
    assert name == "Abyss Realm"
    assert score == 615.0
    assert edge == MAP_EDGE_RIGHT


def test_scan_stops_at_the_left_edge_once_the_answer_is_decisive(monkeypatch):
    """A satanic node the bot can already see settles it. Panning on to the
    other edge would only cost a traverse there and a traverse back."""
    seen = []

    def scores(off):
        seen.append(off)
        return [(615.0, "The Devil's Breach", 630, 747),
                (305.0, "Crystal Village", 309, 404)]

    _scan_env(monkeypatch, scores)
    panned = []
    monkeypatch.setattr(combat, "_scroll_map_to_edge",
                        lambda s, m, e: panned.append(e) or FRAME)
    name, score, edge = combat.scan_map_for_satanic_zone(None, None)
    assert name == "The Devil's Breach"
    assert edge == MAP_EDGE_LEFT
    assert seen == [0.0], f"looked past the answer: {seen}"
    assert panned == [], f"scrolled anyway: {panned}"


def test_scan_rejects_when_nothing_stands_out(monkeypatch):
    """If the satanic node was never in view, the best ordinary node must NOT
    be crowned - that would reset into a zone at random."""
    _scan_env(monkeypatch, lambda off: [(305.0, "Parallel Dimension", 4, 3),
                                        (290.0, "Flooded Plains", 5, 6)])
    name, score, _edge = combat.scan_map_for_satanic_zone(None, None)
    assert name is None
    assert score == 0.0


def test_scan_rejects_a_clear_winner_that_is_still_too_weak(monkeypatch):
    """A big margin over nothing is not evidence of a satanic glow."""
    _scan_env(monkeypatch, lambda off: [(120.0, "Dry Hills", 4, 3),
                                        (-90.0, "Abyss Realm", 5, 6)])
    assert combat.scan_map_for_satanic_zone(None, None)[0] is None


def test_a_single_node_in_view_is_not_a_verdict(monkeypatch):
    """With no runner-up the margin check is vacuous, and a one-node view is
    exactly where the satanic node is off screen."""
    _scan_env(monkeypatch, lambda off: [(615.0, "Dry Hills", 4, 3)])
    assert combat.scan_map_for_satanic_zone(None, None)[0] is None


def test_scan_visits_both_edges_when_the_left_one_settles_nothing(monkeypatch):
    seen = []
    _scan_env(monkeypatch, lambda off: seen.append(off) or [])
    combat.scan_map_for_satanic_zone(None, None)
    assert seen[0] == 0.0
    assert max(seen) == float(config.WORLD_MAP_MAX_SCROLL_PX)


def test_a_pan_that_never_arrives_reports_an_unknown_position(monkeypatch):
    """Scoring world coordinates against a map at an unknown offset is
    fiction; the caller has to know to park it again."""
    _scan_env(monkeypatch, lambda off: [])
    monkeypatch.setattr(combat, "_scroll_map_to_edge", lambda s, m, e: None)
    name, score, edge = combat.scan_map_for_satanic_zone(None, None)
    assert name is None
    assert edge is None
