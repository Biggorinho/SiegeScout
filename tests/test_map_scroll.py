"""Parking the world map at a known edge, and reading zones off it.

Counting wheel notches is a guess the bot cannot check: the map can be moved
by something other than the bot -- a stray wheel event, the game itself -- and
then every coordinate is read at an offset nobody knows, so the clicks land in
empty map and the reset silently fails.

The edge announces itself instead -- panning a clamped map stops changing the
image -- and the map's two edges between them show every zone. These tests
drive the walk against a synthetic map wider than its viewport, then pin the
left/right split the real coordinates produce.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from map_scroll import scroll_to_edge

VIEW_W, VIEW_H = 320, 180
WORLD_W = 1600          # five viewports wide
PX_PER_NOTCH = 30


class FakeMap:
    """A wider-than-screen map that clamps at both edges, like the game's."""

    def __init__(self, offset=0):
        rng = np.random.default_rng(0)
        self.world = rng.integers(0, 255, (VIEW_H, WORLD_W, 3), dtype=np.uint8)
        self.offset = offset
        self.max_offset = WORLD_W - VIEW_W
        self.scrolls = 0

    def grab(self):
        return self.world[:, self.offset:self.offset + VIEW_W].copy()

    def scroll(self, notches):
        # Positive notches pan left (toward offset 0), matching the game.
        self.scrolls += 1
        self.offset = int(np.clip(self.offset - notches * PX_PER_NOTCH,
                                  0, self.max_offset))


def test_reaches_left_edge_from_the_far_right():
    m = FakeMap(offset=WORLD_W - VIEW_W)
    frames, reason = scroll_to_edge(m.grab, m.scroll, step=4, max_steps=50)
    assert reason == "edge"
    assert m.offset == 0


def test_reaches_right_edge_from_the_far_left():
    m = FakeMap(offset=0)
    frames, reason = scroll_to_edge(m.grab, m.scroll, step=-4, max_steps=50)
    assert reason == "edge"
    assert m.offset == m.max_offset


def test_already_at_the_edge_stops_immediately():
    m = FakeMap(offset=0)
    frames, reason = scroll_to_edge(m.grab, m.scroll, step=4, max_steps=50)
    assert reason == "edge"
    assert m.offset == 0


def test_gives_up_at_max_steps_without_claiming_the_edge():
    m = FakeMap(offset=WORLD_W - VIEW_W)
    # Only 2 steps of 1 notch = 60px, nowhere near the 1280px needed.
    frames, reason = scroll_to_edge(m.grab, m.scroll, step=1, max_steps=2)
    assert reason == "max_steps"
    assert m.offset > 0


def test_captures_every_frame_it_visits():
    m = FakeMap(offset=0)
    seen = []
    frames, _ = scroll_to_edge(m.grab, m.scroll, step=-8, max_steps=50,
                               on_capture=lambda i, f: seen.append(i))
    assert seen == list(range(len(frames)))
    assert len(frames) >= 2


def test_abort_signal_stops_the_walk():
    m = FakeMap(offset=WORLD_W - VIEW_W)
    calls = []

    def should_continue():
        calls.append(1)
        return len(calls) < 3          # user hits the stop key on the 3rd check

    frames, reason = scroll_to_edge(m.grab, m.scroll, step=4, max_steps=50,
                                    should_continue=should_continue)
    assert reason == "aborted"
    assert m.offset > 0                # stopped before reaching the edge


def test_a_short_first_step_confirms_an_edge_without_a_full_traverse():
    """The common case: the map is already parked where it needs to be, and
    proving that must not cost a trip across the whole map."""
    m = FakeMap(offset=0)
    scrolled = []
    frames, reason = scroll_to_edge(m.grab,
                                    lambda n: (scrolled.append(n), m.scroll(n)),
                                    step=(3, 60), max_steps=50)
    assert reason == "edge"
    assert scrolled == [3], f"traversed the map to confirm an edge: {scrolled}"


def test_the_schedule_alternates_probe_and_traverse():
    """Arriving is only learned from a scroll that fails to move the map, so
    the proof must not cost another traverse."""
    m = FakeMap(offset=WORLD_W - VIEW_W)          # 1280px away, 43 notches
    scrolled = []
    frames, reason = scroll_to_edge(m.grab,
                                    lambda n: (scrolled.append(n), m.scroll(n)),
                                    step=(3, 60), max_steps=50)
    assert reason == "edge"
    assert m.offset == 0
    assert scrolled == [3, 60, 3], scrolled


def test_overshooting_the_edge_is_harmless():
    """A step wider than the whole map just clamps -- which is what makes the
    big step in the schedule safe."""
    m = FakeMap(offset=WORLD_W - VIEW_W)
    frames, reason = scroll_to_edge(m.grab, m.scroll, step=(1, 999),
                                    max_steps=50)
    assert reason == "edge"
    assert m.offset == 0


# ── which edge shows which zone ─────────────────────────────────────────────
from map_scroll import (  # noqa: E402
    MAP_EDGE_LEFT, MAP_EDGE_RIGHT, edge_offsets, edges_with_zones,
    plan_edge_click, zone_edge_positions,
)

# Calibrated 2026-08-24: the map is 3918px wide and pans 1997.7px before
# clamping, so the left view (world x 0..1920) and the right view
# (1997.7..3917.7) cover it between them.
MAX_SCROLL = 1997.7
SCREEN_W = 1920
MARGIN = 60

TABLE = {
    "Chilling Lake": (104, 411),        # left edge, no scrolling needed
    "Forgotten Caves": (1764, 459),     # last of the left-edge zones
    "Boreal Island": (2363, 486),       # first of the abyss zones
    "Abyss Realm": (3605, 447),         # far right, where the map clamps
}


def test_near_zones_are_read_at_the_left_edge_unchanged():
    edge, x, y = zone_edge_positions(TABLE, MAX_SCROLL)["Chilling Lake"]
    assert edge == MAP_EDGE_LEFT
    assert (x, y) == (104, 411)


def test_abyss_zones_are_read_at_the_right_edge():
    """Their world x is off screen at the left edge; at the right edge the
    map has panned 1997.7px, so they land inside the frame."""
    edge, x, y = zone_edge_positions(TABLE, MAX_SCROLL)["Abyss Realm"]
    assert edge == MAP_EDGE_RIGHT
    assert abs(x - 1607) <= 1
    assert y == 447


def test_every_calibrated_zone_is_visible_at_one_of_the_two_edges():
    """A zone visible at neither edge could never be scanned or clicked."""
    import json
    import os
    path = os.path.join(os.path.dirname(__file__), "fixtures",
                        "zone_table.json")
    table = {n: tuple(v) for n, v in
             json.load(open(path, encoding="utf-8")).items()}
    positions = zone_edge_positions(table, MAX_SCROLL, SCREEN_W, MARGIN)
    missing = [n for n in table if n not in positions]
    assert missing == [], f"zones no edge shows: {missing}"


def test_positions_land_clear_of_the_screen_borders():
    import json
    import os
    path = os.path.join(os.path.dirname(__file__), "fixtures",
                        "zone_table.json")
    table = {n: tuple(v) for n, v in
             json.load(open(path, encoding="utf-8")).items()}
    for name, (_edge, x, _y) in zone_edge_positions(
            table, MAX_SCROLL, SCREEN_W, MARGIN).items():
        assert MARGIN <= x <= SCREEN_W - MARGIN, f"{name} at x={x}"


def test_a_zone_off_both_edges_is_omitted_rather_than_clamped():
    """Better no answer than a click into empty map."""
    positions = zone_edge_positions({"Nowhere": (1950, 400)}, MAX_SCROLL)
    assert "Nowhere" not in positions


def test_a_zone_visible_at_both_edges_prefers_the_left():
    """Deterministic, and it never travels for a node already on screen."""
    edge, _x, _y = zone_edge_positions(
        {"Middle": (1000, 400)}, max_scroll_px=200)["Middle"]
    assert edge == MAP_EDGE_LEFT


def test_a_map_that_cannot_scroll_has_one_edge():
    assert edge_offsets(0) == {MAP_EDGE_LEFT: 0.0, MAP_EDGE_RIGHT: 0.0}


# ── turning a zone name into a click ────────────────────────────────────────
POSITIONS = zone_edge_positions(TABLE, MAX_SCROLL)


def test_the_click_plan_names_the_edge_to_park_at():
    edge, cx, cy = plan_edge_click("Abyss Realm", POSITIONS)
    assert edge == MAP_EDGE_RIGHT
    assert abs(cx - 1607) <= 1          # not 3605, which is off screen
    assert cy == 447


def test_unknown_zone_returns_none():
    assert plan_edge_click("Atlantis", POSITIONS) is None


def test_click_y_bias_compensates_for_the_click_helpers_own_offset():
    """_click_node_and_choose adds ~6.5px before clicking; passing a negative
    bias makes the real click land on the node centre instead of its lower rim
    (median node radius is only 9.6px)."""
    _, _, cy = plan_edge_click("Chilling Lake", POSITIONS, click_y_bias=-7)
    assert cy == 411 - 7


def test_plan_is_integral_so_it_can_be_clicked():
    for name in TABLE:
        _edge, cx, cy = plan_edge_click(name, POSITIONS)
        assert isinstance(cx, int) and isinstance(cy, int)


# ── which edges a colour scan has to visit ──────────────────────────────────

def test_a_scan_visits_the_left_edge_first():
    assert edges_with_zones(POSITIONS)[0] == MAP_EDGE_LEFT


def test_a_scan_visits_both_edges_when_zones_live_at_both():
    assert edges_with_zones(POSITIONS) == (MAP_EDGE_LEFT, MAP_EDGE_RIGHT)


def test_an_edge_with_no_zones_is_not_worth_the_trip():
    left_only = zone_edge_positions({"Chilling Lake": (104, 411)}, MAX_SCROLL)
    assert edges_with_zones(left_only) == (MAP_EDGE_LEFT,)


def test_an_empty_table_still_scans_where_it_stands():
    assert edges_with_zones({}) == (MAP_EDGE_LEFT,)
