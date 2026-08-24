"""Scrolling the world map to a known edge.

The map pans 30px per wheel notch and is far wider than the screen, so
"scroll a fixed number of notches to reach the edge" is a guess that silently
fails: combat._reopen_world_map_for_recenter scrolls 12 notches (360px) to
"normalize to leftmost", which cannot reach the edge from most positions, and
every fixed node coordinate is then read at the wrong offset.

The edge announces itself -- panning stops changing the image -- so these
tests drive the walk against a synthetic map wider than its viewport.
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


# ── turning a world coordinate into a scroll plan ────────────────────────────
from map_scroll import scroll_plan  # noqa: E402

# Calibrated 2026-08-24: the map is 3918px wide, so it pans 1997.7px before
# clamping. That is NOT a whole number of 30px notches, so the final notch is
# short -- a plan that assumes notches * 30 lands the far-right zones wrong.
MAX_SCROLL = 1997.7
SCREEN_W = 1920


def test_zone_already_on_screen_needs_no_scrolling():
    notches, screen_x = scroll_plan(104, MAX_SCROLL)
    assert notches == 0
    assert screen_x == 104


def test_zone_is_centred_when_the_map_can_reach_it():
    notches, screen_x = scroll_plan(2660, MAX_SCROLL)
    assert abs(screen_x - SCREEN_W / 2) <= 30      # within one notch of centre
    assert notches > 0


def test_far_right_zone_accounts_for_the_short_final_notch():
    # Abyss Realm sits at world x=3605. The map cannot centre it: it clamps at
    # 1997.7px, leaving the zone at 1607 -- exactly where OCR observed it.
    notches, screen_x = scroll_plan(3605, MAX_SCROLL)
    assert abs(screen_x - 1607.3) < 1.5


def test_never_returns_negative_notches():
    for wx in (0, 50, 104, 960, 1919):
        notches, _ = scroll_plan(wx, MAX_SCROLL)
        assert notches >= 0, wx


def test_every_calibrated_zone_lands_safely_on_screen():
    """The real table must be reachable -- a zone that plans off-screen is a
    click into empty map."""
    import json
    import os
    path = os.path.join(os.path.dirname(__file__), "fixtures",
                        "zone_table.json")
    if not os.path.exists(path):
        import pytest
        pytest.skip("zone table fixture not generated yet")
    table = json.load(open(path, encoding="utf-8"))
    for name, (wx, wy) in table.items():
        notches, screen_x = scroll_plan(wx, MAX_SCROLL)
        assert 60 <= screen_x <= SCREEN_W - 60, f"{name} plans to x={screen_x}"


def test_scroll_offset_never_exceeds_the_map():
    notches, screen_x = scroll_plan(99999, MAX_SCROLL)
    assert notches * 30 <= MAX_SCROLL + 30
    assert screen_x >= 99999 - MAX_SCROLL - 1
