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
