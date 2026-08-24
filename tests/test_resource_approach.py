"""Approach planning for minimap resource dots (item stars / ore pickaxes).

Regression cover for a silent no-op: main.py asked `_a_star_next_waypoint` for a
waypoint to a far resource and, when A* returned None, fell through without
moving, without logging, and without releasing the target — so the bot stood
still for the full MINIMAP_RESOURCE_MAX_TRAVEL_SECONDS (8s) staring at an item
it had correctly detected. Live logs show exactly that: an ore target acquired
at 28px followed by 12s of unchanged dot counts and no further [RT] line.

A* can legitimately return None here: buffered_walkable_mask erodes walls by
WALL_BUFFER_ITERATIONS, and _nearest_walkable_cell only searches radius=6 grid
cells (18 minimap px at PATH_GRID_STEP=3), so a dot lying inside an eroded
region has no reachable cell.
"""

import os
import sys
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathfinding._navigation import resource_approach_delta


def _open_mask(size=96):
    return np.ones((size, size), dtype=np.uint8) * 255


def test_routes_through_astar_when_a_path_exists():
    dx, dy, routed = resource_approach_delta(_open_mask(), (48, 48), (76, 48))
    assert routed is True
    assert dx > 0, f"delta {(dx, dy)} does not head toward the target"


def test_falls_back_to_straight_line_when_no_path_exists():
    """The bug: this case used to produce no movement at all."""
    blocked = np.zeros((96, 96), dtype=np.uint8)
    dx, dy, routed = resource_approach_delta(blocked, (48, 48), (76, 48))
    assert routed is False
    assert math.hypot(dx, dy) > 0, "no-path case must still produce a move"
    assert dx > 0


def test_falls_back_when_mask_is_missing():
    dx, dy, routed = resource_approach_delta(None, (48, 48), (76, 48))
    assert routed is False
    assert math.hypot(dx, dy) > 0


def test_never_returns_a_standstill_for_a_distant_target():
    """Whatever path is taken, a far dot must never yield a (0, 0) command."""
    for mask in (_open_mask(), np.zeros((96, 96), dtype=np.uint8), None):
        for target in ((76, 48), (20, 48), (48, 80), (30, 22)):
            dx, dy, _ = resource_approach_delta(mask, (48, 48), target)
            assert math.hypot(dx, dy) > 0, f"standstill for target {target}"


def test_target_on_top_of_the_player_yields_no_move():
    dx, dy, _ = resource_approach_delta(_open_mask(), (48, 48), (48, 48))
    assert (dx, dy) == (0, 0)
