"""Tests for pathfinding._masks.adaptive_walkable_mask edge cases."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathfinding import adaptive_walkable_mask


def _open_minimap(h=64, w=64):
    """Bright synthetic frame the walkable-mask builder accepts as walkable.

    See test_minimap.py for why V>190 sidesteps the linework gate.
    """
    return np.full((h, w, 3), 240, dtype=np.uint8)


def test_returns_none_for_none_input():
    assert adaptive_walkable_mask(None, (10, 10)) is None


def test_returns_none_for_empty_input():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert adaptive_walkable_mask(empty, (10, 10)) is None


def test_returns_mask_when_no_target_given():
    minimap = _open_minimap()
    mask = adaptive_walkable_mask(minimap, center=(32, 32))
    assert mask is not None
    assert mask.shape == (64, 64)
    # On a wide-open synthetic frame the player's vicinity must be walkable
    # at *some* buffer level — the function picks the highest one.
    assert int(np.count_nonzero(mask)) > 0


def test_returns_mask_with_reachable_target():
    minimap = _open_minimap()
    mask = adaptive_walkable_mask(minimap, center=(20, 32), target=(50, 32))
    assert mask is not None
    # The path between the two endpoints should be walkable in the chosen mask.
    assert mask[32, 20] == 255 or mask[32, 50] == 255
