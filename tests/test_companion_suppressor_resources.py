import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from state._combat import CompanionSuppressor


def _suppressor_with_companion_at(pos, suppress_radius=12):
    """Build a CompanionSuppressor that has already confirmed a companion at `pos`.
    We bypass the multi-frame confirmation by writing internal state directly.
    """
    s = CompanionSuppressor(suppress_radius=suppress_radius, confirm_frames=2)
    s._candidate = pos
    s._hits = s.confirm_frames   # mark as confirmed
    return s


def test_filter_resources_empty_inputs():
    s = _suppressor_with_companion_at((50, 50))
    items, ores = s.filter_resources([], [])
    assert items == []
    assert ores == []


def test_filter_resources_removes_dot_near_companion():
    s = _suppressor_with_companion_at((50, 50), suppress_radius=12)
    items = [[55, 50, 1, "star"]]   # 5 px from companion -> inside radius
    ores  = [[45, 55, 3, "pickaxe"]]  # ~7 px -> inside radius
    f_items, f_ores = s.filter_resources(items, ores)
    assert f_items == []
    assert f_ores == []


def test_filter_resources_keeps_dot_far_from_companion():
    s = _suppressor_with_companion_at((50, 50), suppress_radius=12)
    items = [[80, 50, 1, "star"]]   # 30 px -> outside radius
    ores  = [[10, 10, 3, "pickaxe"]]  # ~57 px -> outside radius
    f_items, f_ores = s.filter_resources(items, ores)
    assert f_items == items
    assert f_ores == ores


def test_filter_resources_when_no_companion_returns_unchanged():
    s = CompanionSuppressor()   # no _candidate set
    items = [[55, 50, 1, "star"]]
    ores  = [[45, 55, 3, "pickaxe"]]
    f_items, f_ores = s.filter_resources(items, ores)
    assert f_items == items
    assert f_ores == ores
