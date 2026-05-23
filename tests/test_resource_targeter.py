import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from state._resource import ResourceTargeter, ResourceTarget


def test_new_targeter_has_no_active_target():
    rt = ResourceTargeter()
    assert rt.active is None


def test_reset_clears_state():
    rt = ResourceTargeter()
    rt._active = ResourceTarget(pos=(10, 20), kind="item", acquired_at=1.0, close_since=None)
    rt._visited.append(((10, 20), 999.0))
    rt.reset()
    assert rt.active is None
    assert rt._visited == []


def test_release_drops_target_without_adding_to_visited():
    rt = ResourceTargeter()
    rt._active = ResourceTarget(pos=(10, 20), kind="item", acquired_at=1.0, close_since=None)
    rt.release()
    assert rt.active is None
    assert rt._visited == []


def test_tick_returns_none_when_no_dots():
    rt = ResourceTargeter()
    target = rt.tick(now=1.0, center=(50, 50), item_dots=[], ore_dots=[])
    assert target is None


def test_tick_selects_item_before_ore_when_both_present():
    rt = ResourceTargeter()
    # Ore is closer in pixels, but item must still win on priority.
    target = rt.tick(
        now=1.0,
        center=(50, 50),
        item_dots=[[80, 50, 1, "star"]],   # 30 px away
        ore_dots=[[55, 50, 3, "pickaxe"]],  # 5  px away
    )
    assert target is not None
    assert target.kind == "item"
    assert target.pos == (80, 50)


def test_tick_selects_nearest_item_when_multiple():
    rt = ResourceTargeter()
    target = rt.tick(
        now=1.0,
        center=(50, 50),
        item_dots=[[200, 50, 1, "star"], [60, 50, 1, "star"], [150, 50, 1, "star"]],
        ore_dots=[],
    )
    assert target.pos == (60, 50)


def test_tick_selects_nearest_ore_when_no_items():
    rt = ResourceTargeter()
    target = rt.tick(
        now=1.0,
        center=(50, 50),
        item_dots=[],
        ore_dots=[[200, 50, 3, "pickaxe"], [55, 60, 3, "pickaxe"]],
    )
    assert target.kind == "ore"
    assert target.pos == (55, 60)
