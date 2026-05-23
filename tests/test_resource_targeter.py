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


def test_mark_attempted_adds_to_visited_and_clears_active():
    rt = ResourceTargeter(memory_seconds=12.0)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    rt.mark_attempted(now=2.0)
    assert rt.active is None
    assert len(rt._visited) == 1
    pos, expires_at = rt._visited[0]
    assert pos == (60, 50)
    assert expires_at == 14.0   # 2.0 + 12.0


def test_excludes_visited_spots_within_memory_window():
    rt = ResourceTargeter(memory_seconds=12.0, spot_match_radius_px=8)
    # First: pick and attempt the spot at (60, 50).
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    rt.mark_attempted(now=2.0)
    # Within memory window: same spot must be skipped, fall back to ore.
    target = rt.tick(
        now=5.0,
        center=(50, 50),
        item_dots=[[60, 50, 1, "star"]],            # blocked by memory
        ore_dots=[[55, 60, 3, "pickaxe"]],
    )
    assert target.kind == "ore"


def test_visited_spot_expires_after_memory_seconds():
    rt = ResourceTargeter(memory_seconds=12.0, spot_match_radius_px=8)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    rt.mark_attempted(now=2.0)
    # After memory window: same spot can be re-selected.
    target = rt.tick(now=20.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    assert target is not None
    assert target.pos == (60, 50)


def test_visited_match_uses_spot_match_radius():
    rt = ResourceTargeter(memory_seconds=12.0, spot_match_radius_px=8)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    rt.mark_attempted(now=2.0)
    # Jitter of 5 px — still within match radius 8 — blocked.
    target = rt.tick(now=3.0, center=(50, 50), item_dots=[[65, 50, 1, "star"]], ore_dots=[])
    assert target is None
    # Jitter of 12 px — outside match radius — allowed.
    target = rt.tick(now=3.0, center=(50, 50), item_dots=[[72, 50, 1, "star"]], ore_dots=[])
    assert target is not None
    assert target.pos == (72, 50)


def test_reset_clears_everything_including_visited():
    rt = ResourceTargeter(memory_seconds=12.0)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    rt.mark_attempted(now=2.0)
    assert len(rt._visited) == 1
    rt.reset()
    assert rt._visited == []
