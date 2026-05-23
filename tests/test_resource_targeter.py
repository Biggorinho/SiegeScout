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
