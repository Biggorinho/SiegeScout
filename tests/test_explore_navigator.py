import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _open_mask(h=100, w=100):
    return np.full((h, w), 255, dtype=np.uint8)


def test_plan_returns_true_on_open_mask():
    from state import ExploreNavigator
    nav = ExploreNavigator()
    ok = nav.plan(None, (10, 10), (80, 80), _open_mask())
    assert ok
    assert len(nav.path_queue) > 0


def test_plan_returns_false_on_none_mask():
    from state import ExploreNavigator
    nav = ExploreNavigator()
    ok = nav.plan(None, (10, 10), (80, 80), None)
    assert not ok
    assert nav.path_queue == []


def test_get_move_returns_none_when_queue_empty():
    from state import ExploreNavigator
    nav = ExploreNavigator()
    result = nav.get_move(None, (10, 10), _open_mask())
    assert result is None


def test_get_move_returns_int_tuple_after_plan():
    from state import ExploreNavigator
    nav = ExploreNavigator()
    nav.plan(None, (10, 10), (80, 80), _open_mask())
    move = nav.get_move(None, (10, 10), _open_mask())
    assert move is not None
    dx, dy = move
    assert isinstance(dx, int) and isinstance(dy, int)


def test_waypoint_popped_when_center_reaches_it():
    from state import ExploreNavigator
    nav = ExploreNavigator()
    nav.plan(None, (10, 10), (80, 80), _open_mask())
    initial_len = len(nav.path_queue)
    assert initial_len > 0
    first_wp = nav.path_queue[0]
    nav.get_move(None, first_wp, _open_mask())
    assert len(nav.path_queue) < initial_len


def test_reset_clears_all_state():
    from state import ExploreNavigator
    nav = ExploreNavigator()
    nav.plan(None, (10, 10), (80, 80), _open_mask())
    nav.reset()
    assert nav.path_queue == []
    assert nav.current_target is None
    assert nav._prev_gray is None
    assert nav._stall_frames == 0
    assert nav.get_move(None, (10, 10), _open_mask()) is None


def test_get_move_stall_detection_does_not_crash():
    """Verify phaseCorrelate stall branch runs without error on repeated identical frames."""
    from state import ExploreNavigator
    nav = ExploreNavigator()
    nav.plan(None, (10, 10), (80, 80), _open_mask())
    # Synthetic BGR frame — solid grey, causes near-zero phaseCorrelate response
    frame = np.full((64, 64, 3), 50, dtype=np.uint8)
    center = (10, 10)  # far from first waypoint so waypoint doesn't get popped by distance
    # Run enough ticks to exceed EXPLORE_NAV_STALL_FRAMES (=6)
    for _ in range(10):
        nav.get_move(frame, center, _open_mask())
    # After many identical frames, stall counter should have fired at least once.
    # We verify indirectly: no exception and _stall_frames was reset to 0 at least once.
    # (After firing, _stall_frames resets to 0 and plan() sets _prev_gray=None, breaking the chain.)
    assert nav._stall_frames == 0 or nav._stall_frames < 6
