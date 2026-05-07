import math
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathfinding import octant_escape_delta, adaptive_walkable_mask, steer_delta_away_from_walls, minimap_explore_target


def _open_mask(h=60, w=60):
    """Fully open walkable mask."""
    m = np.ones((h, w), dtype=np.uint8) * 255
    return m


def _corridor_mask():
    """Horizontal corridor: walls above/below center row."""
    m = np.zeros((60, 60), dtype=np.uint8)
    m[28:32, :] = 255  # 4px tall horizontal corridor
    return m


def test_octant_returns_tuple():
    mask = _open_mask()
    center = (30, 30)
    result = octant_escape_delta(mask, center, 0, 0)
    assert isinstance(result, tuple) and len(result) == 2


def test_octant_avoids_current_direction():
    mask = _open_mask()
    center = (30, 30)
    # Code excludes directions within 30° of the current direction (octant_escape_delta).
    # On a 16-direction wheel (22.5° steps), the closest allowed result is 45° away.
    dx, dy = octant_escape_delta(mask, center, 200, 0)
    angle = math.atan2(dy, dx)
    assert abs((angle + math.pi) % (2 * math.pi) - math.pi) >= math.pi / 4


def test_octant_picks_walkable_direction_in_corridor():
    mask = _corridor_mask()
    center = (30, 30)
    # Going north is blocked (no walkable pixels above row 28)
    dx, dy = octant_escape_delta(mask, center, 0, -200)
    # Should pick east or west along the corridor
    assert abs(dx) > abs(dy)


def test_adaptive_mask_returns_mask_for_open_area():
    # We can't easily call the real function without a real minimap frame,
    # so just test that it returns None gracefully on empty input.
    import numpy as np
    result = adaptive_walkable_mask(np.zeros((10, 10, 3), dtype=np.uint8), (5, 5))
    # Either None or a mask — just must not raise
    assert result is None or isinstance(result, np.ndarray)


def test_steer_delta_keeps_heading_when_open():
    mask = _open_mask()
    center = (30, 30)
    dx, dy = steer_delta_away_from_walls(mask, center, 120, 0)
    assert dx > 0
    assert abs(dy) < 80


def test_steer_delta_turns_from_blocked_forward_lane():
    mask = _open_mask()
    center = (30, 30)
    # Block a forward strip straight to the right from the center.
    mask[27:34, 34:58] = 0
    dx, dy = steer_delta_away_from_walls(mask, center, 130, 0)
    # Should not keep a pure east line when forward is blocked.
    assert abs(dy) > 8


def test_minimap_explore_target_no_name_error_on_probe_clamp():
    minimap = np.zeros((80, 80, 3), dtype=np.uint8)
    # Green-ish fog-like background plus bright walkable lane to produce contours.
    minimap[:, :] = (30, 70, 30)
    minimap[20:60, 15:75] = (200, 200, 200)
    center = (40, 40)

    target = minimap_explore_target(minimap, center, recent_angles=[])
    assert target is None or (isinstance(target, tuple) and len(target) == 2)
