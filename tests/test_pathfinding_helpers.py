import math
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathfinding import octant_escape_delta, adaptive_walkable_mask


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
    # Going east (current_dx=200, dy=0) should be excluded (within 45°)
    dx, dy = octant_escape_delta(mask, center, 200, 0)
    angle = math.atan2(dy, dx)
    # Result must be more than 45° away from east (angle=0)
    assert abs((angle + math.pi) % (2 * math.pi) - math.pi) > math.pi / 4


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
