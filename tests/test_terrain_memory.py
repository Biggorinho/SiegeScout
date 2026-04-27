import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from state import TerrainMemory


def test_terrain_memory_mark_blocked_reduces_walkability_ahead():
    tm = TerrainMemory()

    minimap = np.zeros((64, 64, 3), dtype=np.uint8)
    minimap[:, :, :] = (30, 30, 30)
    tm.update(minimap)

    center = (32, 32)
    tm.mark_blocked(center, 120, 0)

    mask = np.ones((64, 64), dtype=np.uint8) * 255
    fused = tm.apply(mask)

    assert fused[32, 44] == 0
    dbg = tm.debug_blocked_mask(binary=True)
    assert dbg is not None
    assert tm.blocked_pixel_count() > 0


def test_terrain_memory_reset_clears_blocked_memory():
    tm = TerrainMemory()

    minimap = np.zeros((64, 64, 3), dtype=np.uint8)
    minimap[:, :, :] = (30, 30, 30)
    tm.update(minimap)

    center = (32, 32)
    tm.mark_blocked(center, 120, 0)

    mask = np.ones((64, 64), dtype=np.uint8) * 255
    blocked_before = tm.apply(mask)
    assert blocked_before[32, 44] == 0

    tm.reset()
    cleared_after = tm.apply(mask)
    assert cleared_after[32, 44] == 255
    assert tm.debug_blocked_mask(binary=True) is None
    assert tm.blocked_pixel_count() == 0


def test_terrain_memory_prev_gray_starts_none():
    tm = TerrainMemory()
    assert tm._prev_gray is None


def test_terrain_memory_prev_gray_cleared_on_reset():
    import numpy as np
    tm = TerrainMemory()
    tm._prev_gray = np.zeros((10, 10), dtype=np.float32)
    tm.reset()
    assert tm._prev_gray is None


def test_terrain_memory_no_crash_on_two_updates():
    """Scroll compensation must not raise when called on consecutive frames."""
    import numpy as np
    from unittest.mock import patch
    fake_mask = np.full((64, 64), 255, dtype=np.uint8)
    tm = TerrainMemory()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :] = (30, 30, 30)
    with patch("state.build_raw_walkable_minimap_mask", return_value=fake_mask):
        tm.update(frame)  # first frame — sets _prev_gray
        tm.update(frame)  # second frame — runs phaseCorrelate + warpAffine
    assert tm._prev_gray is not None
