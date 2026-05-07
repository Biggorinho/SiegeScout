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
    tm = TerrainMemory()
    tm._prev_gray = np.zeros((10, 10), dtype=np.float32)
    tm.reset()
    assert tm._prev_gray is None


def test_terrain_memory_no_crash_on_two_updates():
    """Scroll compensation must not raise when called on consecutive frames."""
    from unittest.mock import patch
    fake_mask = np.full((64, 64), 255, dtype=np.uint8)
    tm = TerrainMemory()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :] = (30, 30, 30)
    with patch("state.build_raw_walkable_minimap_mask", return_value=fake_mask):
        tm.update(frame)  # first frame — sets _prev_gray
        tm.update(frame)  # second frame — runs phaseCorrelate + warpAffine
    assert tm._prev_gray is not None


def test_terrain_memory_scroll_shifts_known_walkable():
    from unittest.mock import patch
    import cv2

    tm = TerrainMemory()
    h, w = 64, 64

    # Frame 1: bright region in left-center
    frame1 = np.zeros((h, w, 3), dtype=np.uint8)
    frame1[20:44, 5:30] = 80

    # Frame 2: same region shifted 5px right (minimap scrolled right)
    frame2 = np.zeros((h, w, 3), dtype=np.uint8)
    frame2[20:44, 10:35] = 80

    fake_mask = np.zeros((h, w), dtype=np.uint8)
    with patch("state.build_raw_walkable_minimap_mask", return_value=fake_mask):
        tm.update(frame1)
        # Set a sentinel pixel at a known position in _known_walkable
        sentinel = np.zeros((h, w), dtype=np.uint8)
        sentinel[32, 20] = 255
        tm._known_walkable = sentinel.copy()
        tm.update(frame2)

    # phaseCorrelate(frame1_gray, frame2_gray) ≈ sx=+5 (frame2 is frame1 shifted right by 5px)
    # warpAffine with M=[[1,0,-sx],[0,1,-sy]] shifts the map LEFT by ~5px
    # Sentinel was at column 20 — it should now be near column 15
    # Check that the sentinel moved leftward (columns 12-18), not stayed at 20
    assert tm._known_walkable[32, 20] == 0, "Sentinel should have moved left"
    assert tm._known_walkable[32, 15] == 255 or tm._known_walkable[32, 14] == 255 or tm._known_walkable[32, 16] == 255, \
        "Sentinel should be near column 15 after leftward shift"


def test_terrain_memory_phasecorrelate_skipped_on_identical_frames():
    """When the minimap is unchanged the FFT-scroll path must not run.

    Verified indirectly by placing a sentinel into _known_walkable and confirming
    it stays in place after a second update with an identical frame (warp would
    shift it; the early-out preserves it).
    """
    from unittest.mock import patch
    import cv2  # noqa: F401  — confirms env has cv2 available
    h, w = 64, 64
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (30, 30, 30)
    fake_mask = np.zeros((h, w), dtype=np.uint8)

    tm = TerrainMemory()
    with patch("state.build_raw_walkable_minimap_mask", return_value=fake_mask):
        tm.update(frame)
        sentinel = np.zeros((h, w), dtype=np.uint8)
        sentinel[32, 20] = 255
        tm._known_walkable = sentinel.copy()
        # Same frame again → mean-absdiff is 0 → phaseCorrelate gate skips warp.
        tm.update(frame)

    # Sentinel preserved at exact location: warp didn't run.
    assert tm._known_walkable[32, 20] == 255


def test_terrain_memory_spatial_decay_keeps_marks_near_player():
    """Marks within the keep-radius shouldn't fade; marks beyond should."""
    from state import TERRAIN_BLOCKED_DECAY_KEEP_RADIUS_PX
    from config import (
        TERRAIN_BLOCKED_DECAY_EVERY_UPDATES,
        TERRAIN_BLOCKED_DECAY_VALUE,
    )
    from unittest.mock import patch

    h, w = 240, 240
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (30, 30, 30)
    fake_mask = np.zeros((h, w), dtype=np.uint8)

    tm = TerrainMemory()
    with patch("state.build_raw_walkable_minimap_mask", return_value=fake_mask):
        tm.update(frame)  # seeds _known_blocked etc.

        cx_pl, cy_pl = w // 2, h // 2
        # Near mark: at the player position (always inside keep-radius).
        # Far mark: directly east of the player, beyond the keep-radius.
        far_x = cx_pl + TERRAIN_BLOCKED_DECAY_KEEP_RADIUS_PX + 20
        assert tm._known_blocked is not None
        tm._known_blocked[cy_pl, cx_pl] = 200
        tm._known_blocked[cy_pl, far_x] = 200

        # Run enough additional updates to trigger at least one decay cycle.
        # _updates is already 1 from the first update above; we need it to hit
        # a multiple of TERRAIN_BLOCKED_DECAY_EVERY_UPDATES.
        for _ in range(TERRAIN_BLOCKED_DECAY_EVERY_UPDATES):
            tm.update(frame)

    near_value = int(tm._known_blocked[cy_pl, cx_pl])
    far_value = int(tm._known_blocked[cy_pl, far_x])

    assert near_value == 200, f"near mark should be untouched, got {near_value}"
    assert far_value < 200, f"far mark should be reduced, got {far_value}"
    # Sanity: at least one DECAY_VALUE step was applied to the far mark.
    assert far_value <= 200 - TERRAIN_BLOCKED_DECAY_VALUE
