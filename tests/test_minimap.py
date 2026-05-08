"""Tests for minimap.py — pure math helpers and walkable-mask construction."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minimap import (
    buffered_walkable_mask,
    build_raw_walkable_minimap_mask,
    build_walkable_minimap_mask,
    connected_walkable_mask,
    dot_distance,
    dot_to_screen_delta,
    get_minimap_region,
    minimap_center,
    minimap_center_from_image,
    nearest_dot,
    normalize_capture_region,
    random_explore_delta,
)


# ── pure math ──────────────────────────────────────────────────────────────


def test_dot_distance_zero_for_same_point():
    assert dot_distance((10, 20), (10, 20)) == 0.0


def test_dot_distance_pythagorean():
    assert dot_distance((3, 4), (0, 0)) == 5.0


def test_dot_to_screen_delta_scales_by_4():
    # The function multiplies the offset by a fixed scale of 4.0.
    dx, dy = dot_to_screen_delta((15, 10), (10, 10))
    assert (dx, dy) == (20, 0)


def test_dot_to_screen_delta_returns_ints():
    dx, dy = dot_to_screen_delta((11, 12), (10, 10))
    assert isinstance(dx, int) and isinstance(dy, int)


def test_dot_to_screen_delta_zero_offset():
    assert dot_to_screen_delta((50, 50), (50, 50)) == (0, 0)


def test_nearest_dot_returns_closest():
    center = (50, 50)
    dots = [(70, 50), (52, 50), (50, 90)]
    assert nearest_dot(dots, center) == (52, 50)


def test_nearest_dot_returns_none_for_empty():
    assert nearest_dot([], (10, 10)) is None


def test_random_explore_delta_in_expected_range():
    # The function returns dist * (cos, sin) with dist in [80, 200].
    # Magnitude of the resulting vector should always sit in that range
    # (modulo small int-cast loss).
    for _ in range(50):
        dx, dy = random_explore_delta()
        assert isinstance(dx, int) and isinstance(dy, int)
        mag = math.hypot(dx, dy)
        assert 78 <= mag <= 201  # +/-1 px slop for int truncation


# ── center helpers ────────────────────────────────────────────────────────


def test_minimap_center_returns_tuple():
    cx, cy = minimap_center()
    assert isinstance(cx, int) and isinstance(cy, int)
    assert cx > 0 and cy > 0


def test_minimap_center_from_image_uses_image_shape():
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    assert minimap_center_from_image(img) == (30, 20)


def test_minimap_center_from_image_falls_back_when_invalid():
    fallback = minimap_center()
    assert minimap_center_from_image(None) == fallback
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert minimap_center_from_image(empty) == fallback


# ── region normalisation ──────────────────────────────────────────────────


def test_normalize_capture_region_applies_nudge():
    region = {"left": 100, "top": 50, "width": 200, "height": 150}
    nudge = {"left": 5, "top": 10, "right": 8, "bottom": 4}
    out = normalize_capture_region(region, nudge)
    # left += 5 → 105; top += 10 → 60
    # width = 200 + 8 - 5 = 203; height = 150 + 4 - 10 = 144
    assert out == {"left": 105, "top": 60, "width": 203, "height": 144}


def test_normalize_capture_region_clamps_to_screen():
    # Region way off-screen — must be clamped, never panic.
    region = {"left": 99999, "top": 99999, "width": 99999, "height": 99999}
    nudge = {"left": 0, "top": 0, "right": 0, "bottom": 0}
    out = normalize_capture_region(region, nudge)
    assert out["width"] >= 1 and out["height"] >= 1


def test_get_minimap_region_known_modes():
    small = get_minimap_region("small")
    large = get_minimap_region("large")
    # Both must return well-formed dicts; large should be wider than small.
    for r in (small, large):
        assert {"left", "top", "width", "height"} <= set(r.keys())
        assert r["width"] > 0 and r["height"] > 0
    assert large["width"] >= small["width"]


def test_get_minimap_region_unknown_mode_falls_back_to_small():
    small = get_minimap_region("small")
    other = get_minimap_region("not-a-real-mode")
    assert other == small


# ── walkable mask construction ────────────────────────────────────────────


def _open_minimap(h=64, w=64, value=240):
    """Synthetic minimap that's almost entirely walkable.

    Uses a uniform bright value (>190) so the frame falls outside the
    `_build_terrain_wall_mask` linework hue/value gate, and is uniform
    enough that Canny edge detection finds nothing.
    """
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_build_raw_walkable_minimap_mask_yields_walkable_area():
    minimap = _open_minimap()
    mask = build_raw_walkable_minimap_mask(minimap)
    assert mask is not None
    assert mask.shape == (64, 64)
    # Some chunk of the synthetic frame should be flagged walkable.
    assert int(np.count_nonzero(mask)) > 0


def test_build_walkable_minimap_mask_returns_same_when_no_terrain_memory():
    minimap = _open_minimap()
    raw = build_raw_walkable_minimap_mask(minimap)
    fused = build_walkable_minimap_mask(minimap, terrain_memory=None)
    assert np.array_equal(raw, fused)


def test_connected_walkable_mask_keeps_only_player_component():
    # Two disjoint walkable regions; only the one touching the player center stays.
    h, w = 60, 60
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:20, 10:20] = 255  # left island, contains the centre
    mask[40:50, 40:50] = 255  # right island, far from centre
    out = connected_walkable_mask(mask, center=(15, 15))
    # Left island preserved; right island wiped.
    assert out[15, 15] == 255
    assert out[45, 45] == 0


def test_connected_walkable_mask_handles_off_center_seed():
    # Player center sits on a non-walkable pixel; function should snap to nearest.
    h, w = 40, 40
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[5:10, 5:10] = 255  # small walkable block
    out = connected_walkable_mask(mask, center=(30, 30))  # center is in zeros
    assert out is not None
    # The walkable block must remain in the output.
    assert int(np.count_nonzero(out)) > 0


def test_connected_walkable_mask_returns_none_for_invalid_input():
    assert connected_walkable_mask(None, (0, 0)) is None
    empty = np.zeros((0, 0), dtype=np.uint8)
    assert connected_walkable_mask(empty, (0, 0)) is None


def test_buffered_walkable_mask_returns_none_for_invalid_minimap():
    assert buffered_walkable_mask(None, (10, 10)) is None
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert buffered_walkable_mask(empty, (10, 10)) is None


def test_buffered_walkable_mask_yields_mask_for_open_minimap():
    minimap = _open_minimap()
    mask = buffered_walkable_mask(minimap, center=(32, 32))
    assert mask is not None
    assert mask.shape == (64, 64)
    # After erosion the walkable area shrinks but shouldn't vanish entirely
    # for a wide-open synthetic frame.
    assert int(np.count_nonzero(mask)) > 0
