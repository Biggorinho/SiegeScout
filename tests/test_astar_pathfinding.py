"""Tests for the A* pathfinding helpers and Bresenham line-of-sight."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathfinding import (
    _a_star_full_path,
    _astar_grid_path,
    _line_is_clear,
    smooth_path,
)


def _open_mask(h=100, w=100):
    return np.full((h, w), 255, dtype=np.uint8)


# ── _line_is_clear ──────────────────────────────────────────────────────────


def test_line_is_clear_open_mask():
    mask = _open_mask()
    assert _line_is_clear(mask, (5, 5), (90, 90)) is True


def test_line_is_clear_blocked_by_wall():
    mask = _open_mask()
    mask[50, :] = 0  # full horizontal wall at row 50
    assert _line_is_clear(mask, (10, 10), (90, 90)) is False


def test_line_is_clear_endpoint_outside_mask():
    mask = _open_mask()
    assert _line_is_clear(mask, (5, 5), (150, 5)) is False


def test_line_is_clear_zero_length():
    mask = _open_mask()
    assert _line_is_clear(mask, (10, 10), (10, 10)) is True


def test_line_is_clear_zero_length_on_blocked_pixel():
    mask = _open_mask()
    mask[10, 10] = 0
    assert _line_is_clear(mask, (10, 10), (10, 10)) is False


def test_line_is_clear_handles_none():
    assert _line_is_clear(None, (0, 0), (1, 1)) is False


def test_line_is_clear_detects_one_pixel_wall():
    """Bresenham must catch a 1-pixel wall that ray-sampling could slip through."""
    mask = _open_mask()
    # Single-pixel wall along the diagonal — ray_walkability_score(samples=20)
    # could miss this; Bresenham hits every pixel on the line.
    for k in range(100):
        mask[k, k] = 0
    assert _line_is_clear(mask, (0, 0), (99, 99)) is False


# ── smooth_path ────────────────────────────────────────────────────────────


def test_smooth_path_passthrough_for_short_path():
    mask = _open_mask()
    assert smooth_path([], mask) == []
    assert smooth_path([(5, 5)], mask) == [(5, 5)]
    assert smooth_path([(5, 5), (15, 15)], mask) == [(5, 5), (15, 15)]


def test_smooth_path_passthrough_for_none_mask():
    path = [(5, 5), (10, 10), (20, 20)]
    assert smooth_path(path, None) == path


def test_smooth_path_string_pulls_open_corridor():
    mask = _open_mask()
    path = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50), (60, 60)]
    smoothed = smooth_path(path, mask)
    assert smoothed[0] == path[0]
    assert smoothed[-1] == path[-1]
    assert len(smoothed) < len(path)


def test_smooth_path_keeps_endpoints():
    mask = _open_mask()
    path = [(10, 10), (50, 50), (90, 90)]
    smoothed = smooth_path(path, mask)
    assert smoothed[0] == (10, 10)
    assert smoothed[-1] == (90, 90)


# ── _astar_grid_path / _a_star_full_path ──────────────────────────────────


def test_astar_full_path_open_area():
    mask = _open_mask()
    path = _a_star_full_path(mask, (10, 10), (90, 90))
    assert path is not None
    assert len(path) >= 2
    assert path[-1] == (90, 90) or _line_is_clear(mask, path[-1], (90, 90))


def test_astar_full_path_blocked_by_wall():
    """Full-height wall splits the mask in two; no path exists."""
    mask = _open_mask()
    mask[:, 40:60] = 0
    assert _a_star_full_path(mask, (10, 50), (90, 50)) is None


def test_astar_full_path_invalid_mask():
    assert _a_star_full_path(None, (10, 10), (50, 50)) is None
    assert _a_star_full_path(np.zeros((0, 0), dtype=np.uint8), (10, 10), (50, 50)) is None


def test_astar_grid_path_returns_singleton_when_start_equals_target():
    mask = _open_mask()
    # Same coords should collapse after grid quantisation.
    path = _astar_grid_path(mask, (50, 50), (50, 50), step=4)
    assert path is not None
    assert len(path) == 1


def test_astar_grid_path_returns_full_chain():
    mask = _open_mask()
    path = _astar_grid_path(mask, (10, 10), (90, 90), step=4)
    assert path is not None
    assert len(path) >= 2
    # Endpoints are tuples of grid cells, not minimap pixels
    assert all(isinstance(cell, tuple) and len(cell) == 2 for cell in path)
