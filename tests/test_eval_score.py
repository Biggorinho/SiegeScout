import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.score import (
    match_points, box_iou, match_boxes, precision_recall, score_terrain_strokes,
)


def test_match_points_exact_and_missed():
    truth = [(10, 10), (50, 50), (90, 90)]
    pred = [(12, 11), (49, 52), (200, 200)]
    tp, fp, fn = match_points(pred, truth, max_dist=6.0)
    assert (tp, fp, fn) == (2, 1, 1)


def test_match_points_greedy_does_not_reuse_truth():
    truth = [(10, 10)]
    pred = [(11, 10), (12, 10)]  # both near the single truth point
    tp, fp, fn = match_points(pred, truth, max_dist=6.0)
    assert (tp, fp, fn) == (1, 1, 0)


def test_match_points_empty_inputs():
    assert match_points([], [], max_dist=6.0) == (0, 0, 0)
    assert match_points([(1, 1)], [], max_dist=6.0) == (0, 1, 0)
    assert match_points([], [(1, 1)], max_dist=6.0) == (0, 0, 1)


def test_box_iou_known_value():
    a = (0, 0, 10, 10)
    b = (5, 0, 10, 10)  # overlap 5x10=50, union 200-50=150
    assert box_iou(a, b) == pytest.approx(50.0 / 150.0)
    assert box_iou(a, a) == pytest.approx(1.0)
    assert box_iou(a, (100, 100, 5, 5)) == 0.0


def test_match_boxes_by_iou():
    truth = [(0, 0, 10, 10), (100, 100, 20, 20)]
    pred = [(1, 1, 10, 10), (300, 300, 10, 10)]
    tp, fp, fn = match_boxes(pred, truth, min_iou=0.4)
    assert (tp, fp, fn) == (1, 1, 1)


def test_precision_recall_f1():
    p, r, f1 = precision_recall(8, 2, 2)
    assert p == pytest.approx(0.8)
    assert r == pytest.approx(0.8)
    assert f1 == pytest.approx(0.8)
    assert precision_recall(0, 0, 0) == (0.0, 0.0, 0.0)


def test_score_terrain_strokes():
    # 100x100 walkable mask: left half walkable (255), right half blocked (0).
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:, :50] = 255
    strokes = {
        # "wrongly_blocked" strokes assert pixels SHOULD be walkable.
        # Circle at (25, 50) r=5 sits fully in the walkable half -> fixed.
        "wrongly_blocked": [{"x": 25, "y": 50, "r": 5}],
        # "wrongly_open" strokes assert pixels SHOULD be blocked.
        # Circle at (25, 20) r=5 sits fully in the walkable half -> NOT fixed.
        "wrongly_open": [{"x": 25, "y": 20, "r": 5}],
    }
    result = score_terrain_strokes(mask, strokes)
    assert result["wrongly_blocked"] == pytest.approx(1.0)
    assert result["wrongly_open"] == pytest.approx(0.0)


def test_score_terrain_strokes_empty_modes_are_none():
    mask = np.zeros((10, 10), dtype=np.uint8)
    result = score_terrain_strokes(mask, {"wrongly_blocked": [], "wrongly_open": []})
    assert result == {"wrongly_blocked": None, "wrongly_open": None}
