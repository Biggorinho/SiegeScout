"""Tests for minimap ML/heuristic enemy-dot fusion (spec §3)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fusion import fuse_enemy_dots


def _ml(x, y, conf=0.9):
    return {"x": x, "y": y, "width": 5.0, "height": 5.0,
            "confidence": conf, "class": "enemy"}


def test_no_ml_returns_heuristic_unchanged():
    heur = [[10, 20, 4, "skull"], [30, 40, 1, "red"]]
    out = fuse_enemy_dots(heur, [])
    assert out == heur
    assert out is not heur  # a copy, not the same object


def test_ml_only_dot_added_with_default_tier():
    heur = [[10, 10, 4, "skull"]]
    out = fuse_enemy_dots(heur, [_ml(80, 80)], default_tier=1)
    assert [10, 10, 4, "skull"] in out
    added = [d for d in out if d[:2] == [80, 80]]
    assert len(added) == 1
    assert added[0][2] == 1          # default tier
    assert added[0][3] == "ml"       # marked as ML-sourced


def test_ml_near_heuristic_is_merged_not_duplicated():
    # ML detection within merge_dist of a heuristic dot: heuristic wins,
    # nothing added (tier stays the heuristic's, not the ML default).
    heur = [[50, 50, 3, "yellow"]]
    out = fuse_enemy_dots(heur, [_ml(53, 52)], merge_dist=6.0, default_tier=1)
    assert out == [[50, 50, 3, "yellow"]]


def test_low_confidence_ml_skipped():
    heur = []
    out = fuse_enemy_dots(heur, [_ml(80, 80, conf=0.3)], min_conf=0.5)
    assert out == []


def test_two_close_ml_dets_dedup():
    out = fuse_enemy_dots([], [_ml(80, 80), _ml(83, 81)], merge_dist=6.0)
    assert len(out) == 1


def test_input_not_mutated():
    heur = [[10, 10, 2, "purple"]]
    fuse_enemy_dots(heur, [_ml(80, 80)])
    assert heur == [[10, 10, 2, "purple"]]  # original untouched


def test_ml_coords_rounded_to_int():
    out = fuse_enemy_dots([], [_ml(80.6, 79.4)])
    assert out[0][0] == 81 and out[0][1] == 79
    assert isinstance(out[0][0], int) and isinstance(out[0][1], int)
