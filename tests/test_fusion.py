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


# ── ML ore dots replacing the heuristic ────────────────────────────────────
from fusion import ml_ore_dots  # noqa: E402


def _det(x, y, conf=0.9):
    return {"x": x, "y": y, "width": 6.0, "height": 6.0, "confidence": conf}


def test_all_three_ore_classes_are_merged():
    """The model splits ore by colour; the bot only cares that it is ore."""
    dets = {"oreGold": [_det(10, 10)], "oreGreen": [_det(50, 50)],
            "oreRuby": [_det(90, 90)]}
    out = ml_ore_dots(dets)
    assert sorted((d[0], d[1]) for d in out) == [(10, 10), (50, 50), (90, 90)]


def test_shape_matches_classify_ore_dots():
    """Drop-in replacement: [x, y, tier, kind], x/y ints."""
    out = ml_ore_dots({"oreGold": [_det(10.6, 20.4)]})
    assert len(out[0]) == 4
    x, y, tier, kind = out[0]
    assert (x, y) == (11, 20)
    assert isinstance(x, int) and isinstance(y, int)
    assert isinstance(tier, int)
    assert kind == "ml"


def test_non_ore_classes_are_ignored():
    dets = {"enemy": [_det(10, 10)], "player": [_det(20, 20)],
            "portal": [_det(30, 30)], "oreGold": [_det(40, 40)]}
    out = ml_ore_dots(dets)
    assert [(d[0], d[1]) for d in out] == [(40, 40)]


def test_low_confidence_detections_are_dropped():
    out = ml_ore_dots({"oreGold": [_det(10, 10, conf=0.2)]}, min_conf=0.4)
    assert out == []


def test_same_vein_detected_as_two_colours_is_not_doubled():
    """oreGold and oreRuby firing on one vein must not become two dots, or
    the bot mines the same spot twice."""
    dets = {"oreGold": [_det(30, 30)], "oreRuby": [_det(32, 31)]}
    assert len(ml_ore_dots(dets, merge_dist=6.0)) == 1


def test_distinct_veins_are_kept_apart():
    dets = {"oreGold": [_det(30, 30), _det(60, 60)]}
    assert len(ml_ore_dots(dets, merge_dist=6.0)) == 2


def test_empty_and_missing_inputs():
    assert ml_ore_dots({}) == []
    assert ml_ore_dots(None) == []
    assert ml_ore_dots({"enemy": []}) == []


def test_input_is_not_mutated():
    dets = {"oreGold": [_det(10, 10)]}
    ml_ore_dots(dets)
    assert dets == {"oreGold": [_det(10, 10)]}
