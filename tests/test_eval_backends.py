import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import eval.backends as backends
from eval.backends import HeuristicBackend, get_backend
from eval.score import evaluate_backend


def test_get_backend_factory():
    assert isinstance(get_backend("heuristic"), HeuristicBackend)
    with pytest.raises(ValueError, match="nope"):
        get_backend("nope")


def test_heuristic_backend_maps_detector_outputs(monkeypatch):
    import detectors
    monkeypatch.setattr(detectors, "classify_enemy_dots",
                        lambda mm: [[10, 20, 4, "skull"]])
    monkeypatch.setattr(detectors, "classify_item_dots",
                        lambda mm: [[30, 40, 1, "star"]])
    monkeypatch.setattr(detectors, "classify_ore_dots", lambda mm: [])
    backend = HeuristicBackend()
    out = backend.detect_minimap(np.zeros((50, 50, 3), np.uint8))
    assert out["enemy"] == [(10, 20)]
    assert out["item"] == [(30, 40)]
    assert out["ore"] == []


def test_heuristic_backend_screen_is_empty():
    backend = HeuristicBackend()
    assert backend.detect_screen(np.zeros((50, 50, 3), np.uint8)) == {}


def _write_session(root, stamp, minimap_img, labels_dict):
    session = os.path.join(root, "session_test")
    os.makedirs(session, exist_ok=True)
    cv2.imwrite(os.path.join(session, f"minimap_{stamp}.png"), minimap_img)
    with open(os.path.join(session, f"labels_{stamp}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(labels_dict, fh)
    return session


class FakeBackend:
    name = "fake"

    def detect_minimap(self, minimap_bgr):
        return {"enemy": [(10, 10), (90, 90)], "item": [], "ore": []}

    def detect_screen(self, frame_bgr):
        return {}

    def terrain_mask(self, minimap_bgr):
        mask = np.zeros(minimap_bgr.shape[:2], dtype=np.uint8)
        mask[:, :25] = 255  # left quarter walkable
        return mask


def test_evaluate_backend_aggregates(tmp_path):
    minimap = np.zeros((50, 50, 3), dtype=np.uint8)
    labels = {
        "frame": "", "minimap": "minimap_1.png",
        # One truth enemy near a prediction (tp), one far (fn);
        # second prediction (90,90) has no truth -> fp.
        "minimap_points": [{"x": 12, "y": 11, "cls": "enemy"},
                           {"x": 40, "y": 40, "cls": "enemy"}],
        "screen_boxes": [],
        "terrain_strokes": {
            "wrongly_blocked": [{"x": 10, "y": 25, "r": 3}],  # in walkable quarter
            "wrongly_open": [],
        },
    }
    _write_session(str(tmp_path), "1", minimap, labels)
    report = evaluate_backend(FakeBackend(), str(tmp_path))
    enemy = report["classes"]["enemy@minimap"]
    assert (enemy["tp"], enemy["fp"], enemy["fn"]) == (1, 1, 1)
    assert enemy["precision"] == pytest.approx(0.5)
    assert enemy["recall"] == pytest.approx(0.5)
    assert "enemy@screen" not in report["classes"]
    assert report["terrain"]["wrongly_blocked"] == pytest.approx(1.0)
    assert report["terrain"]["wrongly_open"] is None
    assert report["pairs_scored"] == 1


def test_evaluate_backend_skips_unlabeled_pairs(tmp_path):
    session = os.path.join(str(tmp_path), "session_test")
    os.makedirs(session)
    cv2.imwrite(os.path.join(session, "minimap_9.png"),
                np.zeros((50, 50, 3), np.uint8))
    report = evaluate_backend(FakeBackend(), str(tmp_path))
    assert report["pairs_scored"] == 0
