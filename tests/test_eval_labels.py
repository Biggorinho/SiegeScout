import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.labels import (
    MINIMAP_CLASSES, SCREEN_CLASSES, TERRAIN_MODES,
    FrameLabels, save_labels, load_labels, discover_pairs,
)


def test_labels_roundtrip(tmp_path):
    labels = FrameLabels(frame="frame_1.png", minimap="minimap_1.png")
    labels.minimap_points.append({"x": 10, "y": 20, "cls": "enemy"})
    labels.screen_boxes.append({"x": 5, "y": 6, "w": 30, "h": 40, "cls": "healthbar"})
    labels.terrain_strokes["wrongly_blocked"].append({"x": 50, "y": 60, "r": 4})
    path = str(tmp_path / "labels_1.json")
    save_labels(path, labels)
    loaded = load_labels(path)
    assert loaded == labels


def test_new_framelabels_starts_empty_with_both_stroke_modes():
    labels = FrameLabels()
    assert labels.minimap_points == []
    assert labels.screen_boxes == []
    assert set(labels.terrain_strokes.keys()) == set(TERRAIN_MODES)
    assert all(v == [] for v in labels.terrain_strokes.values())


def test_load_rejects_unknown_point_class(tmp_path):
    labels = FrameLabels()
    labels.minimap_points.append({"x": 1, "y": 2, "cls": "dragon"})
    path = str(tmp_path / "labels_bad.json")
    save_labels(path, labels)
    with pytest.raises(ValueError, match="dragon"):
        load_labels(path)


def test_load_rejects_unknown_box_class(tmp_path):
    labels = FrameLabels()
    labels.screen_boxes.append({"x": 1, "y": 2, "w": 3, "h": 4, "cls": "ufo"})
    path = str(tmp_path / "labels_bad.json")
    save_labels(path, labels)
    with pytest.raises(ValueError, match="ufo"):
        load_labels(path)


def test_class_constants():
    assert MINIMAP_CLASSES == ("enemy", "item", "ore", "portal", "player")
    assert SCREEN_CLASSES == ("enemy", "healthbar", "teleport")


def test_discover_pairs_matches_stamps_and_sorts(tmp_path):
    for name in ("frame_200.png", "minimap_200.png",
                 "minimap_100.png",                    # minimap-only pair
                 "frame_300.png", "minimap_300.png",
                 "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "labels_300.json").write_text("{}")
    pairs = discover_pairs(str(tmp_path))
    assert [p["stamp"] for p in pairs] == ["100", "200", "300"]
    assert pairs[0]["frame"] is None
    assert pairs[0]["minimap"].endswith("minimap_100.png")
    assert pairs[1]["frame"].endswith("frame_200.png")
    assert pairs[2]["labels"].endswith("labels_300.json")
    # Labels path is provided even when the file does not exist yet.
    assert pairs[0]["labels"].endswith("labels_100.json")


def test_discover_pairs_ignores_frames_without_minimap(tmp_path):
    (tmp_path / "frame_500.png").write_bytes(b"x")
    assert discover_pairs(str(tmp_path)) == []
