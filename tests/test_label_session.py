import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.label_tool import LabelSession
from eval.labels import load_labels


def _pairs(tmp_path, stamps=("100", "200")):
    pairs = []
    for stamp in stamps:
        minimap = tmp_path / f"minimap_{stamp}.png"
        minimap.write_bytes(b"x")  # never decoded by LabelSession
        pairs.append({
            "stamp": stamp,
            "frame": None,
            "minimap": str(minimap),
            "labels": str(tmp_path / f"labels_{stamp}.json"),
        })
    return pairs


def test_add_point_uses_active_class(tmp_path):
    s = LabelSession(_pairs(tmp_path))
    assert s.mode == "point"
    assert s.active_class == "enemy"
    s.add_point(10, 20)
    s.cycle_class()
    assert s.active_class == "item"
    s.add_point(30, 40)
    pts = s.current_labels.minimap_points
    assert pts == [{"x": 10, "y": 20, "cls": "enemy"},
                   {"x": 30, "y": 40, "cls": "item"}]


def test_box_mode_normalizes_corners(tmp_path):
    s = LabelSession(_pairs(tmp_path))
    s.set_mode("box")
    assert s.active_class == "enemy"
    s.add_box(50, 60, 10, 20)  # dragged up-left
    box = s.current_labels.screen_boxes[0]
    assert box == {"x": 10, "y": 20, "w": 40, "h": 40, "cls": "enemy"}


def test_box_smaller_than_2x2_is_rejected(tmp_path):
    s = LabelSession(_pairs(tmp_path))
    s.set_mode("box")
    s.add_box(10, 10, 11, 11)
    assert s.current_labels.screen_boxes == []


def test_stroke_modes_write_to_matching_bucket(tmp_path):
    s = LabelSession(_pairs(tmp_path))
    s.set_mode("stroke_blocked")
    s.add_stroke(5, 6)
    s.set_mode("stroke_open")
    s.add_stroke(7, 8, r=9)
    strokes = s.current_labels.terrain_strokes
    assert strokes["wrongly_blocked"] == [{"x": 5, "y": 6, "r": 4}]
    assert strokes["wrongly_open"] == [{"x": 7, "y": 8, "r": 9}]


def test_wrong_mode_actions_are_ignored(tmp_path):
    s = LabelSession(_pairs(tmp_path))
    s.set_mode("box")
    s.add_point(1, 2)          # ignored: not in point mode
    s.add_stroke(3, 4)         # ignored: not in a stroke mode
    assert s.current_labels.minimap_points == []
    assert s.current_labels.terrain_strokes["wrongly_blocked"] == []
    with pytest.raises(ValueError):
        s.set_mode("banana")


def test_undo_removes_last_addition(tmp_path):
    s = LabelSession(_pairs(tmp_path))
    s.add_point(1, 1)
    s.set_mode("stroke_open")
    s.add_stroke(2, 2)
    s.undo()
    assert s.current_labels.terrain_strokes["wrongly_open"] == []
    assert len(s.current_labels.minimap_points) == 1
    s.undo()
    assert s.current_labels.minimap_points == []
    s.undo()  # empty: no-op, must not raise


def test_nav_autosaves_and_reloads(tmp_path):
    pairs = _pairs(tmp_path)
    s = LabelSession(pairs)
    s.add_point(10, 20)
    s.next_pair()                    # autosaves pair 0
    assert s.index == 1
    assert os.path.exists(pairs[0]["labels"])
    saved = load_labels(pairs[0]["labels"])
    assert saved.minimap_points == [{"x": 10, "y": 20, "cls": "enemy"}]
    assert saved.minimap == "minimap_100.png"
    s.prev_pair()
    assert s.current_labels.minimap_points == [{"x": 10, "y": 20, "cls": "enemy"}]
    s.prev_pair()                    # clamps at 0
    assert s.index == 0
    s.next_pair()
    s.next_pair()                    # clamps at count-1
    assert s.index == 1
