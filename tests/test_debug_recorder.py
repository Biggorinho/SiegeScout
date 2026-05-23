import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from debug_utils import DebugRecorder


def _blank_frame(h=720, w=1280):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _blank_minimap(h=200, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _call(rec, now, event="enter_attack", frame=None, minimap=None):
    return rec.maybe_record(
        now=now,
        event=event,
        frame_bgr=frame if frame is not None else _blank_frame(),
        minimap_bgr=minimap if minimap is not None else _blank_minimap(),
        center=(100, 100),
        enemy_dots=[],
        item_dots=[],
        ore_dots=[],
        companion_dot=None,
        nearest_enemy=None,
        resource_target=None,
        walkable_mask=None,
        enemy_color_mask=None,
        state_label="EXPLORE",
        hp_percent=1.0,
        counters_snapshot={},
    )


def test_disabled_recorder_is_noop(tmp_path):
    rec = DebugRecorder(
        enabled=False, debug_dir=str(tmp_path),
        periodic_interval=5.0, max_files=10, event_rate_limit_seconds=1.5,
    )
    result = _call(rec, now=1.0)
    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_enabled_recorder_writes_file(tmp_path):
    rec = DebugRecorder(
        enabled=True, debug_dir=str(tmp_path),
        periodic_interval=5.0, max_files=10, event_rate_limit_seconds=1.5,
    )
    path = _call(rec, now=1.0, event="enter_attack")
    assert path is not None
    assert os.path.exists(path)
    assert path.endswith(".png")
    assert "enter_attack" in os.path.basename(path)


def test_rate_limit_blocks_same_event_within_window(tmp_path):
    rec = DebugRecorder(
        enabled=True, debug_dir=str(tmp_path),
        periodic_interval=5.0, max_files=10, event_rate_limit_seconds=1.5,
    )
    p1 = _call(rec, now=1.0, event="enter_attack")
    p2 = _call(rec, now=2.0, event="enter_attack")   # 1.0 s later, blocked
    p3 = _call(rec, now=3.0, event="enter_attack")   # 2.0 s later, allowed
    assert p1 is not None
    assert p2 is None
    assert p3 is not None


def test_different_events_not_rate_limited_against_each_other(tmp_path):
    rec = DebugRecorder(
        enabled=True, debug_dir=str(tmp_path),
        periodic_interval=5.0, max_files=10, event_rate_limit_seconds=1.5,
    )
    p1 = _call(rec, now=1.0, event="enter_attack")
    p2 = _call(rec, now=1.1, event="hard_stuck")     # different event
    assert p1 is not None
    assert p2 is not None


def test_rotation_removes_oldest(tmp_path):
    rec = DebugRecorder(
        enabled=True, debug_dir=str(tmp_path),
        periodic_interval=5.0, max_files=3, event_rate_limit_seconds=0.0,
    )
    for i in range(5):
        _call(rec, now=float(i), event=f"evt{i}")
    files = sorted(tmp_path.iterdir(), key=lambda p: p.stat().st_mtime)
    assert len(files) == 3


def test_disk_error_disables_recorder(tmp_path, monkeypatch):
    import cv2
    rec = DebugRecorder(
        enabled=True, debug_dir=str(tmp_path),
        periodic_interval=5.0, max_files=10, event_rate_limit_seconds=0.0,
    )

    def _fail(*args, **kwargs):
        raise OSError("simulated disk full")
    monkeypatch.setattr(cv2, "imwrite", _fail)

    p1 = _call(rec, now=1.0)
    assert p1 is None
    # Recorder should be disabled after first failure
    monkeypatch.undo()
    p2 = _call(rec, now=2.0)
    assert p2 is None
