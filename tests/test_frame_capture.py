import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from debug_utils import FrameCapture


def _frame(h=720, w=1280):
    return np.full((h, w, 3), 40, dtype=np.uint8)


def _minimap(h=200, w=200):
    return np.full((h, w, 3), 80, dtype=np.uint8)


def _make(tmp_path, enabled=True, interval=2.0, max_pairs=10):
    return FrameCapture(
        enabled=enabled, base_dir=str(tmp_path),
        interval_seconds=interval, max_pairs=max_pairs,
    )


def test_disabled_capture_is_noop(tmp_path):
    cap = _make(tmp_path, enabled=False)
    assert cap.maybe_save(1.0, _frame(), _minimap()) is None
    assert list(tmp_path.iterdir()) == []


def test_enabled_capture_writes_pair_into_session_dir(tmp_path):
    cap = _make(tmp_path)
    path = cap.maybe_save(1.0, _frame(), _minimap())
    assert path is not None and os.path.exists(path)
    sessions = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(sessions) == 1
    assert sessions[0].name.startswith("session_")
    names = sorted(p.name for p in sessions[0].iterdir())
    assert len(names) == 2
    assert names[0].startswith("frame_") and names[0].endswith(".png")
    assert names[1].startswith("minimap_") and names[1].endswith(".png")
    # Pair shares the same stamp.
    assert names[0][len("frame_"):] == names[1][len("minimap_"):]


def test_interval_rate_limits_saves(tmp_path):
    cap = _make(tmp_path, interval=2.0)
    assert cap.maybe_save(1.0, _frame(), _minimap()) is not None
    assert cap.maybe_save(2.0, _frame(), _minimap()) is None      # 1.0s later
    assert cap.maybe_save(3.5, _frame(), _minimap()) is not None  # 2.5s later


def test_capture_stops_at_max_pairs(tmp_path):
    cap = _make(tmp_path, interval=0.0, max_pairs=2)
    assert cap.maybe_save(1.0, _frame(), _minimap()) is not None
    assert cap.maybe_save(2.0, _frame(), _minimap()) is not None
    assert cap.maybe_save(3.0, _frame(), _minimap()) is None
    session = next(d for d in tmp_path.iterdir() if d.is_dir())
    assert len(list(session.iterdir())) == 4  # 2 pairs


def test_none_or_empty_images_are_skipped(tmp_path):
    cap = _make(tmp_path)
    assert cap.maybe_save(1.0, None, _minimap()) is None
    assert cap.maybe_save(1.0, _frame(), None) is None
    assert cap.maybe_save(1.0, np.zeros((0, 0, 3), np.uint8), _minimap()) is None
    # Skips must not consume the rate-limit window.
    assert cap.maybe_save(1.5, _frame(), _minimap()) is not None


def test_disk_error_disables_capture(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a directory")
    cap = FrameCapture(enabled=True, base_dir=str(blocker / "sub"),
                       interval_seconds=0.0, max_pairs=10)
    assert cap.maybe_save(1.0, _frame(), _minimap()) is None
    # Stays disabled, does not raise on further calls.
    assert cap.maybe_save(2.0, _frame(), _minimap()) is None


def _make_loot(tmp_path, loot_enabled=True, loot_interval=1.5, loot_max=5):
    return FrameCapture(
        enabled=True, base_dir=str(tmp_path), interval_seconds=2.0,
        max_pairs=10, loot_enabled=loot_enabled,
        loot_interval_seconds=loot_interval, loot_max=loot_max,
    )


def test_loot_capture_disabled_by_default(tmp_path):
    cap = _make(tmp_path)  # no loot_* args -> loot capture off
    assert cap.save_loot_frame(1.0, _frame()) is None


def test_loot_frame_saved_into_loot_subdir(tmp_path):
    cap = _make_loot(tmp_path)
    path = cap.save_loot_frame(1.0, _frame())
    assert path is not None and os.path.exists(path)
    assert os.path.basename(os.path.dirname(path)) == "loot"
    assert os.path.basename(path).startswith("lootframe_")


def test_loot_capture_has_independent_rate_limit_and_cap(tmp_path):
    cap = _make_loot(tmp_path, loot_interval=1.5, loot_max=2)
    assert cap.save_loot_frame(1.0, _frame()) is not None
    assert cap.save_loot_frame(2.0, _frame()) is None      # 1.0s < 1.5s
    assert cap.save_loot_frame(3.0, _frame()) is not None   # 2.0s later
    assert cap.save_loot_frame(9.0, _frame()) is None       # cap reached
    # The eval-pair capture is unaffected by loot saves.
    assert cap.maybe_save(1.0, _frame(), _minimap()) is not None


def test_loot_capture_skips_empty_frame(tmp_path):
    cap = _make_loot(tmp_path)
    assert cap.save_loot_frame(1.0, None) is None
    assert cap.save_loot_frame(1.0, np.zeros((0, 0, 3), np.uint8)) is None
