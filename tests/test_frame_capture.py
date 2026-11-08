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


# ── which frames are worth saving for on-screen loot labeling ───────────────

from debug_utils import loot_capture_reason


def test_post_kill_window_is_a_reason_on_its_own():
    """Loot drops on kill, so the label is on screen right then. This is the
    only trigger that does not depend on a detector, which matters because the
    on-screen item detector is the thing being trained."""
    assert loot_capture_reason(
        now=10.0, kill_capture_until=14.0, center=(117, 115),
        item_dots=[], ore_dots=[], near_px=40, capture_on_ore=False) == "post_kill"


def test_outside_the_kill_window_a_near_item_dot_still_counts():
    assert loot_capture_reason(
        now=20.0, kill_capture_until=14.0, center=(117, 115),
        item_dots=[[130, 120, 1, "star"]], ore_dots=[], near_px=40,
        capture_on_ore=False) == "item_dot"


def test_a_far_item_dot_is_not_a_reason():
    assert loot_capture_reason(
        now=20.0, kill_capture_until=0.0, center=(117, 115),
        item_dots=[[10, 10, 1, "star"]], ore_dots=[], near_px=40,
        capture_on_ore=False) is None


def test_ore_alone_no_longer_earns_capture_budget():
    """Ore proximity used to trigger capture, and ore is far more common than
    loot: of 250 frames ranked out of 3.5k captures the user found TWO with a
    labelable item, while ore vein labels were everywhere. Ore already has 214
    on-screen annotations; items have 10."""
    assert loot_capture_reason(
        now=20.0, kill_capture_until=0.0, center=(117, 115),
        item_dots=[], ore_dots=[[130, 120, 3, "pickaxe"]], near_px=40,
        capture_on_ore=False) is None


def test_ore_capture_can_be_switched_back_on():
    assert loot_capture_reason(
        now=20.0, kill_capture_until=0.0, center=(117, 115),
        item_dots=[], ore_dots=[[130, 120, 3, "pickaxe"]], near_px=40,
        capture_on_ore=True) == "ore_dot"


# ── per-trigger quota ───────────────────────────────────────────────────────

def _make_quota_loot(tmp_path, reason_max):
    return FrameCapture(
        enabled=False, base_dir=str(tmp_path), interval_seconds=0.0,
        max_pairs=0, loot_enabled=True, loot_interval_seconds=0.0,
        loot_max=100, loot_reason_max=reason_max,
    )


def test_post_kill_cannot_eat_the_whole_loot_budget(tmp_path):
    """The 2026-09-03 run is the reason this exists: the post-kill window
    fires on EVERY kill, and at 51 zones/hour it burned all 300 frames in 23
    minutes of an 11.6-hour session. The remaining 11.2 hours captured
    nothing -- including every moment the bot stood next to an item, which is
    the only material that would have been worth having."""
    fc = _make_quota_loot(tmp_path, {"post_kill": 2, "item_dot": 10})
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    saved = [fc.save_loot_frame(i, frame, reason="post_kill") for i in range(5)]
    assert sum(p is not None for p in saved) == 2


def test_the_rare_trigger_keeps_its_quota_after_the_common_one_is_spent(tmp_path):
    fc = _make_quota_loot(tmp_path, {"post_kill": 1, "item_dot": 3})
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for i in range(5):
        fc.save_loot_frame(i, frame, reason="post_kill")
    saved = [fc.save_loot_frame(10 + i, frame, reason="item_dot") for i in range(5)]
    assert sum(p is not None for p in saved) == 3


def test_reason_without_a_quota_is_only_bound_by_the_overall_cap(tmp_path):
    fc = _make_quota_loot(tmp_path, {"post_kill": 1})
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    saved = [fc.save_loot_frame(i, frame, reason="item_dot") for i in range(4)]
    assert sum(p is not None for p in saved) == 4
