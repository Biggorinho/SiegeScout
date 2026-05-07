import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from state import CompanionSuppressor


def _make(track_radius=20, confirm_frames=3, suppress_radius=12,
          reacquire_radius=20, forget_seconds=2.0, center_max_dist=30):
    return CompanionSuppressor(
        track_radius=track_radius,
        confirm_frames=confirm_frames,
        suppress_radius=suppress_radius,
        reacquire_radius=reacquire_radius,
        forget_seconds=forget_seconds,
        center_max_dist=center_max_dist,
    )


def test_passes_through_unconfirmed_dots():
    cs = _make()
    dots = [(100, 100), (105, 100)]
    # Filter has no candidate yet — nothing is suppressed.
    assert cs.filter(dots) == dots


def test_confirms_after_n_frames():
    cs = _make(confirm_frames=3)
    center = (100, 100)
    companion = (105, 100)
    # First two frames: hits accumulate but not yet confirmed.
    assert cs.update([companion], center) is None
    assert cs.update([companion], center) is None
    # Third frame: hits == confirm_frames → returns the companion.
    confirmed = cs.update([companion], center)
    assert confirmed == companion


def test_filter_suppresses_confirmed_companion():
    cs = _make(confirm_frames=2, suppress_radius=10)
    center = (100, 100)
    companion = (105, 100)
    cs.update([companion], center)
    cs.update([companion], center)  # confirmed
    # Two dots: the companion (close) plus a real enemy (far).
    enemy = (200, 100)
    out = cs.filter([companion, enemy])
    assert companion not in out
    assert enemy in out


def test_filter_passes_through_when_below_confirm_threshold():
    cs = _make(confirm_frames=5, suppress_radius=10)
    center = (100, 100)
    companion = (105, 100)
    # Only 2 hits — below confirm_frames=5.
    cs.update([companion], center)
    cs.update([companion], center)
    out = cs.filter([companion])
    assert companion in out  # not suppressed yet


def test_dot_outside_track_radius_not_picked_initially():
    cs = _make(track_radius=10)
    center = (100, 100)
    far_dot = (200, 100)  # 100 px away — outside track_radius=10
    assert cs.update([far_dot], center) is None
    assert cs._candidate is None


def test_forget_clears_candidate_after_timeout():
    cs = _make(confirm_frames=1, forget_seconds=0.05)
    center = (100, 100)
    companion = (105, 100)
    cs.update([companion], center)
    assert cs._candidate is not None
    # Wait past forget_seconds, then update with no dots.
    time.sleep(0.10)
    cs.update([], center)
    assert cs._candidate is None
    assert cs._hits == 0


def test_reacquire_keeps_hit_count_growing():
    cs = _make(confirm_frames=4, reacquire_radius=8)
    center = (100, 100)
    # Companion drifts slightly each frame but stays inside reacquire_radius.
    cs.update([(105, 100)], center)
    cs.update([(106, 101)], center)
    cs.update([(107, 100)], center)
    confirmed = cs.update([(108, 101)], center)
    assert confirmed is not None  # 4 consecutive hits → confirmed
