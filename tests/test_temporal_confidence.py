import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from state._temporal_confidence import TemporalDotConfidence


def test_first_frame_not_confirmed_when_threshold_2():
    t = TemporalDotConfidence(match_radius_px=8, confirm_threshold=2)
    confirmed = t.update([[60, 50, 1, "yellow"]], now=1.0)
    assert confirmed == []


def test_second_frame_confirms_at_threshold_2():
    t = TemporalDotConfidence(match_radius_px=8, confirm_threshold=2)
    t.update([[60, 50, 1, "yellow"]], now=1.0)
    confirmed = t.update([[60, 50, 1, "yellow"]], now=1.1)
    assert len(confirmed) == 1
    assert confirmed[0][0] == 60 and confirmed[0][1] == 50


def test_dot_within_match_radius_merges():
    t = TemporalDotConfidence(match_radius_px=8, confirm_threshold=2)
    t.update([[60, 50, 1, "yellow"]], now=1.0)
    # 5 px away — within radius 8
    confirmed = t.update([[63, 52, 1, "yellow"]], now=1.1)
    assert len(confirmed) == 1


def test_dot_outside_match_radius_is_new_entry():
    t = TemporalDotConfidence(match_radius_px=8, confirm_threshold=2)
    t.update([[60, 50, 1, "yellow"]], now=1.0)
    # 12 px away — outside radius 8
    confirmed = t.update([[72, 50, 1, "yellow"]], now=1.1)
    assert confirmed == []        # new entry only at confidence 1


def test_missed_dot_decays_and_removes():
    t = TemporalDotConfidence(match_radius_px=8, confirm_threshold=2,
                              decay_per_miss=1)
    t.update([[60, 50, 1, "yellow"]], now=1.0)       # conf=1
    t.update([[60, 50, 1, "yellow"]], now=1.1)       # conf=2 (confirmed)
    t.update([], now=1.2)                             # conf=1
    confirmed = t.update([], now=1.3)                 # conf=0 -> removed
    assert confirmed == []
    # After removal, fresh dot must start at conf=1 again, NOT carry over
    confirmed2 = t.update([[60, 50, 1, "yellow"]], now=1.4)
    assert confirmed2 == []   # back to conf=1


def test_confidence_caps_at_max():
    t = TemporalDotConfidence(match_radius_px=8, confirm_threshold=2,
                              max_confidence=3)
    for i in range(10):
        t.update([[60, 50, 1, "yellow"]], now=float(i))
    # After 10 same-position frames, internal confidence should still be 3
    # We probe via decay: 3 decays of 1 each should remove it.
    t.update([], now=11.0)   # 3 -> 2
    t.update([], now=12.0)   # 2 -> 1
    t.update([], now=13.0)   # 1 -> 0 (removed)
    confirmed = t.update([[60, 50, 1, "yellow"]], now=14.0)
    assert confirmed == []   # fresh entry at conf=1


def test_reset_clears_all_state():
    t = TemporalDotConfidence(match_radius_px=8, confirm_threshold=2)
    t.update([[60, 50, 1, "yellow"]], now=1.0)
    t.update([[60, 50, 1, "yellow"]], now=1.1)   # confirmed
    t.reset()
    confirmed = t.update([[60, 50, 1, "yellow"]], now=2.0)
    assert confirmed == []   # back to conf=1
