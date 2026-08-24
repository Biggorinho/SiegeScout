"""Enemy target selection must only ever aim at temporally confirmed dots.

Regression cover for the ML phantom-lunge: the minimap model's enemy dots
flicker, and target selection used to run over ALL dots while the gate only
asked "did *any* dot confirm?". A real enemy across the room satisfied the
gate while an unconfirmed one-frame phantom was chosen as the target, so the
bot charged empty ground (enemy_dot_lost_before_confirm 30-124 per zone).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from state._temporal_confidence import TemporalDotConfidence
from targeting import confirmed_enemy_target, select_enemy_target

CENTER = (50, 50)


def _tracker(confirm_threshold=2, match_radius_px=6):
    return TemporalDotConfidence(
        match_radius_px=match_radius_px,
        confirm_threshold=confirm_threshold,
        decay_per_miss=1,
        max_confidence=8,
    )


def test_flickering_dot_never_becomes_the_target():
    """A closer dot that never repeats a position must never be aimed at."""
    tracker = _tracker()
    real = (70, 50, 1)          # 20px away, steady every frame
    # Each phantom is nearer to centre than `real` (so the old distance-based
    # pick would choose it) but lands >6px from the previous one, so it never
    # accumulates confidence.
    phantoms = [(52, 50, 1), (48, 58, 1), (58, 44, 1), (45, 45, 1)]

    seen_targets = []
    for frame, phantom in enumerate(phantoms):
        _, target = confirmed_enemy_target(
            tracker, [real, phantom], now=float(frame), center=CENTER)
        seen_targets.append(target)

    phantom_positions = {(p[0], p[1]) for p in phantoms}
    assert not (set(t for t in seen_targets if t) & phantom_positions), \
        f"aimed at an unconfirmed phantom: {seen_targets}"
    # The steady dot confirms on frame 2 (threshold 2) and stays the target.
    assert seen_targets[0] is None
    assert seen_targets[1:] == [(70, 50)] * 3


def test_no_target_until_a_dot_confirms():
    tracker = _tracker()
    confirmed, target = confirmed_enemy_target(
        tracker, [(60, 50, 1)], now=0.0, center=CENTER)
    assert confirmed == []
    assert target is None


def test_confirmed_dot_is_returned_after_threshold():
    tracker = _tracker()
    dots = [(60, 50, 1)]
    confirmed_enemy_target(tracker, dots, now=0.0, center=CENTER)
    confirmed, target = confirmed_enemy_target(
        tracker, dots, now=1.0, center=CENTER)
    assert len(confirmed) == 1
    assert target == (60, 50)


def test_empty_dots_returns_none():
    assert select_enemy_target([], CENTER) is None


def test_higher_tier_beats_closer_dot_without_mask():
    # No walkability mask: tier dominates, distance breaks ties.
    target = select_enemy_target([(55, 50, 1), (80, 50, 4)], CENTER)
    assert target == (80, 50)


def test_closest_wins_within_same_tier_without_mask():
    target = select_enemy_target([(90, 50, 2), (60, 50, 2)], CENTER)
    assert target == (60, 50)


def test_walkability_score_prefers_reachable_dot():
    # With a mask, an unreachable (score 0) dot loses to a walkable one of the
    # same tier even though it is much closer. Tier is weighted above
    # walkability on purpose, so this only holds within a tier.
    reachable, blocked = (80, 50), (55, 50)

    def walkability(mask, center, dot, samples=12):
        return 1.0 if tuple(dot) == reachable else 0.0

    target = select_enemy_target(
        [(55, 50, 1), (80, 50, 1)], CENTER,
        cached_mask=object(), walkability_fn=walkability)
    assert target == reachable


def test_single_dot_skips_walkability_scoring():
    """One candidate needs no ray casting -- mirrors the len>1 guard."""
    calls = []

    def walkability(mask, center, dot, samples=12):
        calls.append(dot)
        return 1.0

    target = select_enemy_target(
        [(60, 50, 1)], CENTER, cached_mask=object(), walkability_fn=walkability)
    assert target == (60, 50)
    assert calls == []
