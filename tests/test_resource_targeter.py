import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from state._resource import ResourceTargeter, ResourceTarget


def test_new_targeter_has_no_active_target():
    rt = ResourceTargeter()
    assert rt.active is None


def test_reset_clears_state():
    rt = ResourceTargeter()
    rt._active = ResourceTarget(pos=(10, 20), kind="item", acquired_at=1.0, close_since=None)
    rt._visited.append(((10, 20), 999.0))
    rt.reset()
    assert rt.active is None
    assert rt._visited == []


def test_release_drops_target_without_adding_to_visited():
    rt = ResourceTargeter()
    rt._active = ResourceTarget(pos=(10, 20), kind="item", acquired_at=1.0, close_since=None)
    rt.release()
    assert rt.active is None
    assert rt._visited == []


def test_tick_returns_none_when_no_dots():
    rt = ResourceTargeter()
    target = rt.tick(now=1.0, center=(50, 50), item_dots=[], ore_dots=[])
    assert target is None


def test_tick_selects_item_before_ore_when_both_present():
    rt = ResourceTargeter()
    # Ore is closer in pixels, but item must still win on priority.
    target = rt.tick(
        now=1.0,
        center=(50, 50),
        item_dots=[[80, 50, 1, "star"]],   # 30 px away
        ore_dots=[[55, 50, 3, "pickaxe"]],  # 5  px away
    )
    assert target is not None
    assert target.kind == "item"
    assert target.pos == (80, 50)


def test_tick_selects_nearest_item_when_multiple():
    rt = ResourceTargeter()
    target = rt.tick(
        now=1.0,
        center=(50, 50),
        item_dots=[[200, 50, 1, "star"], [60, 50, 1, "star"], [150, 50, 1, "star"]],
        ore_dots=[],
    )
    assert target.pos == (60, 50)


def test_tick_selects_nearest_ore_when_no_items():
    rt = ResourceTargeter()
    target = rt.tick(
        now=1.0,
        center=(50, 50),
        item_dots=[],
        ore_dots=[[200, 50, 3, "pickaxe"], [55, 60, 3, "pickaxe"]],
    )
    assert target.kind == "ore"
    assert target.pos == (55, 60)


def test_mark_attempted_adds_to_visited_and_clears_active():
    rt = ResourceTargeter(memory_seconds=12.0)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    rt.mark_attempted(now=2.0)
    assert rt.active is None
    assert len(rt._visited) == 1
    pos, expires_at = rt._visited[0]
    assert pos == (60, 50)
    assert expires_at == 14.0   # 2.0 + 12.0


def test_excludes_visited_spots_within_memory_window():
    rt = ResourceTargeter(memory_seconds=12.0, spot_match_radius_px=8)
    # First: pick and attempt the spot at (60, 50).
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    rt.mark_attempted(now=2.0)
    # Within memory window: same spot must be skipped, fall back to ore.
    target = rt.tick(
        now=5.0,
        center=(50, 50),
        item_dots=[[60, 50, 1, "star"]],            # blocked by memory
        ore_dots=[[55, 60, 3, "pickaxe"]],
    )
    assert target.kind == "ore"


def test_visited_spot_expires_after_memory_seconds():
    rt = ResourceTargeter(memory_seconds=12.0, spot_match_radius_px=8)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    rt.mark_attempted(now=2.0)
    # After memory window: same spot can be re-selected.
    target = rt.tick(now=20.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    assert target is not None
    assert target.pos == (60, 50)


def test_visited_match_uses_spot_match_radius():
    rt = ResourceTargeter(memory_seconds=12.0, spot_match_radius_px=8)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    rt.mark_attempted(now=2.0)
    # Jitter of 5 px — still within match radius 8 — blocked.
    target = rt.tick(now=3.0, center=(50, 50), item_dots=[[65, 50, 1, "star"]], ore_dots=[])
    assert target is None
    # Jitter of 12 px — outside match radius — allowed.
    target = rt.tick(now=3.0, center=(50, 50), item_dots=[[72, 50, 1, "star"]], ore_dots=[])
    assert target is not None
    assert target.pos == (72, 50)


def test_reset_clears_everything_including_visited():
    rt = ResourceTargeter(memory_seconds=12.0)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    rt.mark_attempted(now=2.0)
    assert len(rt._visited) == 1
    rt.reset()
    assert rt._visited == []


def test_releases_target_when_dot_disappears_from_minimap():
    rt = ResourceTargeter(spot_match_radius_px=8, miss_grace_frames=3)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    assert rt.active is not None
    # Three consecutive miss frames — target must drop, no visited entry.
    rt.tick(now=2.0, center=(50, 50), item_dots=[], ore_dots=[])
    rt.tick(now=2.1, center=(50, 50), item_dots=[], ore_dots=[])
    result = rt.tick(now=2.2, center=(50, 50), item_dots=[], ore_dots=[])
    assert result is None
    assert rt.active is None
    assert rt._visited == []


def test_keeps_target_through_single_miss_frame():
    # Game-icon animation can hide a dot for one frame — tolerate it.
    rt = ResourceTargeter(spot_match_radius_px=8, miss_grace_frames=3)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    assert rt.active is not None
    # One miss frame, then the dot is back. Target must survive.
    rt.tick(now=2.0, center=(50, 50), item_dots=[], ore_dots=[])
    result = rt.tick(now=2.1, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    assert result is not None
    assert result.pos == (60, 50)
    assert result.consecutive_misses == 0  # counter reset on re-acquire


def test_keeps_target_through_minor_minimap_jitter():
    """A dot reported a few px off is still the same target.

    This used to also assert the anchor stayed at the ORIGINAL position, on
    the reading that the offset was detector jitter worth ignoring. It is not
    jitter: the minimap is player-centred, so the offset is mostly the
    player's own movement, and refusing to follow it makes the anchor go
    stale and the target get dropped mid-approach. See
    `test_target_follows_its_dot_as_the_player_walks_toward_it`. What this
    test actually pins -- that a small offset must not drop the target --
    is unchanged.
    """
    rt = ResourceTargeter(spot_match_radius_px=8)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    result = rt.tick(now=2.0, center=(50, 50), item_dots=[[63, 52, 1, "star"]], ore_dots=[])
    assert result is not None
    assert result.pos == (63, 52)   # anchor follows the dot


def test_releases_target_after_max_travel_seconds():
    rt = ResourceTargeter(max_travel_seconds=8.0, spot_match_radius_px=8)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    # Past timeout — even though the dot is still there, drop it AND remember it.
    result = rt.tick(now=10.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    assert result is None
    assert len(rt._visited) == 1
    assert rt._visited[0][0] == (60, 50)


def test_is_close_returns_false_without_target():
    rt = ResourceTargeter()
    assert rt.is_close(center=(50, 50), threshold_px=20) is False


def test_is_close_uses_euclidean_distance():
    rt = ResourceTargeter()
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    # Player at (52, 50) -> distance to (60, 50) is 8.
    assert rt.is_close(center=(52, 50), threshold_px=10) is True
    assert rt.is_close(center=(52, 50), threshold_px=5)  is False


def test_close_since_set_on_first_close_tick():
    rt = ResourceTargeter()
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    assert rt.active.close_since is None
    # Player walked next to it.
    rt.tick(now=2.0, center=(58, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[],
            close_threshold_px=10)
    assert rt.active.close_since == 2.0


def test_close_for_too_long_auto_marks_attempted():
    rt = ResourceTargeter(close_attempt_seconds=4.0)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    rt.tick(now=2.0, center=(58, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[],
            close_threshold_px=10)
    # 4 s later still close & dot still present -> auto-attempt.
    result = rt.tick(now=6.5, center=(58, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[],
                     close_threshold_px=10)
    assert result is None
    assert len(rt._visited) == 1
    assert rt._visited[0][0] == (60, 50)


def test_close_since_resets_when_player_walks_out_of_range():
    rt = ResourceTargeter()
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    # Step close - close_since is set.
    rt.tick(now=2.0, center=(58, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[],
            close_threshold_px=10)
    assert rt.active.close_since == 2.0
    # Walk far away - counter must reset.
    rt.tick(now=3.0, center=(10, 10), item_dots=[[60, 50, 1, "star"]], ore_dots=[],
            close_threshold_px=10)
    assert rt.active.close_since is None
    # Step close again - counter restarts from the new time, NOT the original 2.0.
    rt.tick(now=4.0, center=(58, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[],
            close_threshold_px=10)
    assert rt.active.close_since == 4.0


def test_resource_targeter_reexported_from_state_package():
    from state import ResourceTargeter as Re_RT, ResourceTarget as Re_RTgt
    assert Re_RT is ResourceTargeter
    assert Re_RTgt is ResourceTarget


def test_release_reason_set_on_release():
    rt = ResourceTargeter()
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    rt.release()
    assert rt.last_release_reason == "released"


def test_release_reason_disappeared_when_dot_gone():
    rt = ResourceTargeter(miss_grace_frames=3)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    # Three consecutive miss frames are required before "disappeared" fires.
    rt.tick(now=2.0, center=(50, 50), item_dots=[], ore_dots=[])
    rt.tick(now=2.1, center=(50, 50), item_dots=[], ore_dots=[])
    rt.tick(now=2.2, center=(50, 50), item_dots=[], ore_dots=[])
    assert rt.last_release_reason == "disappeared"


def test_release_reason_timeout_after_max_travel():
    rt = ResourceTargeter(max_travel_seconds=2.0)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    rt.tick(now=10.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    assert rt.last_release_reason == "timeout"


def test_release_reason_cleared_when_new_target_picked():
    rt = ResourceTargeter()
    rt.tick(now=1.0, center=(50, 50), item_dots=[[60, 50, 1, "star"]], ore_dots=[])
    rt.release()
    assert rt.last_release_reason == "released"
    rt.tick(now=2.0, center=(50, 50), item_dots=[[70, 50, 1, "star"]], ore_dots=[])
    assert rt.last_release_reason is None


# ── the minimap is player-centred, so a stationary object MOVES on it ───────

def test_target_follows_its_dot_as_the_player_walks_toward_it():
    """The anchor must track the dot, not the position it was first seen at.

    The minimap is drawn player-centred, so a stationary world object slides
    across it whenever the player moves — walking toward loot is exactly when
    its dot travels furthest. Holding the acquisition position fixed makes the
    anchor go stale, and once cumulative drift passes spot_match_radius the
    dot stops matching and the target is dropped as "disappeared" while it is
    still plainly on the minimap.

    Measured in debug/bot.log: of 209 item targets re-acquired immediately
    after a "disappeared" drop, 80 landed within 30 px of the dropped anchor
    (median 19 px) — the same dot, one 8 px window too far away.
    """
    rt = ResourceTargeter(spot_match_radius_px=8, miss_grace_frames=3)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[90, 50, 1, "star"]], ore_dots=[])
    # Player walks in; the dot slides 5 px per frame — never more than the
    # match radius in one step, but 20 px in total.
    for i, x in enumerate((85, 80, 75, 70)):
        result = rt.tick(now=2.0 + i * 0.2, center=(50, 50),
                         item_dots=[[x, 50, 1, "star"]], ore_dots=[])
        assert result is not None, f"target dropped after drifting to x={x}"
        assert result.pos == (x, 50), "anchor did not follow the dot"


def test_tracking_does_not_latch_onto_a_different_dot():
    """Following the dot must stay inside the match radius: a second star
    20 px away is a different piece of loot, not this one drifting."""
    rt = ResourceTargeter(spot_match_radius_px=8, miss_grace_frames=3)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[90, 50, 1, "star"]], ore_dots=[])
    rt.tick(now=2.0, center=(50, 50), item_dots=[[110, 50, 1, "star"]], ore_dots=[])
    assert rt.active is not None
    assert rt.active.pos == (90, 50), "anchor jumped to an unrelated dot"
    assert rt.active.consecutive_misses == 1


def test_close_tracking_uses_the_dots_current_position():
    """`is_close` and the auto-attempt timer must judge against where the dot
    is now — the whole point of the pickup hand-off."""
    rt = ResourceTargeter(spot_match_radius_px=8)
    rt.tick(now=1.0, center=(50, 50), item_dots=[[70, 50, 1, "star"]], ore_dots=[])
    assert rt.is_close(center=(50, 50), threshold_px=10) is False
    # The dot slid to 6 px from the player as they walked in.
    rt.tick(now=2.0, center=(50, 50), item_dots=[[64, 50, 1, "star"]], ore_dots=[])
    rt.tick(now=2.2, center=(50, 50), item_dots=[[56, 50, 1, "star"]], ore_dots=[])
    assert rt.active.pos == (56, 50)
    assert rt.is_close(center=(50, 50), threshold_px=10) is True
