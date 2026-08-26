"""When a skill row is allowed to fire.

The three guards here are not decoration. Each one was spelled out at a
firing site in main.run before rules existed, and each one exists because a
cast that ignores it is wasted: mid-jump the buff is eaten, and inside the
right-click window the movement ability is swallowed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills import LoopState, SkillRule, SkillRunner


def _state(now=100.0, **kw):
    base = dict(in_combat=False, enemy_visible=False, hp=1.0,
                zone_entered=False, moving=False, aim_dx=0.0, aim_dy=0.0,
                busy_until=0.0, last_jump_t=0.0)
    base.update(kw)
    return LoopState(now=now, **base)


def _rule(**kw):
    base = dict(key="e", style="tap", interval=10.0, trigger="always")
    base.update(kw)
    return SkillRule(**base)


def test_an_always_rule_is_due_on_the_first_tick():
    runner = SkillRunner([_rule()])
    assert runner.due(_state()) == [_rule()]


def test_a_rule_that_just_fired_is_not_due_again():
    runner = SkillRunner([_rule(interval=10.0)])
    assert runner.due(_state(now=100.0))
    assert runner.due(_state(now=105.0)) == []
    assert runner.due(_state(now=110.5)) == [_rule(interval=10.0)]


def test_a_disabled_rule_is_never_due():
    runner = SkillRunner([_rule(enabled=False)])
    assert runner.due(_state()) == []


def test_in_combat_needs_combat_or_a_live_enemy():
    runner = SkillRunner([_rule(trigger="in_combat")])
    assert runner.due(_state()) == []
    assert runner.due(_state(now=200.0, in_combat=True)) != []


def test_enemy_visible_needs_an_enemy_right_now():
    runner = SkillRunner([_rule(trigger="enemy_visible")])
    assert runner.due(_state(in_combat=True)) == []
    assert runner.due(_state(now=200.0, enemy_visible=True)) != []


def test_hp_below_fires_under_the_threshold_only():
    runner = SkillRunner([_rule(trigger="hp_below", hp_percent=40)])
    assert runner.due(_state(hp=0.55)) == []
    assert runner.due(_state(now=200.0, hp=0.39)) != []


def test_hp_below_holds_off_when_the_bar_is_unreadable():
    """get_hp_percent returns None when the HP bar is not visible. Guessing
    there means firing a panic skill at full health."""
    runner = SkillRunner([_rule(trigger="hp_below", hp_percent=40)])
    assert runner.due(_state(hp=None)) == []


def test_zone_entry_fires_once_per_zone():
    runner = SkillRunner([_rule(trigger="zone_entry", interval=0.1)])
    assert runner.due(_state(now=100.0, zone_entered=True)) != []
    assert runner.due(_state(now=101.0, zone_entered=False)) == []
    assert runner.due(_state(now=102.0, zone_entered=True)) != []


def test_while_moving_needs_movement():
    runner = SkillRunner([_rule(trigger="while_moving")])
    assert runner.due(_state(moving=False)) == []
    assert runner.due(_state(now=200.0, moving=True)) != []


def test_the_right_click_window_suppresses_everything():
    """A cast inside the 0.35s right-click window is swallowed by the game."""
    runner = SkillRunner([_rule()])
    assert runner.due(_state(now=100.0, busy_until=100.2)) == []
    assert runner.due(_state(now=100.3, busy_until=100.2)) != []


def test_a_tap_rule_holds_off_just_after_a_jump():
    """Firing the buff mid-jump wastes it -- main.run guarded this with 0.42s."""
    runner = SkillRunner([_rule(style="tap")])
    assert runner.due(_state(now=100.0, last_jump_t=99.8)) == []
    assert runner.due(_state(now=100.3, last_jump_t=99.8)) != []


def test_a_suppressed_rule_does_not_burn_its_interval():
    """Being blocked by a guard is not the same as having fired."""
    runner = SkillRunner([_rule(interval=10.0)])
    assert runner.due(_state(now=100.0, busy_until=101.0)) == []
    assert runner.due(_state(now=101.5)) != []


def test_rows_fire_in_list_order_within_one_tick():
    rules = [_rule(key="e"), _rule(key="q"), _rule(key="2")]
    runner = SkillRunner(rules)
    assert [r.key for r in runner.due(_state())] == ["e", "q", "2"]


def test_replacing_the_rules_resets_the_timers():
    runner = SkillRunner([_rule()])
    runner.due(_state(now=100.0))
    runner.set_rules([_rule(key="q")])
    assert [r.key for r in runner.due(_state(now=100.1))] == ["q"]


def test_defer_all_delays_every_rule_by_its_own_interval():
    """Entering combat must not dump every skill at once -- the old code
    pushed both ability timers to now+cooldown for exactly this reason."""
    runner = SkillRunner([_rule(key="e", interval=10.0),
                          _rule(key="q", interval=2.0)])
    runner.defer_all(100.0)
    assert runner.due(_state(now=101.0)) == []
    assert [r.key for r in runner.due(_state(now=102.5))] == ["q"]
    # q keeps coming due on its own short interval; e is held off by its
    # longer one, which is the point -- defer_all is per-rule, not global.
    assert [r.key for r in runner.due(_state(now=109.9))] == ["q"]
    assert [r.key for r in runner.due(_state(now=110.5))] == ["e"]


def test_defer_all_on_a_fresh_runner_still_delays():
    """set_rules clears timers; defer_all must then push them out again."""
    runner = SkillRunner([_rule(interval=10.0)])
    runner.set_rules([_rule(interval=10.0)])
    runner.defer_all(100.0)
    assert runner.due(_state(now=105.0)) == []
