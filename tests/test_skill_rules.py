"""Skill rules are data, and the bot must survive bad data.

settings.json is a plain file a customer can hand-edit, and these rules
drive key presses. A malformed row must cost that row and nothing else --
the bot starting at all is worth more than any single skill firing.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills import (
    RESERVED_KEYS, STYLES, TRIGGERS, SkillRule, parse_rules, rules_to_json,
    validate_key,
)


def test_a_well_formed_row_parses():
    rules = parse_rules([{"key": "e", "style": "tap", "interval": 10.0,
                          "trigger": "in_combat", "enabled": True}])
    assert rules == [SkillRule(key="e", style="tap", interval=10.0,
                               trigger="in_combat", hp_percent=40,
                               enabled=True)]


def test_a_malformed_row_is_dropped_without_taking_the_list_down():
    rules = parse_rules([
        {"key": "e", "style": "tap", "interval": 10.0, "trigger": "in_combat"},
        {"key": "q", "style": "nonsense", "interval": 5.0, "trigger": "always"},
        {"key": "2", "style": "tap", "interval": 5.0, "trigger": "always"},
    ])
    assert [r.key for r in rules] == ["e", "2"]


def test_garbage_in_place_of_a_list_yields_no_rules():
    for junk in (None, {}, "rules", 7):
        assert parse_rules(junk) == []


def test_reserved_keys_are_rejected_on_parse():
    """A hand-edited file must not get the bot fighting itself for a key."""
    rules = parse_rules([
        {"key": "1", "style": "tap", "interval": 4.0, "trigger": "hp_below"},
        {"key": "f", "style": "tap", "interval": 4.0, "trigger": "always"},
    ])
    assert rules == []


def test_validate_key_names_the_reason():
    assert validate_key("e") is None
    assert "health potion" in validate_key("1")
    assert "vote reset" in validate_key("f")
    assert validate_key("") is not None
    assert validate_key("ctrl") is not None       # single characters only
    assert validate_key("E") is None              # case is normalised


def test_interval_is_clamped_to_something_castable():
    assert parse_rules([{"key": "e", "style": "tap", "interval": 0.0,
                         "trigger": "always"}])[0].interval == 0.1
    assert parse_rules([{"key": "e", "style": "tap", "interval": "12.5",
                         "trigger": "always"}])[0].interval == 12.5


def test_hp_percent_defaults_and_clamps():
    assert parse_rules([{"key": "e", "style": "tap", "interval": 1.0,
                         "trigger": "hp_below"}])[0].hp_percent == 40
    assert parse_rules([{"key": "e", "style": "tap", "interval": 1.0,
                         "trigger": "hp_below", "hp_percent": 500}
                        ])[0].hp_percent == 99
    assert parse_rules([{"key": "e", "style": "tap", "interval": 1.0,
                         "trigger": "hp_below", "hp_percent": 0}
                        ])[0].hp_percent == 1


def test_round_trip_through_json_is_stable():
    rows = [{"key": "e", "style": "tap", "interval": 10.0,
             "trigger": "in_combat", "hp_percent": 40, "enabled": True}]
    assert rules_to_json(parse_rules(rows)) == rows


def test_the_vocabularies_are_exactly_what_the_gui_offers():
    assert STYLES == ("tap", "double", "aim")
    assert TRIGGERS == ("always", "in_combat", "enemy_visible", "hp_below",
                        "zone_entry", "while_moving")
    assert set(RESERVED_KEYS) == {"1", "f"}


from skills import default_rules, rules_from_settings  # noqa: E402

# The settings.json shape shipped before this change.
LEGACY = {
    "zone": "Satanic (auto)",
    "death_sw": True, "melee_sw": False,
    "rc_sw": True, "rc_cd": "10.0",
    "k3_sw": True, "k3_cd": "50.0",
    "mv_sw": False, "mv_cd": "3.0",
}


def test_migration_reproduces_todays_three_abilities():
    """An existing customer must see their setup unchanged on first launch,
    only now editable."""
    rules = rules_from_settings(LEGACY)
    assert [(r.key, r.style, r.interval, r.trigger, r.enabled) for r in rules] == [
        ("e", "tap", 10.0, "in_combat", True),
        ("3", "double", 50.0, "enemy_visible", True),
        ("r", "aim", 3.0, "while_moving", False),
    ]


def test_a_saved_rule_list_wins_over_the_legacy_keys():
    data = dict(LEGACY)
    data["skills"] = [{"key": "q", "style": "tap", "interval": 7.0,
                       "trigger": "always"}]
    assert [r.key for r in rules_from_settings(data)] == ["q"]


def test_an_empty_saved_list_is_respected_not_re_migrated():
    """Deleting every row must stay deleted."""
    assert rules_from_settings({"skills": [], **LEGACY}) == []


def test_settings_with_neither_gets_the_defaults():
    assert rules_from_settings({}) == default_rules()
    assert rules_from_settings(None) == default_rules()


def test_the_defaults_are_the_three_shipped_abilities():
    assert [(r.key, r.style, r.interval, r.trigger) for r in default_rules()] == [
        ("e", "tap", 10.0, "in_combat"),
        ("3", "double", 50.0, "enemy_visible"),
        ("r", "aim", 3.0, "while_moving"),
    ]
