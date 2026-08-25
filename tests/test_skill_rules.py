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
