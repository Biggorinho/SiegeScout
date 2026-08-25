"""Which key the bot presses, how often, and under what condition.

The bot used to have exactly three abilities, each hardcoded in four places:
a switch in the GUI, a keyword argument into main.run, a timer nonlocal, and
a firing site in the loop with its trigger condition baked into an `if`. A
player whose skills are bound differently could not say so at all.

A rule is deliberately flat -- one key, one interval, one condition. No
AND/OR, because the conditions that matter are all things the loop already
computes, and an expression language would mostly be a way to configure the
bot into not fighting.

Import-light on purpose (like targeting.py and fusion.py): no cv2, no numpy,
no keyboard, no config. Casting lives in input_control; this module only ever
decides.
"""

from dataclasses import dataclass

# Keys the bot presses on its own. Both are tuned burst behaviours that a
# flat interval+style row cannot express, so they stay built-in and their
# keys are refused here rather than letting two systems fight over one key.
RESERVED_KEYS = {
    "1": "the health potion",
    "f": "vote reset and mining",
}

STYLES = ("tap", "double", "aim")
TRIGGERS = ("always", "in_combat", "enemy_visible", "hp_below",
            "zone_entry", "while_moving")

MIN_INTERVAL = 0.1
DEFAULT_HP_PERCENT = 40


@dataclass(frozen=True)
class SkillRule:
    key: str
    style: str
    interval: float
    trigger: str
    hp_percent: int = DEFAULT_HP_PERCENT
    enabled: bool = True


def validate_key(key):
    """Return an error message for an unusable key, or None if it is fine.

    Single characters only. Not a shortlist -- that would rule out legitimate
    binds -- but not arbitrary key names either, which would let a customer
    bind `esc` and lock themselves out of their own game.
    """
    if not isinstance(key, str):
        return "Pick a key."
    key = key.strip().lower()
    if len(key) != 1:
        return "One character only (a-z or 0-9)."
    if key in RESERVED_KEYS:
        return f"Key {key} is used by {RESERVED_KEYS[key]}."
    if not (key.isalpha() or key.isdigit()):
        return "Letters and digits only."
    return None


def _clamp(value, low, high):
    return max(low, min(high, value))


def parse_rule(row):
    """Build one SkillRule from a settings dict, or None if it is unusable."""
    if not isinstance(row, dict):
        return None
    key = row.get("key")
    if validate_key(key) is not None:
        return None
    style = row.get("style")
    trigger = row.get("trigger")
    if style not in STYLES or trigger not in TRIGGERS:
        return None
    try:
        interval = max(MIN_INTERVAL, float(row.get("interval", MIN_INTERVAL)))
    except (TypeError, ValueError):
        return None
    try:
        hp_percent = int(_clamp(int(row.get("hp_percent", DEFAULT_HP_PERCENT)), 1, 99))
    except (TypeError, ValueError):
        hp_percent = DEFAULT_HP_PERCENT
    return SkillRule(
        key=str(key).strip().lower(),
        style=style,
        interval=interval,
        trigger=trigger,
        hp_percent=hp_percent,
        enabled=bool(row.get("enabled", True)),
    )


def parse_rules(data):
    """Parse a list of settings rows, dropping the ones that cannot be used."""
    if not isinstance(data, list):
        return []
    parsed = [parse_rule(row) for row in data]
    return [rule for rule in parsed if rule is not None]


def rules_to_json(rules):
    """Serialise rules back to plain dicts for settings.json."""
    return [
        {
            "key": r.key,
            "style": r.style,
            "interval": r.interval,
            "trigger": r.trigger,
            "hp_percent": r.hp_percent,
            "enabled": r.enabled,
        }
        for r in rules
    ]


# The three abilities that were hardcoded before rules existed, and the
# legacy settings.json keys that configured them. Migration exists so an
# existing customer's setup survives the upgrade unchanged.
_LEGACY_ROWS = (
    ("rc_sw", "rc_cd", "e", "tap", 10.0, "in_combat"),
    ("k3_sw", "k3_cd", "3", "double", 50.0, "enemy_visible"),
    ("mv_sw", "mv_cd", "r", "aim", 3.0, "while_moving"),
)


def default_rules():
    """The three shipped abilities, for a settings file that has neither
    a rule list nor the legacy keys."""
    return [
        SkillRule(key=key, style=style, interval=interval, trigger=trigger)
        for _sw, _cd, key, style, interval, trigger in _LEGACY_ROWS
    ]


def _migrate_legacy(data):
    rules = []
    for sw_key, cd_key, key, style, interval, trigger in _LEGACY_ROWS:
        try:
            cd = max(MIN_INTERVAL, float(data.get(cd_key, interval)))
        except (TypeError, ValueError):
            cd = interval
        rules.append(SkillRule(key=key, style=style, interval=cd,
                               trigger=trigger,
                               enabled=bool(data.get(sw_key, True))))
    return rules


def rules_from_settings(data):
    """Rules for a loaded settings.json, migrating the old flat keys.

    Precedence matters: a saved `skills` list always wins, including an empty
    one -- deleting every row has to stay deleted, not silently resurrect the
    three defaults on next launch.
    """
    if not isinstance(data, dict):
        return default_rules()
    if "skills" in data:
        return parse_rules(data["skills"])
    if any(sw in data or cd in data for sw, cd, *_ in _LEGACY_ROWS):
        return _migrate_legacy(data)
    return default_rules()
