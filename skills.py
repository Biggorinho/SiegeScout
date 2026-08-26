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


# Firing the buff within this long after a jump wastes the cast. Carried over
# verbatim from main.run's inline guard.
POST_JUMP_DELAY = 0.42


@dataclass(frozen=True)
class LoopState:
    """What the frame loop knows, at the moment skills are considered."""
    now: float
    in_combat: bool
    enemy_visible: bool
    hp: "float | None"          # None when the HP bar was unreadable
    zone_entered: bool
    moving: bool
    aim_dx: float
    aim_dy: float
    busy_until: float
    last_jump_t: float


def _trigger_holds(rule, state):
    if rule.trigger == "always":
        return True
    if rule.trigger == "in_combat":
        return state.in_combat or state.enemy_visible
    if rule.trigger == "enemy_visible":
        return state.enemy_visible
    if rule.trigger == "hp_below":
        # None means the HP bar was unreadable this frame. Guessing here
        # fires a panic skill at full health.
        return state.hp is not None and (state.hp * 100.0) < rule.hp_percent
    if rule.trigger == "zone_entry":
        return state.zone_entered
    if rule.trigger == "while_moving":
        return state.moving
    return False


class SkillRunner:
    """Owns the per-rule interval timers and answers what should fire now.

    Stateful on purpose: these timers used to be three nonlocals in
    main.run's loop, and keeping them next to the rules is what let those
    disappear. `due()` advances the timer for every rule it returns, matching
    the old behaviour where the timer moved on the attempt rather than on a
    confirmed cast.
    """

    def __init__(self, rules, post_jump_delay=POST_JUMP_DELAY):
        self._post_jump_delay = post_jump_delay
        self.set_rules(rules)

    def set_rules(self, rules):
        self._rules = list(rules)
        self._next_t = {}

    def due(self, state):
        """Rules that should fire this tick, in list order.

        A rule blocked by a guard is NOT marked as fired -- being suppressed
        is not the same as having cast, and burning the interval on a
        swallowed cast is how a skill silently stops going off.
        """
        if state.now < state.busy_until:
            return []
        firing = []
        for index, rule in enumerate(self._rules):
            if not rule.enabled:
                continue
            if state.now < self._next_t.get(index, 0.0):
                continue
            if not _trigger_holds(rule, state):
                continue
            if rule.style == "tap" and \
                    (state.now - state.last_jump_t) < self._post_jump_delay:
                continue
            self._next_t[index] = state.now + rule.interval
            firing.append(rule)
        return firing

    # Only proactive combat skills are deferred. A reactive rule must not be
    # pushed out at the moment it becomes relevant (hp_below), a once-per-zone
    # rule must not be swallowed by a reset (zone_entry), and the movement
    # ability was never touched by these sites before rules existed
    # (while_moving).
    _DEFERRABLE_TRIGGERS = ("always", "in_combat", "enemy_visible")

    def defer_all(self, now):
        """Push each proactive combat rule's next fire out by its own interval.

        Entering combat, and starting a zone, used to reset the two ability
        timers to now+cooldown. That delay is deliberate: without it every
        configured skill fires at once the moment the state changes, which
        wastes the casts the opening sequence already made.

        It applies only to _DEFERRABLE_TRIGGERS. Deferring everything starved
        long-interval user rules on frequent attack entries, pushed an
        hp_below panic rule out at the exact moment combat made it relevant,
        and delayed the movement ability that the old reset sites left alone.
        """
        for index, rule in enumerate(self._rules):
            if rule.trigger not in self._DEFERRABLE_TRIGGERS:
                continue
            self._next_t[index] = now + rule.interval
