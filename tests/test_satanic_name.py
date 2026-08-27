"""Matching an OCR'd HUD line to a known zone name.

The HUD's magenta line names the current satanic zone. OCR of a stylised game
font is imperfect in two specific ways seen on real frames:
  - the leading skull glyph and "Satanic" label come back mangled
    ("Satanie", "Satanice") or merged into the name;
  - the read is sometimes PARTIAL ("Dungeon" for "Prison Dungeon") or carries
    the following HUD lines along with it ("Forgotten Caves Hell (2) ...").
A plain similarity ratio handles neither, so the matcher also accepts a
word-aligned containment that resolves to exactly one zone.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from satanic_name import has_satanic_label, match_zone_name

ZONES = [
    "Outskirts of Inoya", "Fields of Battle", "The Pumpkin Patch",
    "Woodhill Plains", "King's Garden",
    "Crystal Village", "Chilling Lake", "Arctic Tundra",
    "Snowy Mountains", "The Glacial Trail",
    "Prison Dungeon", "The Depths of Hell", "Steam Train",
    "Forgotten Caves", "Camp of Souls", "Flooded Plains",
]


def test_exact_name_matches():
    name, ratio = match_zone_name("The Glacial Trail", ZONES)
    assert name == "The Glacial Trail"
    assert ratio == 1.0


def test_skull_and_satanic_prefix_are_stripped():
    name, _ = match_zone_name("\u2620Satanic The Glacial Trail", ZONES)
    assert name == "The Glacial Trail"


def test_ocr_typo_in_the_satanic_label_is_tolerated():
    # Real reads: "Satanie The Glacial Trail", "Satanice The Glacial Trail".
    for raw in (" Satanie The Glacial Trail", " Satanice The Glacial Trail"):
        name, _ = match_zone_name(raw, ZONES)
        assert name == "The Glacial Trail", raw


def test_partial_read_resolving_to_one_zone_is_accepted():
    # Observed on real frames: OCR returned only "Dungeon".
    name, _ = match_zone_name("Dungeon", ZONES)
    assert name == "Prison Dungeon"


def test_partial_read_matching_several_zones_is_rejected():
    # "Plains" is in both Woodhill Plains and Flooded Plains -> ambiguous.
    name, _ = match_zone_name("Plains", ZONES)
    assert name is None


def test_trailing_hud_lines_are_ignored():
    raw = "Forgotten Caves Hell (2) Zone Level 425"
    name, _ = match_zone_name(raw, ZONES)
    assert name == "Forgotten Caves"


def test_garbage_returns_none():
    for raw in ("", "   ", "11111:1:4 5", "xyzzy qwerty"):
        assert match_zone_name(raw, ZONES)[0] is None, raw


def test_none_input_returns_none():
    assert match_zone_name(None, ZONES)[0] is None


def test_matching_is_case_insensitive():
    name, _ = match_zone_name("the glacial TRAIL", ZONES)
    assert name == "The Glacial Trail"


def test_apostrophe_dropped_by_ocr_still_matches():
    name, _ = match_zone_name("Kings Garden", ZONES)
    assert name == "King's Garden"


def test_empty_candidate_list_returns_none():
    assert match_zone_name("The Glacial Trail", [])[0] is None


def test_below_threshold_similarity_is_rejected():
    # A real zone-shaped string that is not any known zone.
    name, _ = match_zone_name("Volcanic Island", ZONES)
    assert name is None


# ── the "Satanic" label, used when the line is not magenta ──────────────────

def test_label_detected_on_the_cream_in_zone_line():
    # Live 2026-08-24; OCR renders "Satanic" as "Satanio" at this size.
    assert has_satanic_label("Satanio The Highland Mines")
    assert has_satanic_label("Satanic The Highland Mines")
    assert has_satanic_label("☠Satanic The Glacial Trail")


def test_label_not_claimed_for_an_ordinary_zone_line():
    """The line above the satanic one is the zone the player is standing in.
    Treating it as labelled would target the wrong zone."""
    for text in ("Dawn's Chapel", "The Highland Mines", "Nightmare",
                 "Zone Level 210", "Hell (2)", "", None):
        assert not has_satanic_label(text), text


def test_label_not_claimed_for_the_magenta_skull_line():
    """That variant carries no "Satanic" word - colour is its only signal."""
    assert not has_satanic_label("☠Prison Dungeon")


# ── never resolve a truncated read to the WRONG zone ─────────────────────────
# Live 2026-08-27 the HUD OCR degraded to "Satanic Dry" (sim 0.50) and
# "SatanicFuii Coast" (0.67). Both landed on the right zone via the
# containment rule, but a low ratio reaching a confident answer is the shape
# of a bug that resets into the wrong zone -- which is exactly what the user
# reported seeing. It turns out the matcher is already safe: every ambiguous
# fragment refuses instead of guessing. These pin that, because the safety
# comes from the "exactly one hit" rule and a well-meaning edit could drop it.

def test_degraded_live_reads_resolve_to_the_right_zone():
    import config
    zones = list(config.ZONE_NODES_WORLD.keys())
    for text, expected in (
        ("Satanic TheDevil'sBreach", "The Devil's Breach"),
        ("Satanic Dry", "Dry Hills"),
        ("SatanicFuii Coast", "Fuji Coast"),
        ("Satanic Steam Train", "Steam Train"),
    ):
        got, _ratio = match_zone_name(text, zones)
        assert got == expected, f"{text!r} -> {got!r}"


def test_a_shared_first_word_refuses_rather_than_guessing():
    """"The", "Corrupted", "Abyss" and "Pyramid" each begin several zones. A
    read truncated to one of them must return None and let the colour scan
    decide -- picking one at random is how the bot farms the wrong zone."""
    import config
    zones = list(config.ZONE_NODES_WORLD.keys())
    for fragment in ("Satanic The", "Satanic Corrupted", "Satanic Abyss",
                     "Satanic Pyramid"):
        got, ratio = match_zone_name(fragment, zones)
        assert got is None, f"{fragment!r} confidently picked {got!r} ({ratio:.2f})"


def test_no_single_word_fragment_ever_picks_a_different_zone():
    """Exhaustive over the real table: truncating any zone to its first word
    must yield that zone or nothing -- never a different one."""
    import config
    zones = list(config.ZONE_NODES_WORLD.keys())
    for zone in zones:
        got, ratio = match_zone_name("Satanic " + zone.split()[0], zones)
        assert got in (zone, None), \
            f"{zone!r} truncated resolved to {got!r} ({ratio:.2f})"
