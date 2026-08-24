"""Resolve the satanic zone name from the HUD's magenta line.

The game already tells us which zone is satanic -- it is printed top-right,
in magenta, with a skull glyph, above the difficulty and zone-level lines.
Reading it turns zone selection into a coordinate lookup and removes the need
to hunt for a red-glowing node by colour, which is the part that fails on the
sepia map region (its parchment is itself red-hued).

Import-light on purpose, like fusion.py and targeting.py: pure string work,
no cv2 and no OCR engine. The caller supplies both the OCR text and the list
of candidate zone names, so the matching rules are unit-testable on their own.
"""

import difflib
import re

# Leading skull glyph plus the "Satanic" label, which OCR mangles into
# "Satanie" / "Satanice" depending on scale. No real zone name starts with
# these letters, so consuming a "satan..." token outright is safe.
_LABEL = re.compile(r"^\W*satan\w*\s*:?\s*", re.IGNORECASE)
_PUNCT = re.compile(r"[^a-z0-9 ]")

# Similarity above which a whole-string fuzzy match is trusted on its own.
_MIN_RATIO = 0.8


def _normalize(text):
    """Casefold, drop the satanic label, drop punctuation, squeeze spaces.

    Punctuation removal is what lets an OCR read of "Kings Garden" match the
    real "King's Garden" -- the apostrophe is routinely lost.
    """
    s = _LABEL.sub("", str(text).casefold())
    s = _PUNCT.sub("", s)
    return " ".join(s.split())


def _contains_words(haystack, needle):
    """True when `needle` appears in `haystack` on word boundaries."""
    if not haystack or not needle:
        return False
    return re.search(r"(?:^| )" + re.escape(needle) + r"(?:$| )",
                     haystack) is not None


def match_zone_name(text, candidates, min_ratio=_MIN_RATIO):
    """Best zone name for an OCR'd HUD line, as ``(name, ratio)``.

    Returns ``(None, 0.0)`` when nothing matches confidently. Two rules apply,
    in order:

    1. Whole-string similarity at or above `min_ratio`.
    2. Word-aligned containment in either direction, but only when it picks
       out exactly one zone. This is what rescues a partial read ("Dungeon"
       -> "Prison Dungeon") and a read that swallowed the following HUD lines
       ("Forgotten Caves Hell (2) ..."), while still rejecting an ambiguous
       fragment like "Plains", which fits two zones.
    """
    if not text or not candidates:
        return None, 0.0
    query = _normalize(text)
    if not query:
        return None, 0.0

    normalized = [(c, _normalize(c)) for c in candidates]

    best, best_ratio = None, 0.0
    for cand, norm in normalized:
        ratio = difflib.SequenceMatcher(None, query, norm).ratio()
        if ratio > best_ratio:
            best, best_ratio = cand, ratio
    if best_ratio >= min_ratio:
        return best, best_ratio

    hits = [(c, n) for c, n in normalized
            if _contains_words(n, query) or _contains_words(query, n)]
    if len(hits) == 1:
        cand, norm = hits[0]
        return cand, max(best_ratio,
                         difflib.SequenceMatcher(None, query, norm).ratio())
    return None, 0.0
