"""The Vote Reset button is the 2nd of the pause menu's 4-button stack
(Resume / Vote Reset / Options / Save & Exit). Selection must track the stack
position, not a fixed Y, so it still hits Vote Reset when the menu renders lower
in combat zones (the old fixed-Y click landed on Options)."""
import os
import sys
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import combat

_PAUSE = Path(__file__).resolve().parent / "fixtures" / "pause_menu.png"


def _rows():
    if not _PAUSE.exists():
        pytest.skip("pause_menu fixture missing")
    img = cv2.imread(str(_PAUSE))
    assert img is not None
    return combat._find_pause_menu_button_rows(img)


def test_select_vote_reset_picks_second_button_ignoring_target_y():
    rows = _rows()
    # Deliberately wrong target_y — must still pick the 2nd stack row (~248 here).
    sel = combat._select_vote_reset_row(rows, 999)
    assert sel is not None
    assert 235 <= sel[0] <= 262, f"expected Vote Reset (~248), got {sel[0]}"


def test_select_vote_reset_is_position_independent():
    # Simulate the menu rendering 120px lower (combat zone). Vote Reset must move
    # with it, not stay at the old fixed Y.
    rows = [(r[0] + 120, r[1], r[2]) for r in _rows()]
    sel = combat._select_vote_reset_row(rows, 485)
    assert sel is not None
    assert 355 <= sel[0] <= 382, f"expected shifted Vote Reset (~368), got {sel[0]}"


def test_select_vote_reset_returns_none_without_full_stack():
    """SAFETY: with <4 detected rows we can't tell which is Vote Reset, so it must
    return None (caller uses the safe fixed Y) — never guess a nearest-target row,
    which previously returned y=644 = Save & Exit and closed the game."""
    rows = [(419, 5000, 259), (644, 5000, 259)]  # only 2 rows, e.g. Resume + Save&Exit
    assert combat._select_vote_reset_row(rows, 485) is None


_COMBAT = Path(__file__).resolve().parent / "fixtures" / "pause_menu_combat.png"


def test_full_res_paired_bars_merge_to_four_buttons():
    """At 1080p each button renders as 2 red bars (8 rows). They must merge to 4
    buttons and Vote Reset (2nd) must resolve to its real position (~491), not be
    confused by the doubled bars (which used to land the click on Options)."""
    if not _COMBAT.exists():
        pytest.skip("combat pause-menu fixture missing")
    img = cv2.imread(str(_COMBAT))
    assert img is not None
    rows = combat._find_pause_menu_button_rows(img)
    assert len(rows) == 4, f"expected 4 merged buttons, got {[r[0] for r in rows]}"
    sel = combat._select_vote_reset_row(rows, 485)
    assert sel is not None and 470 <= sel[0] <= 512, f"Vote Reset mis-located: {sel}"
