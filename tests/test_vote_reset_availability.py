"""Vote Reset has an in-game cooldown: the button then renders a dark
disabled skin (red coverage ~0.05 vs ~0.66 active) and the game silently
ignores clicks on it. Clicking anyway burned the fail streak into 75s
lockouts, so the click path must detect the disabled skin, report
"unavailable", and back off. Fixtures are real frames from the 2026-07-11
failure run. A click on an ACTIVE button can also be lost (hover-focus
race), so the click verifies the menu reacted and retries once."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import combat

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_VOTE_Y_540P = 248  # Vote Reset row centre in the 960x540 fixtures


def _fixture(name):
    p = _FIXTURES / name
    if not p.exists():
        pytest.skip(f"missing fixture: {name}")
    img = cv2.imread(str(p))
    assert img is not None
    return img


def test_active_button_reads_active():
    assert combat._vote_reset_row_is_active(
        _fixture("votereset_active.png"), _VOTE_Y_540P) is True


def test_disabled_button_reads_inactive():
    assert combat._vote_reset_row_is_active(
        _fixture("votereset_disabled.png"), _VOTE_Y_540P) is False


class _Clicks:
    def __init__(self):
        self.points = []

    def __call__(self, x, y, hold_time=0.05, settle_time=0.0):
        self.points.append((int(x), int(y)))


# A clean 4-button stack whose 2nd row sits on the fixtures' Vote Reset bar.
_STACK_ROWS = [(178, 500, 230), (248, 500, 230), (318, 500, 230), (388, 500, 230)]


def _wire(monkeypatch, frames, clicks):
    seq = iter(frames)
    last = frames[-1]
    monkeypatch.setattr(combat, "grab_frame_bgr", lambda sct, mon: next(seq, last))
    monkeypatch.setattr(combat, "_find_pause_menu_button_rows",
                        lambda f: list(_STACK_ROWS))
    monkeypatch.setattr(combat, "left_click_at", clicks)
    monkeypatch.setattr(combat, "sleep_interruptible", lambda s: True)


def test_click_skipped_when_button_disabled(monkeypatch):
    clicks = _Clicks()
    _wire(monkeypatch, [_fixture("votereset_disabled.png")], clicks)
    result = combat._click_vote_reset_menu_button(None, None, 2)
    assert result == "unavailable"
    assert clicks.points == []


def test_click_returns_true_when_menu_closes(monkeypatch):
    active = _fixture("votereset_active.png")
    gone = np.zeros_like(active)
    clicks = _Clicks()
    _wire(monkeypatch, [active, gone, gone], clicks)
    result = combat._click_vote_reset_menu_button(None, None, 2)
    assert result is True
    assert len(clicks.points) == 1


def test_click_retries_once_when_menu_unchanged(monkeypatch):
    active = _fixture("votereset_active.png")
    clicks = _Clicks()
    _wire(monkeypatch, [active, active, active], clicks)
    result = combat._click_vote_reset_menu_button(None, None, 2)
    assert result is True
    assert len(clicks.points) == 2, "lost click must be retried once"


def test_sequence_reports_unavailable_and_closes_menu(monkeypatch):
    keys = []
    monkeypatch.setattr(combat, "_ensure_pause_menu_open", lambda s, m: True)
    monkeypatch.setattr(combat, "_click_vote_reset_menu_button",
                        lambda s, m, a: "unavailable")
    monkeypatch.setattr(combat, "tap_key",
                        lambda k, hold_time=0.05: keys.append(k))
    monkeypatch.setattr(combat, "sleep_interruptible", lambda s: True)
    monkeypatch.setattr(combat, "release_all_inputs", lambda: None)
    result = combat.run_vote_reset_sequence(None, None, target_zone="X")
    assert result == "unavailable"
    assert "esc" in keys, "menu must be closed, not left open over gameplay"
