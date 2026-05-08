"""Tests for input_control.py pure-logic helpers (no real keyboard input)."""

import os
import sys
from unittest.mock import call, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import input_control


# ── _direction_keys ───────────────────────────────────────────────────────


def test_direction_keys_north():
    keys = input_control._direction_keys(0, -200)
    assert keys == ['w']


def test_direction_keys_south():
    keys = input_control._direction_keys(0, 200)
    assert keys == ['s']


def test_direction_keys_east():
    keys = input_control._direction_keys(200, 0)
    assert keys == ['d']


def test_direction_keys_west():
    keys = input_control._direction_keys(-200, 0)
    assert keys == ['a']


def test_direction_keys_diagonal_northeast():
    keys = input_control._direction_keys(150, -150)
    # Both axes outside the deadzone — both keys present.
    assert set(keys) == {'w', 'd'}


def test_direction_keys_zero_falls_back_to_east():
    # Inside-deadzone vectors must still yield at least one key.
    keys = input_control._direction_keys(0, 0)
    assert keys == ['d']


# ── explore_move_start / explore_move_stop ────────────────────────────────


def test_explore_move_start_presses_only_new_keys():
    with patch("input_control.keyboard") as mock_kb:
        # Already holding 'w'. Move east → should release 'w' (no longer needed
        # if no vertical component) and press 'd'.
        held = frozenset({'w'})
        new = input_control.explore_move_start(200, 0, held_keys=held)

    assert new == frozenset({'d'})
    mock_kb.release.assert_called_once_with('w')
    mock_kb.press.assert_called_once_with('d')


def test_explore_move_start_no_op_when_direction_unchanged():
    with patch("input_control.keyboard") as mock_kb:
        held = frozenset({'d'})
        new = input_control.explore_move_start(200, 0, held_keys=held)

    # Same direction — no keyboard activity at all.
    assert new == held
    assert mock_kb.press.call_count == 0
    assert mock_kb.release.call_count == 0


def test_explore_move_start_diagonal_presses_both():
    with patch("input_control.keyboard") as mock_kb:
        new = input_control.explore_move_start(150, -150, held_keys=frozenset())

    assert new == frozenset({'w', 'd'})
    pressed = {c.args[0] for c in mock_kb.press.call_args_list}
    assert pressed == {'w', 'd'}


def test_explore_move_stop_releases_all_held_keys():
    with patch("input_control.keyboard") as mock_kb:
        held = frozenset({'w', 'a'})
        out = input_control.explore_move_stop(held_keys=held)

    assert out == frozenset()
    released = {c.args[0] for c in mock_kb.release.call_args_list}
    assert released == {'w', 'a'}


def test_explore_move_stop_no_op_for_empty_set():
    with patch("input_control.keyboard") as mock_kb:
        out = input_control.explore_move_stop(held_keys=frozenset())
    assert out == frozenset()
    assert mock_kb.release.call_count == 0


# ── backtrack_move cooldown logic ─────────────────────────────────────────


def test_backtrack_move_skips_when_within_cooldown():
    """A second call within STUCK_BACKTRACK_COOLDOWN must not invoke wasd_move."""
    last_t = [9999999999.0]  # far-future fake "last call"
    with patch("input_control.wasd_move") as mock_wasd, \
         patch("input_control.time.time", return_value=last_t[0] + 0.01):
        input_control.backtrack_move(100, 0, last_t)
    assert mock_wasd.call_count == 0


def test_backtrack_move_fires_when_cooldown_elapsed():
    """When enough time has passed since last_backtrack_t[0], reverse direction."""
    from config import STUCK_BACKTRACK_COOLDOWN
    last_t = [1000.0]
    with patch("input_control.wasd_move") as mock_wasd, \
         patch("input_control.time.time", return_value=last_t[0] + STUCK_BACKTRACK_COOLDOWN + 1.0), \
         patch("input_control.random.uniform", return_value=0.4):
        input_control.backtrack_move(100, 50, last_t)

    mock_wasd.assert_called_once()
    args, _ = mock_wasd.call_args
    # Direction is reversed: -dx, -dy
    assert args[0] == -100
    assert args[1] == -50


def test_backtrack_move_updates_last_t_on_fire():
    from config import STUCK_BACKTRACK_COOLDOWN
    last_t = [1000.0]
    fire_t = last_t[0] + STUCK_BACKTRACK_COOLDOWN + 1.0
    with patch("input_control.wasd_move"), \
         patch("input_control.time.time", return_value=fire_t), \
         patch("input_control.random.uniform", return_value=0.4):
        input_control.backtrack_move(100, 0, last_t)

    assert last_t[0] == fire_t
