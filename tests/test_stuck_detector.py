import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from state import StuckDetector


def test_not_stuck_until_window_filled():
    sd = StuckDetector(window=5, min_move=8)
    for i in range(4):
        sd.update(0, 0)
    assert sd.is_stuck() is False  # window not yet full


def test_stuck_when_position_constant():
    sd = StuckDetector(window=5, min_move=8)
    for _ in range(5):
        sd.update(10, 10)
    assert sd.is_stuck() is True


def test_not_stuck_when_moving():
    sd = StuckDetector(window=5, min_move=8)
    # Move 10 px each tick — far above min_move=8 over the window.
    for i in range(5):
        sd.update(i * 10, 0)
    assert sd.is_stuck() is False


def test_stuck_when_movement_below_threshold():
    sd = StuckDetector(window=5, min_move=20)
    # Total span across window = 4 px in x + 0 in y = 4 < 20.
    for i in range(5):
        sd.update(i, 0)
    assert sd.is_stuck() is True


def test_window_evicts_old_positions():
    sd = StuckDetector(window=3, min_move=5)
    sd.update(0, 0)
    sd.update(0, 0)
    sd.update(0, 0)
    assert sd.is_stuck() is True
    # New non-stuck samples push out the old stationary ones
    sd.update(50, 50)
    sd.update(100, 100)
    sd.update(150, 150)
    assert sd.is_stuck() is False


def test_reset_clears_history():
    sd = StuckDetector(window=3, min_move=5)
    for _ in range(3):
        sd.update(0, 0)
    assert sd.is_stuck() is True
    sd.reset()
    assert sd.positions == []
    assert sd.is_stuck() is False
