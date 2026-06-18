"""Tests for ScreenMotionDetector — phase-correlation camera-shift detection."""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from screen_motion import ScreenMotionDetector


def _texture(h=500, w=800, seed=0):
    """A smooth, gradient-rich noise image so phase correlation has features."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 2.0)


def test_returns_zero_on_first_frame():
    det = ScreenMotionDetector()
    assert det.update(_texture()) == 0.0


def test_static_scene_is_detected_as_static():
    det = ScreenMotionDetector(window=5, static_shift_px=1.2)
    frame = _texture()[50:450, 100:700].copy()
    det.update(frame)
    for _ in range(6):
        det.update(frame.copy())  # identical frame = no scroll
    assert det.recent_shift() < 1.2
    assert det.is_static() is True


def test_panning_scene_is_not_static():
    det = ScreenMotionDetector(window=5, static_shift_px=1.2)
    big = _texture(520, 860, seed=3)
    # Slide a fixed-size window across the texture: simulates the camera panning.
    for i in range(8):
        ox = 100 + 4 * i
        frame = big[60:460, ox:ox + 600].copy()
        det.update(frame)
    assert det.recent_shift() > 1.2, f"pan not detected: {det.recent_shift()}"
    assert det.is_static() is False


def test_reset_clears_history():
    det = ScreenMotionDetector(window=3, static_shift_px=1.2)
    frame = _texture()[50:450, 100:700].copy()
    for _ in range(5):
        det.update(frame.copy())
    assert det.is_static() is True
    det.reset()
    assert det.is_static() is False  # window empty again
    assert det.recent_shift() == 0.0


def test_handles_none_and_empty():
    det = ScreenMotionDetector()
    assert det.update(None) == 0.0
    assert det.update(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0
