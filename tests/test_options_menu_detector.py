import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detectors import is_options_menu_visible, is_pause_menu_visible


def _make_frame(width=1920, height=1080):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _draw_red_rect(frame, x, y, w, h):
    # BGR red — same hue band the detector matches
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 220), thickness=-1)


def test_options_menu_detected_with_six_tabs():
    frame = _make_frame()
    # Six tabs roughly at y ~= 60 px (~5.5% of 1080) spread across the top.
    tab_w, tab_h = 180, 50
    y = 60
    xs = [180, 430, 680, 930, 1180, 1430]
    for x in xs:
        _draw_red_rect(frame, x, y, tab_w, tab_h)

    assert is_options_menu_visible(frame) is True


def test_options_menu_detected_with_four_tabs():
    """4-of-6 threshold tolerates partial occlusion / styling differences."""
    frame = _make_frame()
    tab_w, tab_h = 180, 50
    y = 60
    xs = [180, 430, 680, 930]  # four tabs
    for x in xs:
        _draw_red_rect(frame, x, y, tab_w, tab_h)

    assert is_options_menu_visible(frame) is True


def test_options_menu_rejected_with_three_tabs():
    frame = _make_frame()
    tab_w, tab_h = 180, 50
    y = 60
    xs = [180, 430, 680]  # only three tabs
    for x in xs:
        _draw_red_rect(frame, x, y, tab_w, tab_h)

    assert is_options_menu_visible(frame) is False


def test_options_menu_rejected_for_pause_menu():
    """Vertically stacked centre buttons must not trigger the Options detector."""
    frame = _make_frame()
    btn_w, btn_h = 320, 60
    cx = 1920 // 2 - btn_w // 2
    ys = [380, 470, 560, 650]  # four stacked centred buttons
    for y in ys:
        _draw_red_rect(frame, cx, y, btn_w, btn_h)

    assert is_options_menu_visible(frame) is False
    # Pause-menu detector must still recognise the same frame.
    assert is_pause_menu_visible(frame) is True


def test_options_menu_rejected_for_empty_frame():
    frame = _make_frame()
    assert is_options_menu_visible(frame) is False


def test_options_menu_rejected_for_none_frame():
    assert is_options_menu_visible(None) is False
