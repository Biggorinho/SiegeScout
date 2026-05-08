import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from state import IntentStuckDetector


def _grid_image(size=96):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(0, size, 8):
        img[y:y + 1, :, :] = 180
    for x in range(0, size, 8):
        img[:, x:x + 1, :] = 180
    return img


def test_intent_stuck_detector_triggers_when_intent_high_but_scene_static():
    det = IntentStuckDetector(window=6, intent_min=80.0, min_motion_px=0.6, stuck_ratio=0.7)
    frame = _grid_image()

    det.update(frame, 120, 0)
    for _ in range(8):
        det.update(frame, 120, 0)

    assert det.is_stuck()


def test_intent_stuck_detector_not_stuck_when_scene_moves_with_intent():
    det = IntentStuckDetector(window=6, intent_min=80.0, min_motion_px=0.6, stuck_ratio=0.7)
    base = _grid_image()

    det.update(base, 120, 0)
    for i in range(1, 9):
        shifted = np.roll(base, shift=i, axis=1)
        det.update(shifted, 120, 0)

    assert not det.is_stuck()
