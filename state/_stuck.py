"""Stuck detection: simple position-history check + minimap-motion-vs-intent check."""

import math

import cv2
import numpy as np


class StuckDetector:
    def __init__(self, window=5, min_move=8):
        self.positions = []
        self.window    = window
        self.min_move  = min_move

    def reset(self):
        self.positions.clear()

    def update(self, x, y):
        self.positions.append((x, y))
        if len(self.positions) > self.window:
            self.positions.pop(0)

    def is_stuck(self):
        if len(self.positions) < self.window:
            return False
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        return (max(xs) - min(xs) + max(ys) - min(ys)) < self.min_move


class IntentStuckDetector:
    """Detects stuck state when movement is commanded but minimap barely moves."""

    def __init__(self, window=12, intent_min=85.0, min_motion_px=0.55, stuck_ratio=0.70):
        self.window = max(3, int(window))
        self.intent_min = float(intent_min)
        self.min_motion_px = float(min_motion_px)
        self.stuck_ratio = float(stuck_ratio)
        self._prev_gray = None
        self._flags = []

    def reset(self):
        self._prev_gray = None
        self._flags = []

    def update(self, minimap_bgr, move_dx, move_dy):
        if minimap_bgr is None or minimap_bgr.size == 0:
            return
        gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return

        try:
            (sx, sy), response = cv2.phaseCorrelate(
                self._prev_gray.astype(np.float32),
                gray.astype(np.float32),
            )
            motion = math.hypot(float(sx), float(sy))
            if response < 0.01:
                motion = 0.0
        except cv2.error:
            motion = 0.0

        intent = math.hypot(float(move_dx), float(move_dy))
        stuck_like = (intent >= self.intent_min) and (motion < self.min_motion_px)
        self._flags.append(1 if stuck_like else 0)
        if len(self._flags) > self.window:
            self._flags.pop(0)

        self._prev_gray = gray

    def is_stuck(self):
        if len(self._flags) < self.window:
            return False
        stuck_frames = sum(self._flags)
        return stuck_frames >= int(math.ceil(self.window * self.stuck_ratio))
