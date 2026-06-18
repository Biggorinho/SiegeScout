"""Detect whether the player is physically moving by measuring how far the
on-screen world scrolls between frames.

The minimap tells us where things are, but not whether the character is actually
moving — when the bot walks into a wall it can keep issuing move keys while the
world stays put. This detector measures the *camera shift* between consecutive
centre-screen crops via phase correlation. Enemies, spell effects, and the
player's own beam cause local pixel changes but not a coherent global
translation, so phase correlation isolates real camera/player movement.
"""

from collections import deque
import math

import cv2
import numpy as np


class ScreenMotionDetector:
    def __init__(
        self,
        roi_frac=(0.28, 0.30, 0.72, 0.68),  # (x0, y0, x1, y1) — central world, no HUD
        downscale_w=256,
        window=6,
        static_shift_px=1.2,
    ):
        self._roi_frac = roi_frac
        self._downscale_w = int(downscale_w)
        self._window = max(1, int(window))
        self._static_shift_px = float(static_shift_px)
        self._prev = None
        self._hann = None
        self._history = deque(maxlen=self._window)

    def reset(self):
        """Forget history (call on zone change / teleport — the scene jumps)."""
        self._prev = None
        self._hann = None
        self._history.clear()

    def _prep(self, frame_bgr):
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return None
        h, w = frame_bgr.shape[:2]
        x0 = int(w * self._roi_frac[0])
        y0 = int(h * self._roi_frac[1])
        x1 = int(w * self._roi_frac[2])
        y1 = int(h * self._roi_frac[3])
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self._downscale_w and gray.shape[1] > self._downscale_w:
            scale = self._downscale_w / float(gray.shape[1])
            new_h = max(1, int(round(gray.shape[0] * scale)))
            gray = cv2.resize(gray, (self._downscale_w, new_h), interpolation=cv2.INTER_AREA)
        return np.float32(gray)

    def update(self, frame_bgr):
        """Feed the current frame; return the camera-shift magnitude (in downscaled
        px) since the previous frame. ~0 means the world didn't scroll."""
        cur = self._prep(frame_bgr)
        if cur is None:
            return 0.0
        if self._prev is None or self._prev.shape != cur.shape:
            self._prev = cur
            self._hann = cv2.createHanningWindow((cur.shape[1], cur.shape[0]), cv2.CV_32F)
            return 0.0
        (dx, dy), _response = cv2.phaseCorrelate(self._prev, cur, self._hann)
        self._prev = cur
        mag = math.hypot(dx, dy)
        self._history.append(mag)
        return mag

    def recent_shift(self):
        """Mean camera shift over the recent window."""
        return (sum(self._history) / len(self._history)) if self._history else 0.0

    def is_static(self):
        """True once the window is full and the average camera shift is below the
        'not moving' threshold — i.e. the world isn't scrolling."""
        return len(self._history) >= self._window and self.recent_shift() < self._static_shift_px
