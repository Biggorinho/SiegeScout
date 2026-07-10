"""Interactive labeling tool for captured frame pairs.

Usage: python -m eval.label_tool --session eval/captures/session_<stamp>

Windows: "minimap" (zoomed 3x — click points, brush terrain corrections)
and, when the pair has a full frame, "frame" (drag boxes).

Keys:
  1/2/3/4  mode: point / box / stroke-blocked / stroke-open
  c        cycle class (point + box modes)
  u        undo last addition
  n / p    next / previous pair (autosaves)
  q or ESC quit (autosaves)

In the stroke modes the current heuristic wall mask is overlaid so you can
see what you are correcting: brush stroke-blocked (key 3) where the mask
wrongly blocks walkable ground, stroke-open (key 4) where it wrongly opens
a wall.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from eval.labels import (
    MINIMAP_CLASSES, SCREEN_CLASSES, TERRAIN_MODES,
    FrameLabels, discover_pairs, load_labels, save_labels,
)

MODES = ("point", "box", "stroke_blocked", "stroke_open")
_STROKE_BUCKET = {"stroke_blocked": "wrongly_blocked", "stroke_open": "wrongly_open"}
MINIMAP_SCALE = 3
FRAME_SCALE = 0.75
DEFAULT_STROKE_RADIUS = 4


class LabelSession:
    """Pure-logic labeling state machine; the cv2 UI is a thin shell over it."""

    def __init__(self, pairs):
        if not pairs:
            raise ValueError("no capture pairs to label")
        self._pairs = list(pairs)
        self._idx = 0
        self._labels_cache: dict[int, FrameLabels] = {}
        self._undo: dict[int, list] = {}
        self._dirty: set[int] = set()
        self._mode = "point"
        self._class_idx = {"point": 0, "box": 0}

    # ── navigation ────────────────────────────────────────────────────────
    @property
    def index(self) -> int:
        return self._idx

    @property
    def count(self) -> int:
        return len(self._pairs)

    @property
    def current_pair(self) -> dict:
        return self._pairs[self._idx]

    def next_pair(self) -> None:
        self._autosave()
        self._idx = min(self._idx + 1, len(self._pairs) - 1)

    def prev_pair(self) -> None:
        self._autosave()
        self._idx = max(self._idx - 1, 0)

    # ── labels access ─────────────────────────────────────────────────────
    @property
    def current_labels(self) -> FrameLabels:
        if self._idx not in self._labels_cache:
            pair = self.current_pair
            if os.path.exists(pair["labels"]):
                labels = load_labels(pair["labels"])
            else:
                labels = FrameLabels(
                    frame=os.path.basename(pair["frame"]) if pair["frame"] else "",
                    minimap=os.path.basename(pair["minimap"]),
                )
            self._labels_cache[self._idx] = labels
        return self._labels_cache[self._idx]

    # ── mode / class ──────────────────────────────────────────────────────
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode!r}")
        self._mode = mode

    @property
    def active_class(self) -> str:
        if self._mode == "box":
            return SCREEN_CLASSES[self._class_idx["box"]]
        return MINIMAP_CLASSES[self._class_idx["point"]]

    def cycle_class(self) -> None:
        if self._mode == "box":
            self._class_idx["box"] = (self._class_idx["box"] + 1) % len(SCREEN_CLASSES)
        else:
            self._class_idx["point"] = (self._class_idx["point"] + 1) % len(MINIMAP_CLASSES)

    # ── labeling actions ──────────────────────────────────────────────────
    def add_point(self, x: int, y: int) -> None:
        if self._mode != "point":
            return
        entry = {"x": int(x), "y": int(y), "cls": self.active_class}
        self.current_labels.minimap_points.append(entry)
        self._push_undo(("point", entry))

    def add_box(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self._mode != "box":
            return
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        if w < 2 or h < 2:
            return
        entry = {"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                 "cls": self.active_class}
        self.current_labels.screen_boxes.append(entry)
        self._push_undo(("box", entry))

    def add_stroke(self, x: int, y: int, r: int = DEFAULT_STROKE_RADIUS) -> None:
        bucket = _STROKE_BUCKET.get(self._mode)
        if bucket is None:
            return
        entry = {"x": int(x), "y": int(y), "r": int(r)}
        self.current_labels.terrain_strokes[bucket].append(entry)
        self._push_undo(("stroke", bucket, entry))

    def undo(self) -> None:
        stack = self._undo.get(self._idx, [])
        if not stack:
            return
        action = stack.pop()
        labels = self.current_labels
        if action[0] == "point":
            labels.minimap_points.remove(action[1])
        elif action[0] == "box":
            labels.screen_boxes.remove(action[1])
        else:
            labels.terrain_strokes[action[1]].remove(action[2])
        self._dirty.add(self._idx)

    # ── persistence ───────────────────────────────────────────────────────
    def save(self) -> None:
        save_labels(self.current_pair["labels"], self.current_labels)
        self._dirty.discard(self._idx)

    def _autosave(self) -> None:
        if self._idx in self._dirty:
            self.save()

    def _push_undo(self, action) -> None:
        self._undo.setdefault(self._idx, []).append(action)
        self._dirty.add(self._idx)


# ── cv2 UI shell (not unit-tested; logic lives in LabelSession) ────────────

_CLASS_COLORS = {
    "enemy": (0, 0, 255), "item": (0, 255, 255), "ore": (0, 165, 255),
    "portal": (255, 255, 0), "player": (0, 255, 0),
    "healthbar": (0, 0, 255), "teleport": (255, 255, 0),
}


def _render_minimap(session, minimap_bgr, wall_mask):
    img = minimap_bgr.copy()
    if session.mode in ("stroke_blocked", "stroke_open") and wall_mask is not None:
        # Blocked areas tinted red so corrections are visible.
        blocked = wall_mask == 0
        img[blocked] = (0.55 * img[blocked] + np.array([0, 0, 112])).astype(np.uint8)
    labels = session.current_labels
    for p in labels.minimap_points:
        cv2.drawMarker(img, (p["x"], p["y"]), _CLASS_COLORS.get(p["cls"], (255, 255, 255)),
                       cv2.MARKER_CROSS, 8, 1)
    for mode, color in (("wrongly_blocked", (0, 255, 0)), ("wrongly_open", (0, 0, 255))):
        for s in labels.terrain_strokes[mode]:
            cv2.circle(img, (s["x"], s["y"]), s["r"], color, 1)
    img = cv2.resize(img, None, fx=MINIMAP_SCALE, fy=MINIMAP_SCALE,
                     interpolation=cv2.INTER_NEAREST)
    header = (f"[{session.index + 1}/{session.count}] mode={session.mode} "
              f"cls={session.active_class}  1/2/3/4 mode  c class  u undo  "
              f"n/p nav  q quit")
    cv2.putText(img, header, (4, 16), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 220, 255), 1)
    return img


def _render_frame(session, frame_bgr, drag_start, drag_cur):
    img = frame_bgr.copy()
    for b in session.current_labels.screen_boxes:
        color = _CLASS_COLORS.get(b["cls"], (255, 255, 255))
        cv2.rectangle(img, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), color, 2)
    if drag_start is not None and drag_cur is not None:
        cv2.rectangle(img, drag_start, drag_cur, (255, 255, 255), 1)
    return cv2.resize(img, None, fx=FRAME_SCALE, fy=FRAME_SCALE,
                      interpolation=cv2.INTER_AREA)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Label captured frame pairs.")
    parser.add_argument("--session", required=True,
                        help="capture session dir (eval/captures/session_*)")
    args = parser.parse_args(argv)

    pairs = discover_pairs(args.session)
    if not pairs:
        print(f"no capture pairs found in {args.session}")
        return 1
    session = LabelSession(pairs)

    from eval.backends import HeuristicBackend
    backend = HeuristicBackend()

    drag = {"start": None, "cur": None}

    def on_minimap_mouse(event, x, y, flags, _param):
        mx, my = x // MINIMAP_SCALE, y // MINIMAP_SCALE
        if event == cv2.EVENT_LBUTTONDOWN:
            if session.mode == "point":
                session.add_point(mx, my)
            else:
                session.add_stroke(mx, my)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            if session.mode in ("stroke_blocked", "stroke_open"):
                session.add_stroke(mx, my)

    def on_frame_mouse(event, x, y, _flags, _param):
        fx, fy = int(x / FRAME_SCALE), int(y / FRAME_SCALE)
        if event == cv2.EVENT_LBUTTONDOWN:
            drag["start"] = (fx, fy)
            drag["cur"] = (fx, fy)
        elif event == cv2.EVENT_MOUSEMOVE and drag["start"] is not None:
            drag["cur"] = (fx, fy)
        elif event == cv2.EVENT_LBUTTONUP and drag["start"] is not None:
            session.add_box(*drag["start"], fx, fy)
            drag["start"] = drag["cur"] = None

    cv2.namedWindow("minimap")
    cv2.setMouseCallback("minimap", on_minimap_mouse)

    frame_window_open = False
    loaded_idx = -1
    minimap_bgr = frame_bgr = wall_mask = None

    while True:
        if loaded_idx != session.index:
            pair = session.current_pair
            minimap_bgr = cv2.imread(pair["minimap"])
            frame_bgr = cv2.imread(pair["frame"]) if pair["frame"] else None
            wall_mask = backend.terrain_mask(minimap_bgr) if minimap_bgr is not None else None
            loaded_idx = session.index

        if minimap_bgr is not None:
            cv2.imshow("minimap", _render_minimap(session, minimap_bgr, wall_mask))
        if frame_bgr is not None:
            if not frame_window_open:
                cv2.namedWindow("frame")
                cv2.setMouseCallback("frame", on_frame_mouse)
                frame_window_open = True
            cv2.imshow("frame", _render_frame(session, frame_bgr,
                                              drag["start"], drag["cur"]))

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            session.save()
            break
        elif key == ord("n"):
            session.next_pair()
        elif key == ord("p"):
            session.prev_pair()
        elif key == ord("c"):
            session.cycle_class()
        elif key == ord("u"):
            session.undo()
        elif key == ord("1"):
            session.set_mode("point")
        elif key == ord("2"):
            session.set_mode("box")
        elif key == ord("3"):
            session.set_mode("stroke_blocked")
        elif key == ord("4"):
            session.set_mode("stroke_open")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
