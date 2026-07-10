"""Detector scoring for the eval harness.

Metric core (this file, top half) plus a CLI that runs a detector backend
over labeled capture sessions and prints a per-class comparison table
(added in a later task).

Matching rules from the design spec: minimap points match ground truth at
distance <= 6 px, on-screen boxes at IoU >= 0.4.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from eval.labels import TERRAIN_MODES


def match_points(pred, truth, max_dist: float = 6.0):
    """Greedy nearest-first point matching. Returns (tp, fp, fn)."""
    candidates = []
    for pi, (px, py) in enumerate(pred):
        for ti, (tx, ty) in enumerate(truth):
            dist = math.hypot(px - tx, py - ty)
            if dist <= max_dist:
                candidates.append((dist, pi, ti))
    candidates.sort()
    used_pred, used_truth = set(), set()
    for _, pi, ti in candidates:
        if pi in used_pred or ti in used_truth:
            continue
        used_pred.add(pi)
        used_truth.add(ti)
    tp = len(used_pred)
    return tp, len(pred) - tp, len(truth) - tp


def box_iou(a, b) -> float:
    """IoU of two (x, y, w, h) boxes, x/y = top-left."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_boxes(pred, truth, min_iou: float = 0.4):
    """Greedy highest-IoU-first box matching. Returns (tp, fp, fn)."""
    candidates = []
    for pi, pbox in enumerate(pred):
        for ti, tbox in enumerate(truth):
            iou = box_iou(pbox, tbox)
            if iou >= min_iou:
                candidates.append((-iou, pi, ti))
    candidates.sort()
    used_pred, used_truth = set(), set()
    for _, pi, ti in candidates:
        if pi in used_pred or ti in used_truth:
            continue
        used_pred.add(pi)
        used_truth.add(ti)
    tp = len(used_pred)
    return tp, len(pred) - tp, len(truth) - tp


def precision_recall(tp: int, fp: int, fn: int):
    """Returns (precision, recall, f1); zero-division yields 0.0."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


def score_terrain_strokes(walkable_mask, strokes: dict) -> dict:
    """Fraction of correction-stroke pixels the walkable mask now gets right.

    "wrongly_blocked" strokes assert those pixels SHOULD be walkable (255);
    "wrongly_open" strokes assert they SHOULD be blocked (0). Modes without
    strokes score None.
    """
    result = {}
    walkable = walkable_mask > 0
    for mode in TERRAIN_MODES:
        circles = strokes.get(mode, [])
        if not circles:
            result[mode] = None
            continue
        stroke_mask = np.zeros(walkable_mask.shape[:2], dtype=np.uint8)
        for c in circles:
            cv2.circle(stroke_mask, (int(c["x"]), int(c["y"])), int(c["r"]), 255, -1)
        selected = stroke_mask > 0
        total = int(np.count_nonzero(selected))
        if total == 0:
            result[mode] = None
            continue
        if mode == "wrongly_blocked":
            correct = int(np.count_nonzero(walkable & selected))
        else:
            correct = int(np.count_nonzero(~walkable & selected))
        result[mode] = correct / total
    return result
