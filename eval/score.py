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


def evaluate_backend(backend, captures_root: str) -> dict:
    """Run a backend over every labeled pair under captures_root.

    Aggregates tp/fp/fn per class across all sessions; pairs without an
    existing labels file are skipped. Terrain scores average per-mode over
    pairs that have strokes for that mode.
    """
    import glob as _glob
    import os as _os

    from eval.labels import (
        MINIMAP_CLASSES, SCREEN_CLASSES, discover_pairs, load_labels,
    )

    counts = {cls: {"tp": 0, "fp": 0, "fn": 0}
              for cls in set(MINIMAP_CLASSES) | set(SCREEN_CLASSES)}
    terrain_totals = {mode: [] for mode in TERRAIN_MODES}
    pairs_scored = 0

    session_dirs = sorted(
        d for d in _glob.glob(_os.path.join(captures_root, "*"))
        if _os.path.isdir(d)
    )
    for session_dir in session_dirs:
        for pair in discover_pairs(session_dir):
            if not _os.path.exists(pair["labels"]):
                continue
            labels = load_labels(pair["labels"])
            minimap_bgr = cv2.imread(pair["minimap"])
            if minimap_bgr is None:
                continue
            pairs_scored += 1

            pred_minimap = backend.detect_minimap(minimap_bgr)
            for cls in MINIMAP_CLASSES:
                truth = [(p["x"], p["y"])
                         for p in labels.minimap_points if p["cls"] == cls]
                pred = pred_minimap.get(cls, [])
                if not truth and not pred:
                    continue
                tp, fp, fn = match_points(pred, truth, max_dist=6.0)
                counts[cls]["tp"] += tp
                counts[cls]["fp"] += fp
                counts[cls]["fn"] += fn

            frame_bgr = cv2.imread(pair["frame"]) if pair["frame"] else None
            if frame_bgr is not None and labels.screen_boxes:
                pred_screen = backend.detect_screen(frame_bgr)
                for cls in SCREEN_CLASSES:
                    truth = [(b["x"], b["y"], b["w"], b["h"])
                             for b in labels.screen_boxes if b["cls"] == cls]
                    pred = pred_screen.get(cls, [])
                    if not truth and not pred:
                        continue
                    tp, fp, fn = match_boxes(pred, truth, min_iou=0.4)
                    counts[cls]["tp"] += tp
                    counts[cls]["fp"] += fp
                    counts[cls]["fn"] += fn

            if any(labels.terrain_strokes.get(m) for m in TERRAIN_MODES):
                mask = backend.terrain_mask(minimap_bgr)
                if mask is not None:
                    scores = score_terrain_strokes(mask, labels.terrain_strokes)
                    for mode in TERRAIN_MODES:
                        if scores[mode] is not None:
                            terrain_totals[mode].append(scores[mode])

    classes = {}
    for cls, c in counts.items():
        if c["tp"] + c["fp"] + c["fn"] == 0:
            continue
        precision, recall, f1 = precision_recall(c["tp"], c["fp"], c["fn"])
        classes[cls] = {**c, "precision": precision, "recall": recall, "f1": f1}
    terrain = {mode: (sum(vals) / len(vals) if vals else None)
               for mode, vals in terrain_totals.items()}
    return {"classes": classes, "terrain": terrain, "pairs_scored": pairs_scored}


def format_report(backend_name: str, report: dict) -> str:
    lines = [
        f"backend: {backend_name}   pairs scored: {report['pairs_scored']}",
        f"{'class':<12} {'P':>6} {'R':>6} {'F1':>6} {'tp':>5} {'fp':>5} {'fn':>5}",
    ]
    for cls in sorted(report["classes"]):
        c = report["classes"][cls]
        lines.append(
            f"{cls:<12} {c['precision']:>6.2f} {c['recall']:>6.2f} "
            f"{c['f1']:>6.2f} {c['tp']:>5} {c['fp']:>5} {c['fn']:>5}"
        )
    for mode, score in report["terrain"].items():
        shown = f"{score:.2f}" if score is not None else "n/a"
        lines.append(f"terrain {mode}: {shown}")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    from eval.backends import get_backend

    parser = argparse.ArgumentParser(
        description="Score a detector backend against labeled captures.")
    parser.add_argument("--captures", default="eval/captures",
                        help="root folder holding session_* capture dirs")
    parser.add_argument("--detector", default="heuristic",
                        help="backend name (heuristic)")
    args = parser.parse_args(argv)
    backend = get_backend(args.detector)
    report = evaluate_backend(backend, args.captures)
    print(format_report(backend.name, report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
