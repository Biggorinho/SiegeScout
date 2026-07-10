"""Label schema for the eval harness: one JSON per captured frame pair.

Dev-only module — never imported by shipped code. Terrain is labeled as
sparse correction strokes (circles) rather than exhaustive masks: strokes
mark where the current wall mask is wrong, and the scorer only judges those
pixels.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os

MINIMAP_CLASSES = ("enemy", "item", "ore", "portal", "player")
SCREEN_CLASSES = ("enemy", "healthbar", "teleport")
TERRAIN_MODES = ("wrongly_blocked", "wrongly_open")


def _empty_strokes():
    return {mode: [] for mode in TERRAIN_MODES}


@dataclasses.dataclass
class FrameLabels:
    frame: str = ""
    minimap: str = ""
    minimap_points: list = dataclasses.field(default_factory=list)
    screen_boxes: list = dataclasses.field(default_factory=list)
    terrain_strokes: dict = dataclasses.field(default_factory=_empty_strokes)


def save_labels(path: str, labels: FrameLabels) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dataclasses.asdict(labels), fh, indent=1)


def load_labels(path: str) -> FrameLabels:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    for mode in raw.get("terrain_strokes", {}):
        if mode not in TERRAIN_MODES:
            raise ValueError(f"unknown terrain stroke mode: {mode!r} in {path}")
    labels = FrameLabels(
        frame=raw.get("frame", ""),
        minimap=raw.get("minimap", ""),
        minimap_points=list(raw.get("minimap_points", [])),
        screen_boxes=list(raw.get("screen_boxes", [])),
        terrain_strokes={mode: list(raw.get("terrain_strokes", {}).get(mode, []))
                         for mode in TERRAIN_MODES},
    )
    for point in labels.minimap_points:
        if point.get("cls") not in MINIMAP_CLASSES:
            raise ValueError(f"unknown minimap class: {point.get('cls')!r} in {path}")
    for box in labels.screen_boxes:
        if box.get("cls") not in SCREEN_CLASSES:
            raise ValueError(f"unknown screen class: {box.get('cls')!r} in {path}")
    return labels


def discover_pairs(session_dir: str) -> list[dict]:
    """Find capture pairs in a session dir, keyed by shared <stamp>.

    Returns entries sorted by stamp: {"stamp", "frame" (path or None),
    "minimap" (path), "labels" (labels_<stamp>.json path, existing or not)}.
    Pairs need at least a minimap image; frame-only stamps are ignored
    (labeling and scoring both start from the minimap).
    """
    frames = {}
    for path in glob.glob(os.path.join(session_dir, "frame_*.png")):
        stamp = os.path.basename(path)[len("frame_"):-len(".png")]
        frames[stamp] = path
    pairs = []
    for path in glob.glob(os.path.join(session_dir, "minimap_*.png")):
        stamp = os.path.basename(path)[len("minimap_"):-len(".png")]
        pairs.append({
            "stamp": stamp,
            "frame": frames.get(stamp),
            "minimap": path,
            "labels": os.path.join(session_dir, f"labels_{stamp}.json"),
        })
    pairs.sort(key=lambda p: p["stamp"])
    return pairs
