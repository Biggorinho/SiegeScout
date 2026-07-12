"""Detector backends for the eval scorer.

Every backend exposes the same three methods so the scorer is
backend-agnostic. Later rollout steps register "roboflow" (dev-only API
vetting) and "onnx" (local models) here.

detect_minimap(minimap_bgr) -> {class_name: [(x, y), ...]}
detect_screen(frame_bgr)    -> {class_name: [(x, y, w, h), ...]}
terrain_mask(minimap_bgr)   -> walkable mask (uint8, 255 = walkable) or None
"""

from __future__ import annotations


class HeuristicBackend:
    """Adapts the shipped OpenCV heuristics to the backend protocol."""

    name = "heuristic"

    def detect_minimap(self, minimap_bgr):
        import detectors
        return {
            "enemy": [(d[0], d[1]) for d in detectors.classify_enemy_dots(minimap_bgr)],
            "item": [(d[0], d[1]) for d in detectors.classify_item_dots(minimap_bgr)],
            "ore": [(d[0], d[1]) for d in detectors.classify_ore_dots(minimap_bgr)],
        }

    def detect_screen(self, frame_bgr):
        # The heuristics have no on-screen box detector; the ONNX backend
        # (later rollout step) fills this in.
        return {}

    def terrain_mask(self, minimap_bgr):
        from minimap import build_walkable_minimap_mask
        return build_walkable_minimap_mask(minimap_bgr)


# Model class name → eval label class (eval.labels.MINIMAP_CLASSES). The
# nav-entrance classes all score against "portal": the eval vocabulary keeps
# one navigation-target class and the bot treats them alike downstream.
_ONNX_TO_EVAL_CLASS = {
    "enemy": "enemy",
    "player": "player",
    "portal": "portal",
    "dungeon": "portal",
    "enter": "portal",
    "exit": "portal",
    "oreGold": "ore",
    "oreGreen": "ore",
    "oreRuby": "ore",
    # item classes dropped from the model (too few labels); item detection
    # stays heuristic for now.
}


class OnnxBackend:
    """Local ONNX models via local_inference.OnnxDetector (rollout step 4).

    Constructing with missing weights is fine — the detectors fail soft and
    the backend just returns empty detections (the scorer will show zeros).
    """

    name = "onnx"

    def __init__(self):
        import config
        from local_inference import OnnxDetector
        self._minimap = OnnxDetector(
            config.LOCAL_MINIMAP_MODEL_PATH,
            config.LOCAL_MINIMAP_MODEL_CLASSES,
            config.LOCAL_MINIMAP_MODEL_INPUT_SIZE,
            conf_threshold=config.LOCAL_MINIMAP_MODEL_CONF,
        )
        self._screen = OnnxDetector(
            config.LOCAL_SCREEN_MODEL_PATH,
            config.LOCAL_SCREEN_MODEL_CLASSES,
            config.LOCAL_SCREEN_MODEL_INPUT_SIZE,
            conf_threshold=config.LOCAL_SCREEN_MODEL_CONF,
        )

    def detect_minimap(self, minimap_bgr):
        points = {}
        for cls, dets in self._minimap.detect(minimap_bgr).items():
            eval_cls = _ONNX_TO_EVAL_CLASS.get(cls, cls)
            points.setdefault(eval_cls, []).extend(
                (d["x"], d["y"]) for d in dets)
        return points

    def detect_screen(self, frame_bgr):
        boxes = {}
        for cls, dets in self._screen.detect(frame_bgr).items():
            boxes.setdefault(cls, []).extend(
                (d["x"] - d["width"] / 2, d["y"] - d["height"] / 2,
                 d["width"], d["height"])
                for d in dets)
        return boxes

    def terrain_mask(self, minimap_bgr):
        return None  # terrain stays pure OpenCV (spec §4)


def get_backend(name: str):
    if name == "heuristic":
        return HeuristicBackend()
    if name == "onnx":
        return OnnxBackend()
    raise ValueError(f"unknown backend: {name!r} (available: heuristic, onnx)")
