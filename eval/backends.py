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


def get_backend(name: str):
    if name == "heuristic":
        return HeuristicBackend()
    raise ValueError(f"unknown backend: {name!r} (available: heuristic)")
