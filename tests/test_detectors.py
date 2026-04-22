import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detectors import classify_enemy_dots


def _load_minimap_from_debug(name: str):
    path = Path(__file__).resolve().parent.parent / "debug" / name
    image = cv2.imread(str(path))
    assert image is not None, f"missing debug fixture: {name}"
    return image[:, : image.shape[1] // 3].copy()


def test_classify_enemy_dots_keeps_real_skull_frame():
    minimap = _load_minimap_from_debug("skull_dbg_attack_z26_1776812055144.png")

    dots = classify_enemy_dots(minimap)

    assert any(kind == "skull" for _, _, _, kind in dots)


def test_classify_enemy_dots_rejects_duck_like_blob():
    minimap = np.zeros((120, 120, 3), dtype=np.uint8)

    cv2.circle(minimap, (60, 60), 15, (235, 235, 235), -1)
    cv2.circle(minimap, (68, 60), 6, (0, 165, 255), -1)
    cv2.circle(minimap, (56, 54), 4, (245, 245, 245), -1)

    dots = classify_enemy_dots(minimap)

    assert not any(kind == "skull" for _, _, _, kind in dots)