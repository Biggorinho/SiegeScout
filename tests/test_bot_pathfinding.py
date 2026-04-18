import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import cv2
from hero_siege_demonspawn_satanic_bot import buffered_walkable_mask

def _solid_minimap(color_bgr=(180, 180, 180), size=(60, 60)):
    img = np.full((size[0], size[1], 3), color_bgr, dtype=np.uint8)
    return img

def test_buffered_mask_returns_none_for_empty():
    assert buffered_walkable_mask(None, (30, 30)) is None
    assert buffered_walkable_mask(np.array([]), (30, 30)) is None

def test_buffered_mask_smaller_than_unbuffered():
    img = _solid_minimap()
    center = (30, 30)
    from hero_siege_demonspawn_satanic_bot import build_walkable_minimap_mask, connected_walkable_mask
    raw = connected_walkable_mask(build_walkable_minimap_mask(img), center)
    buffered = buffered_walkable_mask(img, center)
    if raw is not None and buffered is not None:
        assert buffered.sum() <= raw.sum()
