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

from hero_siege_demonspawn_satanic_bot import _a_star_full_path, smooth_path

def _open_mask(size=(40, 40)):
    import numpy as np
    return np.full(size, 255, dtype=np.uint8)

def _mask_with_wall(size=(40, 40), wall_col=20):
    import numpy as np
    mask = np.full(size, 255, dtype=np.uint8)
    # Wall must be PATH_GRID_STEP (3) pixels wide so it survives downsampling.
    # Zero out columns wall_col-1 .. wall_col+1 to guarantee one grid column is blocked.
    mask[:, wall_col - 1:wall_col + 2] = 0
    return mask

def test_full_path_open_area_returns_list():
    mask = _open_mask()
    path = _a_star_full_path(mask, (2, 2), (36, 36))
    assert path is not None
    assert len(path) >= 2
    assert path[0] != path[-1]

def test_full_path_unreachable_returns_none():
    mask = _mask_with_wall()
    path = _a_star_full_path(mask, (2, 20), (36, 20))
    assert path is None

def test_smooth_path_shortens_straight_line():
    mask = _open_mask()
    path = [(i * 3, i * 3) for i in range(10)]
    result = smooth_path(path, mask)
    assert len(result) <= len(path)

def test_smooth_path_identity_on_short_path():
    mask = _open_mask()
    path = [(0, 0), (5, 5)]
    result = smooth_path(path, mask)
    assert len(result) == 2

from hero_siege_demonspawn_satanic_bot import ExploreHeatMap

def test_heatmap_penalty_increases_with_visits():
    hm = ExploreHeatMap(cell_size=8)
    shape = (80, 80)
    center = (20, 20)
    assert hm.penalty(20, 20) == 0.0
    hm.update(center, shape)
    hm.update(center, shape)
    assert hm.penalty(20, 20) > 0.0

def test_heatmap_resets_on_shape_change():
    hm = ExploreHeatMap(cell_size=8)
    hm.update((20, 20), (80, 80))
    assert hm.penalty(20, 20) > 0.0
    hm.update((20, 20), (100, 100))
    assert hm.penalty(20, 20) == 0.0

from hero_siege_demonspawn_satanic_bot import EnemyChaser
import numpy as np

def test_enemy_chaser_returns_tuple():
    chaser = EnemyChaser()
    mask = np.full((80, 80), 255, dtype=np.uint8)
    minimap = np.full((80, 80, 3), (180, 180, 180), dtype=np.uint8)
    center = (40, 40)
    nearest = (60, 30)
    delta = chaser.get_chase_delta(minimap, center, nearest, cached_mask=mask)
    assert isinstance(delta, tuple)
    assert len(delta) == 2

def test_enemy_chaser_fallback_on_no_mask():
    chaser = EnemyChaser()
    center = (40, 40)
    nearest = (60, 30)
    delta = chaser.get_chase_delta(None, center, nearest, cached_mask=None)
    assert isinstance(delta, tuple)
