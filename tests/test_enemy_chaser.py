import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from state import EnemyChaser


def _grid_minimap(size=96):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(0, size, 8):
        img[y:y + 1, :, :] = 170
    for x in range(0, size, 8):
        img[:, x:x + 1, :] = 170
    return img


def test_enemy_chaser_forces_lateral_bypass_when_stalled():
    ch = EnemyChaser()
    mini = _grid_minimap()
    center = (48, 48)
    nearest = (76, 48)  # direct right
    mask = np.ones((96, 96), dtype=np.uint8) * 255

    got_lateral = False
    for _ in range(16):
        dx, dy = ch.get_chase_delta(mini, center, nearest, cached_mask=mask)
        if abs(dy) > 20:
            got_lateral = True
            break

    assert got_lateral


def test_enemy_chaser_returns_chase_vector_shape():
    ch = EnemyChaser()
    mini = _grid_minimap()
    center = (48, 48)
    nearest = (76, 48)
    mask = np.ones((96, 96), dtype=np.uint8) * 255

    dx, dy = ch.get_chase_delta(mini, center, nearest, cached_mask=mask)
    assert isinstance(dx, int)
    assert isinstance(dy, int)
    assert math.hypot(dx, dy) > 0


def test_enemy_chaser_bypass_triggers_with_noisy_frames_and_stagnant_distance():
    ch = EnemyChaser()
    base = _grid_minimap()
    center = (48, 48)
    nearest = (78, 48)
    mask = np.ones((96, 96), dtype=np.uint8) * 255

    got_lateral = False
    for i in range(18):
        noisy = np.roll(base, shift=(i % 3), axis=0)
        dx, dy = ch.get_chase_delta(noisy, center, nearest, cached_mask=mask)
        if abs(dy) > 20:
            got_lateral = True
            break

    assert got_lateral


def test_enemy_chaser_breaks_41px_plateau_with_lateral_move():
    ch = EnemyChaser()
    mini = _grid_minimap()
    center = (48, 48)
    nearest = (89, 48)  # 41px: reproduces flat chase plateau from runtime logs
    mask = np.ones((96, 96), dtype=np.uint8) * 255

    got_lateral = False
    for _ in range(20):
        dx, dy = ch.get_chase_delta(mini, center, nearest, cached_mask=mask)
        if abs(dy) > 16:
            got_lateral = True
            break

    assert got_lateral
