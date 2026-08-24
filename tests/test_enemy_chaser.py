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


def test_enemy_chaser_first_move_is_not_the_players_own_cell():
    """A* returns the player's own grid cell as path[0]; chasing it means standing still.

    _astar_grid_path starts the path at the start cell, so smooth_path keeps the
    player's own position at the head of the queue. Converted by
    dot_to_screen_delta that is a 0-6px command while the enemy is 112px away,
    which is what produced `[PATH] A* chase dist=132px -> 0°` repeating at an
    unchanging distance in the live logs. atan2(0, 0) is 0.0, so those 0°
    entries are "no movement", not "due east".
    """
    ch = EnemyChaser()
    mini = _grid_minimap()
    center = (48, 48)
    nearest = (76, 48)  # 28px right on the minimap = 112px on screen
    mask = np.ones((96, 96), dtype=np.uint8) * 255

    dx, dy = ch.get_chase_delta(mini, center, nearest, cached_mask=mask)

    assert math.hypot(dx, dy) > 40, (
        f"chase command {(dx, dy)} is a stand-still nudge, not a move toward an "
        f"enemy 112px away"
    )
    assert dx > 0, f"chase command {(dx, dy)} does not head toward the enemy"


def test_enemy_chaser_drains_every_reached_waypoint():
    """Popping one waypoint per call is not enough when several are already behind."""
    ch = EnemyChaser()
    mini = _grid_minimap()
    center = (48, 48)
    nearest = (76, 48)
    mask = np.ones((96, 96), dtype=np.uint8) * 255

    ch.get_chase_delta(mini, center, nearest, cached_mask=mask)
    # Stack several waypoints the player is already standing on.
    ch.path_queue = [(48, 48), (49, 48), (48, 49)] + list(ch.path_queue)
    dx, dy = ch.get_chase_delta(mini, center, nearest, cached_mask=mask)

    assert math.hypot(dx, dy) > 40, (
        f"chase command {(dx, dy)} still points at an already-reached waypoint"
    )


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
