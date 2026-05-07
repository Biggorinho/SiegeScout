"""pathfinding — package façade.

Re-exports the full public surface of the previous flat pathfinding.py module
so callers (main.py, combat.py, state.py, tests) work unchanged. New code is
encouraged to import directly from the relevant submodule.

Submodule layout:
    _astar       — A* primitives, Bresenham line-of-sight, string-pull smoothing
    _masks       — adaptive walkable-mask construction
    _navigation  — explore targeting, escape, attack-move heuristics
"""

from pathfinding._astar import (
    ray_walkability_score,
    smooth_path,
    _line_is_clear,
    _a_star_next_waypoint,
    _a_star_full_path,
    _astar_grid_path,
    _run_astar,
    _nearest_walkable_cell,
    _OCTILE_DIAG,
)
from pathfinding._masks import adaptive_walkable_mask
from pathfinding._navigation import (
    octant_escape_delta,
    path_delta_from_center,
    angle_is_blocked,
    scale_vector,
    steer_delta_away_from_walls,
    minimap_explore_target,
    choose_explore_delta,
    choose_recent_enemy_delta,
    choose_attack_move_delta,
)
# Backward-compat re-exports — these symbols actually live in `minimap`,
# but the old flat pathfinding.py re-exported them; some callers (state.py)
# still import them from `pathfinding`.
from minimap import buffered_walkable_mask, random_explore_delta
