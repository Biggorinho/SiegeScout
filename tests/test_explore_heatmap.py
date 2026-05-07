import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from state import ExploreHeatMap
from config import EXPLORE_HEATMAP_PENALTY


def test_unvisited_cells_have_zero_penalty():
    hm = ExploreHeatMap(cell_size=8)
    assert hm.penalty(50, 50) == 0


def test_penalty_grows_after_visits():
    hm = ExploreHeatMap(cell_size=8)
    for _ in range(3):
        hm.update((50, 50), (200, 200))
    # Same cell as the visited centre — full strength penalty.
    assert hm.penalty(50, 50) == 3 * EXPLORE_HEATMAP_PENALTY


def test_adjacent_cells_get_partial_penalty():
    hm = ExploreHeatMap(cell_size=8)
    hm.update((50, 50), (200, 200))
    # Adjacent cell (east of centre): one step in cell coords.
    east_x = (50 // 8 + 1) * 8 + 1  # any pixel inside that cell
    assert hm.penalty(east_x, 50) == EXPLORE_HEATMAP_PENALTY
    # Cell two steps away — untouched.
    far_x = (50 // 8 + 3) * 8 + 1
    assert hm.penalty(far_x, 50) == 0


def test_shape_change_clears_visits():
    hm = ExploreHeatMap(cell_size=8)
    hm.update((50, 50), (200, 200))
    assert hm.penalty(50, 50) > 0
    # On a shape change `update()` clears _visits without crediting a new visit
    # (the visit only registers on the *next* call once shape stabilises).
    hm.update((50, 50), (300, 300))
    assert hm.penalty(50, 50) == 0
    # The next same-shape call writes the cell — exactly one increment.
    hm.update((50, 50), (300, 300))
    assert hm.penalty(50, 50) == EXPLORE_HEATMAP_PENALTY


def test_reset_clears_all_state():
    hm = ExploreHeatMap(cell_size=8)
    for _ in range(5):
        hm.update((50, 50), (200, 200))
    assert hm.penalty(50, 50) > 0
    hm.reset()
    assert hm.penalty(50, 50) == 0
    assert hm._visits == {}
    assert hm._last_shape is None
    assert hm._updates == 0


def test_visit_count_capped_at_max_visits():
    hm = ExploreHeatMap(cell_size=8)
    # Push way past _max_visits (24) to confirm the cap clamps.
    for _ in range(100):
        hm.update((50, 50), (200, 200))
    assert hm.penalty(50, 50) <= hm._max_visits * EXPLORE_HEATMAP_PENALTY
