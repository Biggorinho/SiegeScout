import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from state import SectorTracker


def test_sector_for_east():
    st = SectorTracker()
    # angle=0 (east) → sector 0
    assert st._angle_to_sector(0.0) == 0


def test_sector_for_south():
    st = SectorTracker()
    # angle=π/2 (south) → sector 1
    assert st._angle_to_sector(math.pi / 2) == 1


def test_update_increments_sector():
    st = SectorTracker()
    st.update(0.0)  # east, sector 0
    assert st.visits(0) == 1
    assert st.visits(1) == 0


def test_penalty_scales_with_visits():
    st = SectorTracker(penalty_per_visit=25.0)
    st.update(0.0)
    st.update(0.0)
    assert st.penalty(0.0) == 50.0


def test_decay_reduces_counts():
    st = SectorTracker(decay_every=3)
    st.update(0.0)
    st.update(0.0)
    st.update(0.0)  # 3rd update → decay fires
    assert st.visits(0) == 2  # was 3, decayed by 1


def test_reset_clears_all():
    st = SectorTracker()
    st.update(0.0)
    st.reset()
    assert st.visits(0) == 0


def test_max_visits_capped():
    st = SectorTracker()
    for _ in range(30):
        st.update(0.0)
    assert st.visits(0) <= 20
