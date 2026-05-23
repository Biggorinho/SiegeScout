"""state — package façade.

Re-exports the full public surface of the previous flat state.py module so
callers (main.py, tests) work unchanged. New code is encouraged to import
directly from the relevant submodule.

Submodule layout:
    _stuck    — StuckDetector, IntentStuckDetector
    _explore  — ExploreHeatMap, SectorTracker, TerrainMemory, ExploreNavigator,
                TERRAIN_BLOCKED_DECAY_KEEP_RADIUS_PX
    _combat   — CompanionSuppressor, EnemyChaser
    _resource — ResourceTargeter, ResourceTarget
"""

from enum import Enum


class State(Enum):
    EXPLORE = "EXPLORE"
    ATTACK  = "ATTACK"
    LOOT    = "LOOT"
    FLEE    = "FLEE"


from state._stuck import StuckDetector, IntentStuckDetector
from state._explore import (
    ExploreHeatMap,
    SectorTracker,
    TerrainMemory,
    ExploreNavigator,
    TERRAIN_BLOCKED_DECAY_KEEP_RADIUS_PX,
)
from state._combat import CompanionSuppressor, EnemyChaser
from state._resource import ResourceTargeter, ResourceTarget
