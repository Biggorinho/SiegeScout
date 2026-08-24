"""The colour scan must still find the red satanic node, as a fallback.

Zone selection now reads the satanic zone by name from the HUD, but that path
needs the HUD to be readable and the map to be normalized, so the colour scan
remains the last resort. It has to keep working.

The fixture is the left edge of the map as captured on 2026-08-24, with the
rotation sitting on Forest of the Slain -- an Act 8 zone, which is exactly the
group that had no coordinates at all until that calibration.

This replaces a test built on worldmap_corrupted_cave.jpg, a frame from the
pre-2026-08 map layout. That layout no longer exists: the season moved every
node (only 1 of the 35 old coordinates still landed within 15px of a real
one), so the old frame could not be satisfied by any correct coordinate table.
"""
import os
import sys
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import combat
import config

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "worldmap_satanic_forest.png"
_EXPECTED = "Forest of the Slain"


def _frame():
    if not _FIXTURE.exists():
        pytest.skip("missing fixture: worldmap_satanic_forest.png")
    img = cv2.imread(str(_FIXTURE))
    assert img is not None
    return img


def test_fixed_nodes_find_the_satanic_node():
    result = combat._find_satanic_node_from_fixed_nodes(_frame())
    assert result is not None, "red satanic node not found by fixed-node scoring"
    ex, ey = config.ZONE_NODES_WORLD[_EXPECTED]
    assert abs(result[0] - ex) <= 30 and abs(result[1] - ey) <= 30, \
        f"picked wrong node at {result[:2]}, expected ~({ex},{ey})"


def test_full_finder_returns_the_satanic_node():
    result = combat._find_satanic_node(_frame())
    assert result is not None
    ex, ey = config.ZONE_NODES_WORLD[_EXPECTED]
    assert abs(result[0] - ex) <= 30 and abs(result[1] - ey) <= 30


def test_the_scan_agrees_with_the_calibrated_coordinate():
    """Forest of the Slain had no blue node at capture time -- it renders as
    the red satanic glow -- so its coordinate was inferred from the label
    offset. The colour scan locating the glow independently is what confirms
    that inference was right."""
    result = combat._find_satanic_node_from_fixed_nodes(_frame())
    ex, ey = config.ZONE_NODES_WORLD[_EXPECTED]
    assert abs(result[0] - ex) <= 4 and abs(result[1] - ey) <= 4
