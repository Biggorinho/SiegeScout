"""The satanic zone rotates hourly; the picker must find the red node for
every zone in SATANIC_FIXED_NODES. On 2026-07-11 the 11:00 rotation moved it
to Corrupted Cave, whose calibrated coordinates were wrong (pointed at empty
map), and the broad fallback scan cut off at 82% screen height — right above
the node. The fixture is the actual failure frame (node at ~(565, 868),
map scrolled +25px from the reference frame, corrected by offset voting)."""
import os
import sys
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import combat

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "worldmap_corrupted_cave.jpg"


def _frame():
    if not _FIXTURE.exists():
        pytest.skip("missing fixture: worldmap_corrupted_cave.jpg")
    img = cv2.imread(str(_FIXTURE))
    assert img is not None
    return img


def test_fixed_nodes_find_corrupted_cave():
    result = combat._find_satanic_node_from_fixed_nodes(_frame())
    assert result is not None, "red satanic node not found by fixed-node scoring"
    x, y = result[0], result[1]
    assert abs(x - 565) <= 30 and abs(y - 868) <= 30, f"picked wrong node at ({x},{y})"


def test_full_finder_returns_corrupted_cave():
    result = combat._find_satanic_node(_frame())
    assert result is not None
    assert abs(result[0] - 565) <= 30 and abs(result[1] - 868) <= 30
