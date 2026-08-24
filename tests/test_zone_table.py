"""The world-map zone table must cover every zone the game can rotate to.

Act 8 (Forest of the Slain, Flooded Plains, Forgotten Caves, Camp of Souls,
Hellheim) existed in the packet name table but never had coordinates, so when
the rotation landed there the bot had nothing to click and silently fell
through to a colour scan. A debug frame from 2026-07-18 caught it live with
satanic = "Forgotten Caves". This pins the gap shut.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
import stats_tracker as st
from map_scroll import scroll_plan

KNOWN = [z for act in st._ZONE_NAMES.values() for z in act]


def test_every_rotatable_zone_has_coordinates():
    missing = [z for z in KNOWN if z not in config.ZONE_NODES_WORLD]
    assert missing == [], f"zones the bot cannot click: {missing}"


def test_the_new_abyss_zones_are_present():
    for z in ("Volcanic Island", "Boreal Island", "Abyss Realm",
              "Abyss Jungle", "Tormented Reef", "Shipwreck Cove"):
        assert z in config.ZONE_NODES_WORLD, z


def test_all_coordinates_lie_inside_the_map():
    for name, (x, y) in config.ZONE_NODES_WORLD.items():
        assert 0 < x < config.WORLD_MAP_WIDTH_PX, f"{name} x={x}"
        assert 0 < y < config.SCREEN_H, f"{name} y={y}"


def test_every_zone_can_be_scrolled_onto_the_screen():
    """A zone that plans off-screen would be clicked into empty map."""
    for name, (x, _y) in config.ZONE_NODES_WORLD.items():
        notches, screen_x = scroll_plan(
            x, config.WORLD_MAP_MAX_SCROLL_PX,
            screen_w=config.SCREEN_W,
            px_per_notch=config.WORLD_MAP_PX_PER_NOTCH)
        assert notches >= 0, name
        assert 60 <= screen_x <= config.SCREEN_W - 60, \
            f"{name} plans to screen x={screen_x}"


def test_zones_do_not_share_a_node():
    """Two zones resolving to the same node means a mis-associated label."""
    seen = {}
    for name, pos in config.ZONE_NODES_WORLD.items():
        clash = seen.get(pos)
        assert clash is None, f"{name} and {clash} share node {pos}"
        seen[pos] = name
