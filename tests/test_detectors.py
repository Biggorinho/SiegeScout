import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detectors import (
    classify_enemy_dots,
    classify_item_dots,
    classify_ore_dots,
    is_world_map_visible,
)


def _load_minimap_from_debug(name: str):
    path = Path(__file__).resolve().parent.parent / "debug" / name
    if not path.exists():
        pytest.skip(f"missing debug fixture: {name}")
    image = cv2.imread(str(path))
    assert image is not None, f"failed to decode debug fixture: {name}"
    return image[:, : image.shape[1] // 3].copy()


def test_classify_enemy_dots_detects_synthetic_skull_blob():
    minimap = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.circle(minimap, (80, 40), 6, (235, 235, 235), -1)
    cv2.circle(minimap, (80, 40), 3, (120, 120, 120), -1)

    dots = classify_enemy_dots(minimap)

    assert any(kind == "skull" for _, _, _, kind in dots)


def test_classify_enemy_dots_keeps_real_skull_frame():
    minimap = _load_minimap_from_debug("skull_dbg_attack_z26_1776812055144.png")

    dots = classify_enemy_dots(minimap)

    assert any(kind == "skull" for _, _, _, kind in dots)


def test_classify_enemy_dots_rejects_duck_like_blob():
    minimap = np.zeros((120, 120, 3), dtype=np.uint8)

    cv2.circle(minimap, (60, 60), 15, (235, 235, 235), -1)
    cv2.circle(minimap, (68, 60), 6, (0, 165, 255), -1)
    cv2.circle(minimap, (56, 54), 4, (245, 245, 245), -1)

    dots = classify_enemy_dots(minimap)

    assert not any(kind == "skull" for _, _, _, kind in dots)


# ── item-star detection (real captured frame) ──────────────────────────────

_DARK_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dark_minimap.png"


def _load_dark_fixture():
    if not _DARK_FIXTURE.exists():
        pytest.skip("dark_minimap fixture not present")
    img = cv2.imread(str(_DARK_FIXTURE))
    assert img is not None
    return img


def test_classify_item_dots_catches_real_star():
    """The red item star (~108,137 in the fixture) must be detected. The old
    gate's hard `defects >= 4` requirement silently dropped real stars."""
    dots = classify_item_dots(_load_dark_fixture())
    assert any((d[0] - 108) ** 2 + (d[1] - 137) ** 2 <= 8 * 8 for d in dots), \
        f"item star not detected; got {dots}"


def test_classify_item_dots_rejects_high_confidence_lookalikes():
    """The high-confidence non-items must not be reported: teleport crosses
    (bright cyan) and the round NPC/objective dot. Ore pickaxes are knowingly
    not asserted here — they're near-identical to stars at icon size and are
    handled by the separate ore detector/toggle."""
    dots = classify_item_dots(_load_dark_fixture())
    lookalikes = [(162, 175), (91, 82), (189, 144)]  # crossA, crossB, round dot
    for lx, ly in lookalikes:
        assert not any((d[0] - lx) ** 2 + (d[1] - ly) ** 2 <= 8 * 8 for d in dots), \
            f"look-alike at {(lx, ly)} wrongly classified as item; got {dots}"


def _star_polygon(cx, cy, r_out=7, r_in=3, n=5):
    import math
    pts = []
    for i in range(2 * n):
        r = r_out if i % 2 == 0 else r_in
        a = -math.pi / 2 + i * math.pi / n
        pts.append([int(round(cx + r * math.cos(a))), int(round(cy + r * math.sin(a)))])
    return np.array(pts, dtype=np.int32)


def test_classify_item_dots_is_color_agnostic():
    """Same star glyph in different rarity colours should all be detected
    (real minimap stars are ~13px; the gate caps area at 300)."""
    for color in [(0, 0, 220), (220, 0, 0), (0, 220, 0), (0, 200, 220)]:  # red, blue, green, gold
        minimap = np.zeros((120, 120, 3), dtype=np.uint8)
        cv2.fillPoly(minimap, [_star_polygon(40, 40)], color)  # off-centre (center is excluded)
        dots = classify_item_dots(minimap)
        assert dots, f"star of colour {color} not detected"


def _fixture(name):
    p = Path(__file__).resolve().parent / "fixtures" / name
    if not p.exists():
        pytest.skip(f"missing fixture: {name}")
    img = cv2.imread(str(p))
    assert img is not None
    return img


def test_is_world_map_visible_detects_map_selection():
    """The map-selection (world map) screen must be detected as the map state."""
    assert is_world_map_visible(_fixture("map_selection.png")) is True


def test_is_world_map_visible_false_in_safe_zone():
    """The safe zone is NOT the world map — must not be mistaken for map selection."""
    assert is_world_map_visible(_fixture("safe_zone.png")) is False


def test_is_world_map_visible_false_in_purple_zone_combat():
    """Hell 'Parallel Dimension' zones tint ordinary gameplay purple (ratio 0.6+),
    which the pre-fix detector mistook for the map parchment on every frame,
    derailing the whole vote-reset flow (captured 2026-07-11 failure run)."""
    assert is_world_map_visible(_fixture("purple_zone_combat.png")) is False


def test_is_world_map_visible_false_in_town_with_pause_menu():
    """Post-reset town + open pause menu (dim purple palette) is not the map.
    The pre-fix false positive here made pick_zone click a phantom map while
    the pause menu had focus, so every vote reset 'failed' after succeeding."""
    assert is_world_map_visible(_fixture("purple_town_pause_menu.png")) is False


def test_is_world_map_visible_false_on_pause_menu_in_combat():
    """Pause menu over a combat scene (purple ratio 0.41) is not the map."""
    assert is_world_map_visible(_fixture("pause_menu_combat.png")) is False


def test_is_pause_menu_visible_false_on_map_selection():
    """The world map's red node glow must not read as pause-menu button bars —
    guards the menu-before-map check order in the vote-reset flow."""
    from detectors import is_pause_menu_visible
    assert is_pause_menu_visible(_fixture("map_selection.png")) is False


def test_classify_ore_dots_finds_real_ore():
    """Real orange ore icons (~64,131 and ~67,111 in the fixture) are detected."""
    ore = classify_ore_dots(_load_dark_fixture())
    assert any((o[0] - 65) ** 2 + (o[1] - 120) ** 2 <= 20 * 20 for o in ore), \
        f"real ore not detected; got {ore}"


def test_classify_ore_dots_rejects_item_star():
    """The red item star (~107,136) must NOT be tagged as ore — the old 'any
    saturated colour' mask wrongly claimed it (and red enemy dots) as ore."""
    ore = classify_ore_dots(_load_dark_fixture())
    assert not any((o[0] - 107) ** 2 + (o[1] - 136) ** 2 <= 8 * 8 for o in ore), \
        f"item star wrongly tagged as ore; got {ore}"


def test_classify_item_dots_rejects_cyan_cross():
    """A bright-cyan plus (teleport) must not be reported as an item."""
    minimap = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.rectangle(minimap, (36, 28), (44, 52), (255, 230, 60), -1)  # off-centre vertical bar
    cv2.rectangle(minimap, (28, 36), (52, 44), (255, 230, 60), -1)  # horizontal bar
    dots = classify_item_dots(minimap)
    assert not dots, f"cyan cross wrongly classified as item; got {dots}"