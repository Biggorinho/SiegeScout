"""Pan the world map to one of its two edges, and say what is visible there.

The zone-select map is 3918px wide, the screen shows 1920 of it, and it pans
1997.7px before clamping. Two positions therefore show the whole thing: parked
hard LEFT you see world x 0..1920 -- 40 of the 46 zones -- and parked hard
RIGHT you see 1997.7..3917.7, which is the abyss region and nothing else. There
is no third view worth visiting.

That is why everything here is expressed as an *edge* rather than a scroll
offset. An offset has to be counted out in wheel notches and believed
afterwards; the bot has no way to check it, so a map that scrolled on its own
-- a stray wheel event, the game nudging it -- leaves every coordinate wrong by
an unknown amount and the clicks land in empty map. An edge announces itself
instead: once the map is clamped, panning further stops changing the image.
Reaching an edge is a fact that can be verified, so the coordinates read there
are too.

The walk takes injected `grab`/`scroll` callables so it is testable without a
game attached.
"""

import cv2
import numpy as np

# Mean absolute per-pixel difference below which two captures are "the same".
# Measured margins are wide: a 4px pan of a real map frame scores ~13.7 and a
# static frame with capture noise scores ~0.67.
STABLE_MAD = 3.0

# Measured pan distance for one wheel notch, at 1920x1080.
PX_PER_NOTCH = 30

# Keep nodes this far from the screen edges when deciding what an edge shows:
# the scoring patch has a 14px radius, and a node clipped by the frame border
# scores badly enough to be missed.
EDGE_SCAN_MARGIN = 60

MAP_EDGE_LEFT = "left"
MAP_EDGE_RIGHT = "right"


def frame_delta(a, b):
    """Mean absolute difference between two frames; inf if incomparable."""
    if a is None or b is None or a.shape != b.shape:
        return float("inf")
    return float(np.mean(cv2.absdiff(a, b)))


def scroll_to_edge(grab, scroll, step, max_steps=200, on_capture=None,
                   should_continue=None, settle=None, stable_mad=STABLE_MAD):
    """Capture, scroll by `step` notches, repeat until the map stops moving.

    Args:
        grab: () -> frame.
        scroll: (notches) -> None. Positive notches pan LEFT, matching
            input_control.mouse_scroll.
        step: notches per iteration -- either one number, or a schedule of
            them, cycled. A schedule is what makes the walk cheap, because
            arriving at an edge is only ever learned one step late: something
            has to be scrolled that fails to move the map. Alternating
            ``(probe, traverse)`` pays a few notches for that proof instead of
            another map's width, and the traverse step can be as wide as the
            whole map because overshooting an edge simply clamps.
            The sign of every value chooses the direction to walk.
        max_steps: safety cap, so a map that never settles cannot spin forever.
        on_capture: optional (index, frame) callback, for saving frames.
        should_continue: optional () -> bool, checked before each iteration so
            a stop key can abort the walk.
        settle: optional () -> None, called after each scroll to let the map
            finish animating before the next capture.

    Returns:
        ``(frames, reason)`` where reason is "edge" when the map stopped
        moving, "max_steps" when the cap was hit -- in which case the edge was
        NOT reached and the caller must not assume it was -- or "aborted".
    """
    schedule = (step,) if isinstance(step, (int, float)) else tuple(step)
    if not schedule:
        raise ValueError("scroll_to_edge needs at least one step")
    frames = []
    prev = None
    for i in range(max_steps):
        if should_continue is not None and not should_continue():
            return frames, "aborted"
        frame = grab()
        frames.append(frame)
        if on_capture is not None:
            on_capture(i, frame)
        if frame_delta(prev, frame) <= stable_mad:
            return frames, "edge"
        prev = frame
        scroll(schedule[i % len(schedule)])
        if settle is not None:
            settle()
    return frames, "max_steps"


def edge_offsets(max_scroll_px):
    """How far the map has panned at each edge, in world pixels."""
    return {MAP_EDGE_LEFT: 0.0, MAP_EDGE_RIGHT: max(0.0, float(max_scroll_px))}


def zone_edge_positions(zone_table, max_scroll_px, screen_w=1920,
                        margin=EDGE_SCAN_MARGIN):
    """Work out, once, which edge shows each zone and where it sits there.

    Returns ``{name: (edge, screen_x, screen_y)}``. Zone coordinates are stored
    in *world* space -- their x on a map panned fully left -- so this is just
    ``world_x - edge_offset``, but doing it up front turns every later question
    ("where do I click Hellheim?", "which edges does a colour scan have to
    visit?") into a lookup instead of arithmetic that can drift.

    A zone visible at both edges is filed under the left one, so the bot has a
    single answer per zone and never travels for a node it can already see. A
    zone visible at neither is omitted rather than clamped to an edge: callers
    must fall back instead of clicking a guessed position.
    """
    offsets = edge_offsets(max_scroll_px)
    positions = {}
    for name, (world_x, world_y) in zone_table.items():
        for edge in (MAP_EDGE_LEFT, MAP_EDGE_RIGHT):
            screen_x = world_x - offsets[edge]
            if margin <= screen_x <= screen_w - margin:
                positions[name] = (edge, int(round(screen_x)),
                                   int(round(world_y)))
                break
    return positions


def plan_edge_click(zone_name, edge_positions, click_y_bias=0):
    """Plan the click for a named zone: ``(edge, click_x, click_y)``.

    Returns None for a zone with no calibrated position, so callers can fall
    back rather than click a guess.

    `click_y_bias` offsets the returned y. combat's _click_node_and_choose
    adds ~0.6% of screen height before clicking, and the median map node has a
    radius of only 9.6px, so passing the negative of that lands the real click
    on the node centre instead of its lower rim.
    """
    entry = edge_positions.get(zone_name)
    if entry is None:
        return None
    edge, screen_x, screen_y = entry
    return edge, int(screen_x), int(round(screen_y + click_y_bias))


def edges_with_zones(edge_positions):
    """Edges a colour scan has to visit, left first.

    A scan can only score nodes that are actually on screen, so it has to
    visit both edges when zones live at both -- that is what lets a satanic
    zone in the abyss region be found at all. It also means an edge that hosts
    no zone is never worth the trip.
    """
    present = {edge for edge, _x, _y in edge_positions.values()}
    visiting = tuple(e for e in (MAP_EDGE_LEFT, MAP_EDGE_RIGHT) if e in present)
    return visiting or (MAP_EDGE_LEFT,)
