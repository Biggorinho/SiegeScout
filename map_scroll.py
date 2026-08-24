"""Pan the world map to a known edge, by detecting when it stops moving.

The zone-select map is far wider than the screen and pans exactly 30px per
wheel notch (measured by phase correlation across a 40-notch capture: -30.00
px/notch, zero vertical drift). Because every calibrated node coordinate is
expressed relative to the left edge, reaching that edge reliably is the
foundation everything else stands on.

Scrolling a fixed number of notches cannot do that. combat's
_reopen_world_map_for_recenter scrolls 12 notches -- 360px -- to "normalize to
leftmost", which only works if the map happened to be near the left already.
The map is wide enough that a right-scrolled wheel leaves the bot reading
every fixed coordinate at the wrong offset, silently.

The edge announces itself instead: once the map is clamped, panning further
stops changing the image. That is a fact about the map, not a guess about its
width, so it keeps working when the next season widens it again.

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
        step: notches per iteration; sign chooses the direction to walk.
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
        scroll(step)
        if settle is not None:
            settle()
    return frames, "max_steps"


def scroll_plan(world_x, max_scroll_px, screen_w=1920, px_per_notch=PX_PER_NOTCH):
    """Plan how far to scroll right to put a world coordinate on screen.

    Zone coordinates are stored in *world* space -- their x on a map panned
    fully left -- because the map is 3918px wide and most zones are off screen
    at any one time. Screen position is then simply ``world_x - offset``.

    Returns ``(notches_right, screen_x)``: how many wheel-down notches to send
    from the left edge, and where the zone will then be on screen.

    Two clamps matter and both are easy to get wrong:

    * The map stops at `max_scroll_px`, so the right-most zones cannot be
      centred and simply sit wherever the clamp leaves them.
    * `max_scroll_px` is not a whole number of notches (1997.7 vs 30px), so
      the final notch is short. Computing screen position as
      ``notches * px_per_notch`` puts the far-right zones ~12px off, which is
      most of the click tolerance.
    """
    desired = world_x - screen_w / 2.0
    desired = max(0.0, min(desired, float(max_scroll_px)))
    notches = int(round(desired / px_per_notch))
    actual = min(notches * px_per_notch, float(max_scroll_px))
    return notches, world_x - actual


def plan_zone_click(zone_name, zone_table, max_scroll_px, screen_w=1920,
                    px_per_notch=PX_PER_NOTCH, click_y_bias=0):
    """Plan the click for a named zone: ``(notches_right, click_x, click_y)``.

    Returns None for a zone with no calibrated coordinate, so callers can fall
    back rather than click a guessed position.

    `click_y_bias` offsets the returned y. combat's _click_node_and_choose
    adds ~0.6% of screen height before clicking, and the median map node has a
    radius of only 9.6px, so passing the negative of that lands the real click
    on the node centre instead of its lower rim.
    """
    entry = zone_table.get(zone_name)
    if entry is None:
        return None
    world_x, world_y = entry
    notches, screen_x = scroll_plan(world_x, max_scroll_px, screen_w,
                                    px_per_notch)
    return notches, int(round(screen_x)), int(round(world_y + click_y_bias))
