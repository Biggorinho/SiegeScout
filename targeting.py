"""Pick which minimap enemy dot to attack, from *confirmed* dots only.

Split out of main.run()'s frame loop. The selection itself is unchanged --
walkability-weighted scoring when a mask is available, tier-then-distance
otherwise -- but it now runs over the temporally confirmed dots instead of
every raw detection.

That distinction is the whole point of this module. Confirmation and target
choice used to be independent: `nearest` was chosen from all dots, and the
firing gate only asked whether *any* dot had confirmed. With the ML minimap
detector fused in, its dots flicker, so a real enemy across the room kept the
gate open while a one-frame phantom -- typically nearer, therefore preferred
-- was chosen as the target, and the bot charged empty ground.

Import-light on purpose (like fusion.py): no cv2, no numpy. The caller
injects `walkability_fn` (pathfinding.ray_walkability_score) so this stays
unit-testable.
"""

# Scoring weights, carried over verbatim from the original inline block.
_WALKABILITY_WEIGHT = 100.0
_DISTANCE_PENALTY = 0.3
_TIER_WEIGHT = 120.0
_RAY_SAMPLES = 12


def _distance(dot, center):
    return ((dot[0] - center[0]) ** 2 + (dot[1] - center[1]) ** 2) ** 0.5


def select_enemy_target(enemy_dots, center, cached_mask=None,
                        walkability_fn=None):
    """Return ``(x, y)`` of the best dot to attack, or None if there are none.

    Args:
        enemy_dots: ``(x, y, tier)`` entries -- already confirmed by the
            caller. Passing raw per-frame detections here is the bug this
            module exists to prevent.
        center: the player's ``(x, y)`` on the minimap.
        cached_mask: walkable-area mask, or None to score on tier/distance.
        walkability_fn: ``fn(mask, center, dot, samples=) -> 0..1``.

    Tier outranks walkability by design (120 vs 100), so walkability only
    decides between dots of the same tier.
    """
    if not enemy_dots:
        return None

    if len(enemy_dots) > 1 and cached_mask is not None and walkability_fn is not None:
        def _score(entry):
            dot = (entry[0], entry[1])
            return (walkability_fn(cached_mask, center, dot, samples=_RAY_SAMPLES)
                    * _WALKABILITY_WEIGHT
                    - _distance(dot, center) * _DISTANCE_PENALTY
                    + entry[2] * _TIER_WEIGHT)
        best = max(enemy_dots, key=_score)
    else:
        # Single candidate, or no mask to ray-cast against.
        best = max(enemy_dots,
                   key=lambda e: (e[2], -_distance((e[0], e[1]), center)))
    return (best[0], best[1])


def confirmed_enemy_target(tracker, dots, now, center, cached_mask=None,
                           walkability_fn=None):
    """Advance `tracker` one frame and pick a target from what it confirms.

    Returns ``(confirmed_dots, target_or_None)``. Keeping these two together
    is what stops them drifting apart again: there is no way to get a target
    without the confirmation step that produced it.
    """
    confirmed = tracker.update(dots, now)
    return confirmed, select_enemy_target(
        confirmed, center, cached_mask, walkability_fn)
