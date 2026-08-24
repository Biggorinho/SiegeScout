"""Fusion of local-ML minimap detections with the OpenCV heuristics.

Spec §3 (docs/superpowers/specs/2026-07-10-cv-recognition-improvements-
design.md). The minimap model's biggest win is enemy recall — the heuristic
`classify_enemy_dots` misses ~⅔ of enemies on the eval set (recall 0.36 vs
the model's 0.74), which is what starves the vote-reset combat gate.

This first integration is deliberately **additive**: ML enemy detections
fill in enemies the heuristic *missed*, but never remove a heuristic dot.
That captures the recall win while staying strictly safer than the heuristic
alone (the temporal-decay suppression of heuristic false positives from the
spec is left for a later pass, once the additive path is proven live).

Enemy tier is not something the model predicts, so ML-only dots get a fixed
low tier (they still count for the combat gate and get targeted, just not
prioritized over color-classified higher tiers) — matching spec §3's rule
that tiers stay heuristic-derived.
"""

# Lowest enemy tier (detectors.ENEMY_TIER_RED); duplicated as a plain int so
# this module stays import-light and unit-testable without cv2/detectors.
_DEFAULT_ML_TIER = 1


def fuse_enemy_dots(heuristic_dots, ml_enemy_dets, merge_dist=6.0,
                    min_conf=0.65, default_tier=_DEFAULT_ML_TIER):
    """Return heuristic enemy dots augmented with ML-only detections.

    Args:
        heuristic_dots: list of ``[x, y, tier, kind]`` from
            ``detectors.classify_enemy_dots``.
        ml_enemy_dets: list of ``{"x","y","confidence",...}`` dicts for the
            model's ``enemy`` class (``OnnxDetector.detect(...)["enemy"]``).
        merge_dist: an ML detection within this many px of an already-present
            dot is considered the same entity and dropped (heuristic wins).
        min_conf: ML detections below this confidence are ignored.
        default_tier: tier assigned to accepted ML-only dots.

    Returns:
        A new list of ``[x, y, tier, kind]``; heuristic dots are preserved
        as-is and ML-only dots are appended with ``kind == "ml"``. The input
        list is never mutated.
    """
    fused = [list(d) for d in heuristic_dots]
    md2 = merge_dist * merge_dist
    for det in ml_enemy_dets:
        if det.get("confidence", 0.0) < min_conf:
            continue
        x, y = det["x"], det["y"]
        if any((x - d[0]) ** 2 + (y - d[1]) ** 2 <= md2 for d in fused):
            continue  # already covered by a heuristic (or accepted ML) dot
        fused.append([round(x), round(y), default_tier, "ml"])
    return fused


# The model splits ore by vein colour; downstream only cares that it is ore.
_ORE_CLASSES = ("oreGold", "oreGreen", "oreRuby")
# Tier used by detectors.classify_ore_dots for its own ore dots.
_ORE_TIER = 3


def ml_ore_dots(ml_dets, min_conf=0.40, merge_dist=4.0, tier=_ORE_TIER,
                kind="ml"):
    """Ore dots from the minimap model, shaped like classify_ore_dots output.

    Unlike `fuse_enemy_dots` this REPLACES the heuristic rather than adding to
    it, because the two classes fail in opposite directions. Enemies were a
    recall problem, so adding was strictly safer. Ore is a precision problem:
    on the 102-frame benchmark the heuristic scores P 0.28 / R 0.32 with 81
    false positives, against the model's P 0.93 / R 0.90 with 7. Adding the
    model's dots to the heuristic's would keep all 81 -- making the very
    symptom worse. The model wins on both axes, so it stands alone.

    Returns ``[x, y, tier, kind]`` entries with integer coordinates, matching
    the heuristic's shape so callers need no special handling.
    """
    if not ml_dets:
        return []
    dots = []
    md2 = merge_dist * merge_dist
    for cls in _ORE_CLASSES:
        for det in ml_dets.get(cls) or ():
            if det.get("confidence", 0.0) < min_conf:
                continue
            x, y = det["x"], det["y"]
            # One vein can trip two colour classes at once; keep it one dot,
            # or the bot mines the same spot twice.
            if any((x - d[0]) ** 2 + (y - d[1]) ** 2 <= md2 for d in dots):
                continue
            dots.append([round(x), round(y), tier, kind])
    return dots
