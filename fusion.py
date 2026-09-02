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


# The retrained minimap model emits one collapsed `ore` class; the per-colour
# names are the pre-2026-09-03 export and stay accepted so a restored older
# model still works. Downstream only ever cared that it is ore.
_ORE_CLASSES = ("ore", "oreGold", "oreGreen", "oreRuby")
# Tier used by detectors.classify_item_dots for its own star dots.
_ITEM_TIER = 1
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


def ml_item_tags(ml_dets, min_conf=0.40, merge_dist=8.0):
    """On-screen loot labels from the screen model, shaped like
    `detectors.find_item_tags_on_screen` output — ``(x, y, score)`` tuples,
    strongest first.

    Replaces the heuristic for the same reason `ml_ore_dots` does, only more
    starkly: `find_item_tags_on_screen` masks any red/green/cyan blob, and
    over the 300 captured loot frames it returned a mean of 93 "tags" per
    frame — every one of the 300 frames had at least 10. Both pickup sites
    click the tag nearest screen centre, so what got clicked was whatever fire
    particle, enemy aura or HUD glyph happened to sit near the crosshair while
    the real label sat at the edge of the screen. Clicking a label that never
    resolves keeps it on screen, which is the `explore_stall_loot_tag` signal
    that dominated the last session report (598 stalled frames, 2 items
    actually collected).

    The screen model carries an `item` class trained on the user's own loot
    frames and was already being loaded — it was just never asked about items,
    only ore.
    """
    if not ml_dets:
        return []
    tags = []
    md2 = merge_dist * merge_dist
    for det in sorted(ml_dets.get("item") or (),
                      key=lambda d: d.get("confidence", 0.0), reverse=True):
        conf = det.get("confidence", 0.0)
        if conf < min_conf:
            continue
        x, y = int(round(det["x"])), int(round(det["y"]))
        # One label can trip two boxes; keep the stronger, or the bot clicks
        # the same loot twice and cancels its own auto-walk.
        if any((x - tx) ** 2 + (y - ty) ** 2 <= md2 for tx, ty, _ in tags):
            continue
        tags.append((x, y, float(conf)))
    return tags


def ml_item_dots(ml_dets, min_conf=0.40, merge_dist=4.0, tier=_ITEM_TIER,
                 kind="ml"):
    """Minimap item stars from the model, shaped like classify_item_dots.

    Replaces the heuristic rather than adding to it, for the same reason as
    `ml_ore_dots`: `classify_item_dots` is a precision problem, not a recall
    one. Over the 400 minimaps of session_20260827-062546 it put 36% of its
    dots inside the player-label box -- magenta wall linework and the player's
    own blue marker -- and since ResourceTargeter picks the NEAREST dot, false
    positives near the player out-compete real loot anywhere else on the map.
    Adding the model's dots on top would keep every one of those.

    Caveat worth knowing: the item class trained on 10 annotations and scores
    AP50 0.875 (P 1.000, R 0.835) on 8 validation instances. The precision is
    what this replacement is buying, and it is the half that held; the recall
    figure rests on 8 boxes and should not be trusted until there are more
    labels. `python -m eval.score` on real captures is the honest gate.
    """
    if not ml_dets:
        return []
    dots = []
    md2 = merge_dist * merge_dist
    for det in ml_dets.get("item") or ():
        if det.get("confidence", 0.0) < min_conf:
            continue
        x, y = det["x"], det["y"]
        if any((x - d[0]) ** 2 + (y - d[1]) ** 2 <= md2 for d in dots):
            continue
        dots.append([round(x), round(y), tier, kind])
    return dots
