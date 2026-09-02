"""The minimap trainer's class remap — where silent data corruption lives.

Class ids in a YOLO label file are positional, so a remap that is wrong or
incomplete does not raise: it trains the model on mislabeled boxes, or turns
labeled objects into background. Both failures are invisible until the model
is live. These tests pin the invariants instead.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train_minimap_local import KEEP_CLASSES, MINIMAP_MAP, RAW_NAMES, build_remap


def test_every_raw_class_is_mapped_somewhere():
    """An unmapped class is dropped from its label file while its IMAGE stays
    in the training set, so the object trains as background. That is how the
    v4 export's 157 oreCopper and all 18 item annotations -- 15% of the
    labeled crops -- would have taught the model that copper veins and loot
    are nothing."""
    unmapped = [n for n in RAW_NAMES if n not in MINIMAP_MAP]
    assert unmapped == [], f"these raw classes would train as background: {unmapped}"


def test_map_targets_are_all_real_output_classes():
    for raw, dst in MINIMAP_MAP.items():
        assert dst in KEEP_CLASSES, f"{raw} maps to {dst}, which is not an output class"


def test_ore_subtypes_collapse_to_one_class():
    """The bot never reads the ore subtype: fusion.ml_ore_dots gives every ore
    class the same tier, and ResourceTargeter never reads tier at all. Split
    across five names they were thin enough that two fell out of training;
    pooled they are the second-largest class in the set."""
    ore_ids = {build_remap()[RAW_NAMES.index(n)]
               for n in RAW_NAMES if n.startswith("ore")}
    assert len(ore_ids) == 1
    assert KEEP_CLASSES[ore_ids.pop()] == "ore"


def test_item_subtypes_collapse_to_one_class():
    item_ids = {build_remap()[RAW_NAMES.index(n)]
                for n in RAW_NAMES if n.startswith("item")}
    assert len(item_ids) == 1
    assert KEEP_CLASSES[item_ids.pop()] == "item"


def test_remap_is_total_over_the_raw_schema():
    remap = build_remap()
    assert sorted(remap) == list(range(len(RAW_NAMES)))
    assert all(0 <= v < len(KEEP_CLASSES) for v in remap.values())


def test_output_classes_match_the_shipped_config():
    """The class list is positional: the ONNX model emits indices, and
    `OnnxDetector` names them from this list. Retraining with a different
    KEEP_CLASSES than config carries silently mislabels every detection --
    ore read as enemy, item as portal. The two must move together, which is
    exactly what this test forces."""
    import config
    assert list(config.LOCAL_MINIMAP_MODEL_CLASSES) == list(KEEP_CLASSES)
