"""Local (CPU) on-screen resource model — trains screen.onnx from the user's
labeled full-screen loot frames. Replaces the noisy find_ore_nodes_on_screen
heuristic (which over-detects 50-90 phantom "ore nodes" and makes looting
click randomly).

    unzip export -> keep only lootframe_* (full 1920x1080) images ->
    collapse ore subtypes -> `ore`, item subtypes -> `item` ->
    train YOLO11-nano @640 -> export ONNX -> assets/models/screen.onnx

Unlike the minimap dots, on-screen ore veins are LARGE (~50-100 px), so the
full frame trains fine at 640 with no tiling. Null (background) loot frames
are kept as negatives to suppress false positives. The minimap crops and old
frames in the same export are ignored here (they belong to the minimap model).

Usage (from project root; dev-only, needs `pip install ultralytics`):
    python training/train_screen_local.py
    python training/train_screen_local.py --zip "some other export.zip"

screen.onnx is live by default (LOCAL_SCREEN_MODEL_ENABLED in config.py).
Retraining overwrites the shipped weights, so re-check mAP50 and eyeball a few
frames before keeping a new export; set the flag False to fall back to the
heuristic.
"""

import argparse
import os
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_minimap_local import yolo_line_to_bbox  # polygon/box -> bbox

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP = ROOT / "minimapVision.yolov12.zip"
DATASETS_DIR = ROOT / "training" / "datasets"
RUNS_DIR = ROOT / "training" / "runs"
MODELS_DIR = ROOT / "assets" / "models"

# 14-class Roboflow schema, in data.yaml order.
RAW_NAMES = ["dungeon", "enemy", "enter", "exit", "itemAngel", "itemBlue",
             "itemSatan", "oreBlue", "oreCopper", "oreGold", "oreGreen",
             "oreRuby", "player", "portal"]
# Collapse every ore/item subtype to two on-screen classes. For clicking the
# subtype does not matter, and one `ore` class pools all 214 examples.
SCREEN_MAP = {
    "oreGold": "ore", "oreGreen": "ore", "oreRuby": "ore",
    "oreBlue": "ore", "oreCopper": "ore",
    "itemBlue": "item", "itemSatan": "item", "itemAngel": "item",
}
# Order == the model's class ids == LOCAL_SCREEN_MODEL_CLASSES in config.py.
SCREEN_CLASSES = ["ore", "item"]

IMGSZ = 640  # full frame; must match LOCAL_SCREEN_MODEL_INPUT_SIZE


def _is_screen_frame(basename: str) -> bool:
    return basename.lower().startswith("lootframe")


def build_screen_dataset(zip_path: Path, val_frac: float = 0.15,
                         seed: int = 0) -> Path:
    """Extract the export, keep only lootframe_* images, remap to ore/item.

    Roboflow's own train/valid/test split can dump every labeled frame into
    train (leaving an all-background val set that can't be scored). So ignore
    it: pool all loot frames and do our own seeded split, stratified so both
    labeled and background frames land in train AND val.
    """
    import random

    tmp = DATASETS_DIR / "_screen_extract"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)

    to_id = {c: i for i, c in enumerate(SCREEN_CLASSES)}
    entries = []  # (img_path, [label rows])
    for srcsplit in ("train", "valid", "test"):
        img_dir = tmp / srcsplit / "images"
        if not img_dir.is_dir():
            continue
        for img in img_dir.iterdir():
            if not _is_screen_frame(img.name):
                continue  # minimap crops / old frames belong to the other model
            rows = []
            lbl = tmp / srcsplit / "labels" / f"{img.stem}.txt"
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    parsed = yolo_line_to_bbox(line.split())
                    if parsed is None:
                        continue
                    screen_cls = SCREEN_MAP.get(RAW_NAMES[parsed[0]])
                    if screen_cls is None:
                        continue
                    rows.append(f"{to_id[screen_cls]} {parsed[1]:.6f} "
                                f"{parsed[2]:.6f} {parsed[3]:.6f} {parsed[4]:.6f}")
            entries.append((img, rows))

    rng = random.Random(seed)
    labeled = [e for e in entries if e[1]]
    background = [e for e in entries if not e[1]]
    rng.shuffle(labeled)
    rng.shuffle(background)

    def _split(lst):
        k = min(len(lst) - 1, max(1, round(len(lst) * val_frac))) if lst else 0
        return lst[k:], lst[:k]  # (train, val)

    tr_l, va_l = _split(labeled)
    tr_b, va_b = _split(background)
    assign = {"train": tr_l + tr_b, "val": va_l + va_b}

    out = DATASETS_DIR / "screen_frames"
    shutil.rmtree(out, ignore_errors=True)
    kept = Counter()
    moved = Counter()
    bg = Counter()
    for dst, items in assign.items():
        (out / dst / "images").mkdir(parents=True)
        (out / dst / "labels").mkdir(parents=True)
        for img, rows in items:
            shutil.copy(img, out / dst / "images" / img.name)
            (out / dst / "labels" / f"{img.stem}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else ""))
            moved[dst] += 1
            if rows:
                for r in rows:
                    kept[SCREEN_CLASSES[int(r.split()[0])]] += 1
            else:
                bg[dst] += 1

    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: train/images\nval: val/images\n"
        f"nc: {len(SCREEN_CLASSES)}\nnames: {SCREEN_CLASSES}\n")
    print(f"[prep] train {moved['train']} ({bg['train']} bg) | "
          f"val {moved['val']} ({bg['val']} bg) | labels: {dict(kept)}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", default=str(DEFAULT_ZIP))
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    data_dir = build_screen_dataset(Path(args.zip))

    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(data_dir / "data.yaml"),
        imgsz=IMGSZ,
        epochs=args.epochs,
        patience=40,
        batch=16,
        seed=0,
        degrees=0.0,   # the game view never rotates
        workers=2,
        project=str(RUNS_DIR),
        name="screen",
        exist_ok=True,
        plots=True,
    )
    best = YOLO(str(model.trainer.best))
    metrics = best.val(data=str(data_dir / "data.yaml"), imgsz=IMGSZ)
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        print(f"[val] {SCREEN_CLASSES[int(cls_idx)]:<6} AP50={metrics.box.ap50[i]:.3f}")
    print(f"[val] mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")

    onnx_path = best.export(format="onnx", opset=12, imgsz=IMGSZ,
                            dynamic=False, simplify=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / "screen.onnx"
    shutil.copy(onnx_path, target)
    print(f"[export] {target} ({os.path.getsize(target) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
