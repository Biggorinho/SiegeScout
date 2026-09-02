"""Local (CPU) minimap training — trains on the user's real labeled minimap
crops and exports assets/models/minimap.onnx. No Colab required.

    unzip export -> keep eval_/train_ crops -> remap 14->9 classes ->
    split by prefix (train_ = train, eval_ = val) -> train YOLO11-nano ->
    export ONNX (opset 12, static 480) -> assets/models/minimap.onnx

The crops are native ~235x230 minimap captures (the bot's real resolution),
so NO tiling/upscaling is needed — train at 480 to match the bot's inference
letterbox. The eval_ crops are the held-out scoring set (their ground truth
also lives in eval/captures via labels_*.json); they go to val, never train.

Usage (from project root; dev-only, needs `pip install ultralytics`):
    python training/train_minimap_local.py
    python training/train_minimap_local.py --zip "some other export.zip"

The exported model stays INACTIVE until LOCAL_MINIMAP_MODEL_ENABLED is set in
config.py — flip it only after `python -m eval.score --detector all` shows it
beats the heuristic baseline.
"""

import argparse
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The combined Roboflow export also contains the 249 minimap crops
# (eval_/train_ prefixed); build_crop_dataset filters to just those.
DEFAULT_ZIP = ROOT / "minimapVision.yolov12.zip"
DATASETS_DIR = ROOT / "training" / "datasets"
RUNS_DIR = ROOT / "training" / "runs"
MODELS_DIR = ROOT / "assets" / "models"

# The 14-class Roboflow export schema, in data.yaml order.
RAW_NAMES = ["dungeon", "enemy", "enter", "exit", "itemAngel", "itemBlue",
             "itemSatan", "oreBlue", "oreCopper", "oreGold", "oreGreen",
             "oreRuby", "player", "portal"]
# The 9 classes with enough real-crop labels to learn. Order == the model's
# Collapse the ore and item subtypes, mirroring train_screen_local.SCREEN_MAP.
#
# The bot never reads the subtype -- fusion.ml_ore_dots stamps every ore class
# with the same tier, and ResourceTargeter never reads tier at all -- so the
# split bought nothing and cost a great deal. KEEP_CLASSES used to name the
# five survivors one by one, and `build_crop_dataset` drops any annotation
# whose class is not listed while KEEPING its image, so every unlisted label
# trained as background. On the v4 export that was 157 oreCopper, 13
# itemSatan, 5 itemBlue and 1 oreBlue across 65 of 445 crops -- 15% of the
# labeled data, teaching the model that copper veins and loot are nothing.
#
# Pooled instead: ore 449 (the second-largest class in the set), item 18 --
# thin, but on par with `enter` (15) and `dungeon` (32), which both train fine.
MINIMAP_MAP = {
    "enemy": "enemy", "player": "player", "portal": "portal",
    "dungeon": "dungeon", "enter": "enter", "exit": "exit",
    "oreGold": "ore", "oreGreen": "ore", "oreRuby": "ore",
    "oreBlue": "ore", "oreCopper": "ore",
    "itemBlue": "item", "itemSatan": "item", "itemAngel": "item",
}
# Order == the model's class ids == LOCAL_MINIMAP_MODEL_CLASSES in config.py.
# Changing this list without retraining and updating config mislabels every
# detection the bot makes, silently.
KEEP_CLASSES = ["enemy", "player", "portal", "dungeon", "enter", "exit",
                "ore", "item"]


def build_remap():
    """raw class id -> output class id, total over RAW_NAMES."""
    return {RAW_NAMES.index(raw): KEEP_CLASSES.index(dst)
            for raw, dst in MINIMAP_MAP.items()}

IMGSZ = 480  # static export size - must match LOCAL_MINIMAP_MODEL_INPUT_SIZE


def yolo_line_to_bbox(fields):
    """Parse one YOLO label line -> (cls_id, cx, cy, w, h), all normalized.

    Roboflow exported the hand-drawn crop annotations as *segmentation
    polygons* (class + 2N vertices), while the forked images are plain
    boxes (class + 4). Handle both: a polygon becomes its axis-aligned
    bounding box. Returns None for malformed lines.
    """
    if len(fields) < 5:
        return None
    cid = int(fields[0])
    coords = list(map(float, fields[1:]))
    if len(coords) == 4:
        cx, cy, w, h = coords
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs, ys = coords[0::2], coords[1::2]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        w, h = max(xs) - min(xs), max(ys) - min(ys)
    else:
        return None
    return cid, cx, cy, w, h


def build_crop_dataset(zip_path: Path) -> Path:
    """Extract the export and build a train/val crop dataset split by prefix."""
    remap = build_remap()
    tmp = DATASETS_DIR / "_export_extract"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)

    out = DATASETS_DIR / "minimap_crops"
    shutil.rmtree(out, ignore_errors=True)
    for s in ("train", "val"):
        (out / s / "images").mkdir(parents=True)
        (out / s / "labels").mkdir(parents=True)

    kept = Counter()
    moved = Counter()
    for srcsplit in ("train", "valid", "test"):
        img_dir = tmp / srcsplit / "images"
        if not img_dir.is_dir():
            continue
        for img in img_dir.iterdir():  # crops may be .jpg or .png per export
            base = img.name
            noext = img.stem
            if base.startswith("eval_"):
                dst = "val"
            elif base.startswith("train_"):
                dst = "train"
            else:
                continue  # skip fullscreen + forked 800x600 (different domain)
            rows = []
            lbl = tmp / srcsplit / "labels" / f"{noext}.txt"
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    parsed = yolo_line_to_bbox(line.split())
                    if parsed is None or parsed[0] not in remap:
                        continue
                    cid, cx, cy, w, h = parsed
                    rows.append(f"{remap[cid]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    kept[KEEP_CLASSES[remap[cid]]] += 1
            shutil.copy(img, out / dst / "images" / base)
            (out / dst / "labels" / f"{noext}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else ""))
            moved[dst] += 1

    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: train/images\nval: val/images\n"
        f"nc: {len(KEEP_CLASSES)}\nnames: {KEEP_CLASSES}\n")
    print(f"[prep] crops: {dict(moved)} | labels: "
          f"{dict(sorted(kept.items(), key=lambda kv: -kv[1]))}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", default=str(DEFAULT_ZIP))
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    data_dir = build_crop_dataset(Path(args.zip))

    from ultralytics import YOLO  # deferred: heavy import

    model = YOLO("yolo11n.pt")  # nano: smallest, fastest on the bot's CPU
    model.train(
        data=str(data_dir / "data.yaml"),
        imgsz=IMGSZ,
        epochs=args.epochs,
        patience=40,
        batch=16,
        seed=0,
        mosaic=0.0,    # mosaic quarters each image - lethal for tiny dots
        scale=0.3,     # mild zoom jitter only
        degrees=0.0,   # the minimap never rotates
        hsv_h=0.01,    # classes are told apart by HUE (item rarity, enemy
                       # red, portal blue) - keep jitter tiny
        workers=2,
        project=str(RUNS_DIR),
        name="minimap",
        exist_ok=True,
        plots=True,
    )
    best = YOLO(str(model.trainer.best))
    metrics = best.val(data=str(data_dir / "data.yaml"), imgsz=IMGSZ)
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        print(f"[val] {KEEP_CLASSES[int(cls_idx)]:<9} AP50={metrics.box.ap50[i]:.3f}")
    print(f"[val] mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")

    onnx_path = best.export(format="onnx", opset=12, imgsz=IMGSZ,
                            dynamic=False, simplify=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / "minimap.onnx"
    shutil.copy(onnx_path, target)
    print(f"[export] {target} ({os.path.getsize(target) / 1e6:.1f} MB)")
    print("[export] now score it: python -m eval.score --detector all")


if __name__ == "__main__":
    main()
