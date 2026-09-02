"""Rank the full-screen captures already on disk by how much labeling them
would teach the on-screen model, and stage the winners for Roboflow.

Why this exists: the July-2026 export gave `screen.onnx` 214 world-size ore
annotations and TEN item annotations, which is exactly the ratio the bot
shows in play -- mining works, looting does not. The fix is more *item*
labels, and there is no need to farm new footage for them: eval/captures and
loot_frames already hold ~3.5k unlabeled 1920x1080 frames. This finds the
ones worth a human's time.

Selection is uncertainty-first (active learning): a frame the model already
nails at 0.9 teaches it nothing, and a frame with no detection at any
threshold is probably empty. The interesting band is in between.

    python tools/build_label_queue.py                # scan + stage top 250
    python tools/build_label_queue.py --top 400 --class item
"""

import argparse
import csv
import glob
import os
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from local_inference import OnnxDetector  # noqa: E402

# Well below LOCAL_SCREEN_MODEL_CONF: we want the model's hesitant guesses,
# which are the frames whose labels would move it most.
SCAN_CONF = 0.05
OUT_DIR = ROOT / "training" / "label_queue"


def frame_paths():
    seen = []
    seen += sorted(glob.glob(str(ROOT / "loot_frames" / "*.png")))
    seen += sorted(glob.glob(str(ROOT / "eval" / "captures" / "*" / "frame_*.png")))
    return seen


def score_frame(dets, cls):
    """Uncertainty score for one class: peak confidence, and how many boxes.

    A frame scores highest when the model saw something but is unsure of it
    (0.05-0.55). Confident hits sink -- they are already learned -- and
    frames with nothing at all sink to zero.
    """
    boxes = dets.get(cls) or []
    if not boxes:
        return 0.0, 0
    peak = max(b["confidence"] for b in boxes)
    if peak >= 0.55:
        uncertainty = 0.25          # already confident; low teaching value
    else:
        uncertainty = 1.0 - abs(peak - 0.30) / 0.30
    return round(max(0.0, uncertainty) * peak, 4), len(boxes)


def staged_name(src_path):
    """Stage under a `lootframe_` prefix so the screen trainer accepts it.

    `train_screen_local._is_screen_frame` selects images by filename prefix,
    and eval/captures writes its full frames as `frame_<stamp>.png`. Those are
    the same domain as loot frames -- 1920x1080 in-game screens -- but the
    prefix alone would make the trainer skip every one of them, so labeling
    them would be wasted work. Renaming on the way out keeps provenance in the
    name while putting them in the set the trainer actually reads.
    """
    base = os.path.basename(src_path)
    if base.startswith("lootframe"):
        return base
    return "lootframe_evalcap_" + base.split("_", 1)[-1]


def stage(rows, key, top, cls):
    stage_dir = OUT_DIR / cls
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    for r in rows[:top]:
        if float(r[key]) <= 0:
            break
        shutil.copy2(r["path"], stage_dir / staged_name(r["path"]))
    n = len(list(stage_dir.glob("*.png")))
    print(f"staged {n} frames -> {stage_dir}")
    print(f"drag that folder into the Roboflow project and label the {cls} labels.")


MINIMAP_GLOB = "eval/captures/*/minimap_*.png"
# The minimap trainer splits train/val purely by filename prefix, so staging
# has to write the split it wants (train_minimap_local.build_crop_dataset).
VAL_EVERY = 5          # every 5th staged crop goes to the eval_ split
MIN_SECONDS_APART = 20  # captures are ~2 s apart and near-identical


def _looks_like_a_minimap(img):
    """Reject frames captured while the minimap was not on screen.

    The capture path writes a crop of the minimap rectangle every tick, so
    loading screens, menus and the world-map screen land in the same folder.
    The real minimap renders on a near-black ground in every biome measured
    (median grey 2 across 481 sampled crops, max 24); the rejects sit at 47
    (world map) and 79-93 (loading art). Checked that the bright ones are not
    a snow biome -- they are cutscene and menu screens.
    """
    import numpy as np
    return float(np.median(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))) < 40.0


def minimap_candidates(top):
    """Rank minimap crops by how likely they hold an unlabeled item star.

    There is no ML proposer for this one -- the whole problem is that the
    minimap model has no item class -- so the heuristic `classify_item_dots`
    stands in. It is noisy (a third of its dots land in the player-label box,
    and it fires on magenta wall linework), but a human labeling the frame
    filters that in a glance; what matters is that it almost never misses a
    real star, so it is a usable candidate finder even though it is a poor
    detector.

    Frames are spaced out in time because consecutive captures are ~2 s apart
    and near-identical -- 250 crops of one corridor teach nothing.
    """
    import config
    from detectors import classify_item_dots

    dxlo, dxhi = config.PLAYER_LABEL_DX
    dylo, dyhi = config.PLAYER_LABEL_DY
    scored = []
    paths = sorted(glob.glob(str(ROOT / MINIMAP_GLOB)))
    print(f"scanning {len(paths)} minimap crops ...")
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            continue
        if not _looks_like_a_minimap(img):
            continue
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
        dots = [d for d in classify_item_dots(img)
                if not (dxlo <= d[0] - cx <= dxhi and dylo <= d[1] - cy <= dyhi)]
        if not dots:
            continue
        # 1-3 candidates is a plausible scene; a dozen is wall linework.
        score = 1.0 if 1 <= len(dots) <= 3 else 0.4
        scored.append((score, p))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(paths)}")

    scored.sort(key=lambda t: -t[0])
    picked, staged_stamps = [], {}
    for score, p in scored:
        session = Path(p).parent.name
        try:
            stamp = int(Path(p).stem.split("_")[-1]) / 1000.0
        except ValueError:
            stamp = 0.0
        # Compare against EVERY frame already staged from this session, not
        # just the most recent one: `scored` is in score order, not time
        # order, so remembering only the last stamp lets a frame 5 s from an
        # earlier pick slip through behind a far-apart one.
        near = staged_stamps.setdefault(session, [])
        if any(abs(stamp - s2) < MIN_SECONDS_APART for s2 in near):
            continue
        near.append(stamp)
        picked.append(p)
        if len(picked) >= top:
            break
    print(f"crops with a candidate star: {len(scored)}/{len(paths)}"
          f"  -> {len(picked)} kept after {MIN_SECONDS_APART}s spacing")
    return picked


def stage_minimap(paths):
    out = OUT_DIR / "minimap"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for i, p in enumerate(paths):
        prefix = "eval" if i % VAL_EVERY == 0 else "train"
        shutil.copy2(p, out / f"{prefix}_{Path(p).stem.split('_')[-1]}.png")
    n = len(list(out.glob("*.png")))
    v = len(list(out.glob("eval_*.png")))
    print(f"staged {n} crops -> {out}  ({n - v} train_ / {v} eval_)")
    print("drag that folder into Roboflow and label ONLY the item stars.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=250, help="frames to stage")
    ap.add_argument("--class", dest="cls", default="item", choices=["item", "ore"])
    ap.add_argument("--no-copy", action="store_true", help="write the CSV only")
    ap.add_argument("--from-csv", action="store_true",
                    help="re-stage from an existing ranked.csv, skipping the scan")
    ap.add_argument("--source", default="screen", choices=["screen", "minimap"],
                    help="screen = full loot frames; minimap = minimap crops")
    args = ap.parse_args()

    if args.source == "minimap":
        stage_minimap(minimap_candidates(args.top))
        return

    key = f"{args.cls}_score"
    if args.from_csv:
        with open(OUT_DIR / "ranked.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        rows.sort(key=lambda r: float(r[key]), reverse=True)
        stage(rows, key, args.top, args.cls)
        return

    det = OnnxDetector(config.LOCAL_SCREEN_MODEL_PATH,
                       config.LOCAL_SCREEN_MODEL_CLASSES,
                       config.LOCAL_SCREEN_MODEL_INPUT_SIZE,
                       conf_threshold=SCAN_CONF)
    if not det.available:
        sys.exit(f"screen model not loadable at {config.LOCAL_SCREEN_MODEL_PATH}")

    paths = frame_paths()
    print(f"scanning {len(paths)} frames at conf>={SCAN_CONF} ...")
    rows = []
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            continue
        d = det.detect(img)
        i_score, i_n = score_frame(d, "item")
        o_score, o_n = score_frame(d, "ore")
        rows.append({"path": p, "item_score": i_score, "item_boxes": i_n,
                     "ore_score": o_score, "ore_boxes": o_n})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(paths)}")

    rows.sort(key=lambda r: r[key], reverse=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "ranked.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    nonzero = [r for r in rows if r[key] > 0]
    print(f"\nframes with any {args.cls} signal: {len(nonzero)}/{len(rows)}")
    print(f"ranked CSV -> {csv_path}")

    if args.no_copy:
        return
    stage(rows, key, args.top, args.cls)


if __name__ == "__main__":
    main()
