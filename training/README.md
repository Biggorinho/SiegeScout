# Training kit (dev-only — never ships)

Produces the local ONNX models consumed by `local_inference.OnnxDetector`.
Spec: `docs/superpowers/specs/2026-07-10-cv-recognition-improvements-design.md`
(§2 model pipeline, rollout step 4).

Contributors who want to improve detection should start with the
[contribution guide](../CONTRIBUTING.md). It covers safe data sharing,
evaluation expectations, and what to include in a model pull request.

## Minimap model

**Dataset:** our own
[Roboflow project](https://universe.roboflow.com/test-jmbou/minimapvision-2),
version 3, CC BY 4.0. It contains 76 images with a 44/24/8 split. The
training export is intentionally not stored in this repository; download or
export it from Roboflow when reproducing training.

The images are full 1920×1080 screenshots **downscaled to 800×600**, mostly
with the large zone-map overlay open; labeled map dots are ~5×7 px. Prep
therefore cuts **240×240 tiles** (4×3 overlapping grid, ~250 train tiles) and
trains at **480 input**: the 2× upscale puts objects at ~11–14 px, and the
same 2× ratio applies when the bot letterboxes its ~235×230 minimap crop at
inference, so training and runtime scales match. Whole-image training at 320
left objects at 2–3 px and scored mAP50 0.08 — do not go back to it. Mosaic
augmentation stays OFF for the same reason.

Same 76 source frames as the older `Downloads/minimapVision.v22i.yolov12`
export (whose 133 images were baked-augmentation duplicates — redundant since
ultralytics augments at train time) and the same 13-class schema. Prefer the
v3 zip: more unique train frames. If you re-export from Roboflow, check
`data.yaml` still matches the notebook's `RAW_NAMES`.

**Workflow:**

1. Have the export zip ready (already in the project root).
2. Open `train_minimap_colab.ipynb` in [Google Colab](https://colab.research.google.com)
   (upload it via File → Upload notebook), switch the runtime to a T4 GPU.
3. Run all cells top to bottom; upload the zip when asked. ~20–35 min.
4. The last cell downloads `minimap.onnx` → copy it to `assets/models/minimap.onnx`.
5. Score it against real labeled captures on the dev machine:
   `python -m eval.score --detector all`
6. Only set `LOCAL_MINIMAP_MODEL_ENABLED = True` in `config.py` if it beats
   the heuristic baseline by the acceptance gate (+10 F1 on its classes).

**Contracts between the notebook and `config.py` — keep in sync:**

| Notebook | config.py | Current value |
|---|---|---|
| `KEEP_CLASSES` (order = class ids) | `LOCAL_MINIMAP_MODEL_CLASSES` | dungeon, enter, exit, itemBlue, oreGold, oreGreen, oreRuby, player, portal |
| `IMGSZ` (static export size) | `LOCAL_MINIMAP_MODEL_INPUT_SIZE` | 480 |

**Classes dropped for now** (too few annotations; they train as background):
`itemAngel` (2), `oreBlue` (2), `itemSatan` (6), `oreCopper` (15). To add one
back: label more instances in Roboflow, re-export, add it to `KEEP_CLASSES`
**and** `LOCAL_MINIMAP_MODEL_CLASSES` (same position), retrain. Note the
dataset has **no enemy class** — minimap enemies stay heuristic-only.

**Known risks, measured by the eval harness (not here):**

- Domain gap: most labeled dots come from the downscaled large-map overlay;
  the bot detects on the small minimap crop, where dots render slightly
  differently. Scale matches (see above) but appearance may not — Colab mAP
  does not prove transfer, `eval.score` on real captures does.
- CPU budget: the spec wants ≤ 15 ms/tick; 480² on CPU will likely run
  25–45 ms. Watch `OnnxDetector.avg_inference_ms` — if it's too slow, either
  re-export the same weights with `IMGSZ = 320` (update the config value,
  expect some accuracy loss on the smallest dots) or run the minimap model
  async like the screen model when fusion lands (spec §3 watchdog covers the
  session-disable case).

## Screen model (`screen.onnx`)

Not started — no dataset yet. Config wiring exists
(`LOCAL_SCREEN_MODEL_*`, default off). Planned source per spec: auto-pre-label
captured frames with the best available detector, hand-correct in
`eval/label_tool.py`.

## Shipping note

`build.py` follows imports from `gui.py`, so `training/` and `eval/` never
reach the exe. When a model passes the gate, `assets/models/*.onnx` must be
added to the Nuitka `--include-data-files` list in `build.py` — weights are
not bundled automatically.
