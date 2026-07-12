"""Local ONNX inference — the shipped, no-cloud detection path.

OnnxDetector wraps a YOLO-nano model exported by the training kit
(training/train_minimap_colab.ipynb: ultralytics export, opset 12, static
input, NMS kept out of the graph) and returns detections in the same shape
as the Roboflow wrappers in detectors.py, so downstream consumers are
backend-agnostic:

    detect(bgr) -> {class_name: [{"x", "y", "width", "height",
                                  "confidence", "class"}, ...]}

x/y are box centers in the *input image's* pixel space (letterboxing is
undone internally), matching the Roboflow prediction convention.

Fails soft everywhere: missing weights, missing onnxruntime, or a broken
model leave `available` False and `detect()` returning {} — the bot then
behaves exactly as it does today on pure heuristics. Errors are logged once,
never raised into the bot loop.
"""

import os
import time
from collections import deque

import cv2
import numpy as np

# Padding color for letterboxing (ultralytics convention: grey 114).
_PAD_VALUE = 114


class OnnxDetector:
    """CPU object detector over a YOLO-style ONNX model.

    Args:
        model_path:     path to the .onnx file (e.g. assets/models/minimap.onnx)
        classes:        class names in the model's training order — must match
                        the `names:` list baked by the training notebook.
        input_size:     static square input size the model was exported with.
        conf_threshold: minimum class confidence to keep a detection.
        iou_threshold:  IoU above which overlapping same-class boxes are merged
                        by NMS.
    """

    def __init__(self, model_path, classes, input_size,
                 conf_threshold=0.40, iou_threshold=0.45):
        self.model_path = str(model_path)
        self.classes = list(classes)
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.available = False
        # Rolling per-run inference times (ms) — consumed by the step-5
        # scheduling watchdog; harmless to record before that lands.
        self.inference_times = deque(maxlen=20)
        self._session = None
        self._input_name = None
        self._detect_error_logged = False
        try:
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(f"model file not found: {self.model_path}")
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.log_severity_level = 3  # errors only; ort warns on CPU EP otherwise
            self._session = ort.InferenceSession(
                self.model_path, sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self.available = True
        except Exception as e:
            # Fail soft: report unavailable, bot continues on heuristics.
            print(f"[ONNX] {os.path.basename(self.model_path)} unavailable: "
                  f"{type(e).__name__}: {e}")

    @property
    def avg_inference_ms(self):
        """Rolling average inference time in ms, or None before the first run."""
        if not self.inference_times:
            return None
        return sum(self.inference_times) / len(self.inference_times)

    def detect(self, bgr):
        """Run the model over a BGR frame. Returns {} when unavailable."""
        if not self.available or bgr is None or getattr(bgr, "size", 0) == 0:
            return {}
        try:
            t0 = time.perf_counter()
            blob, scale, pad_x, pad_y = self._letterbox(bgr)
            raw = self._session.run(None, {self._input_name: blob})[0]
            h, w = bgr.shape[:2]
            detections = self._decode(raw, scale, pad_x, pad_y, w, h)
            self.inference_times.append((time.perf_counter() - t0) * 1000.0)
            return detections
        except Exception as e:
            if not self._detect_error_logged:
                self._detect_error_logged = True
                print(f"[ONNX] {os.path.basename(self.model_path)} detect error "
                      f"(logged once): {type(e).__name__}: {e}")
            return {}

    # ── preprocess ──────────────────────────────────────────────────────────

    def _letterbox(self, bgr):
        """Resize keeping aspect, pad to a grey square, HWC-BGR → NCHW-RGB float.

        Returns (blob, scale, pad_x, pad_y) where scale/pads map original
        pixel coords into letterbox space: lb = orig * scale + pad.
        """
        size = self.input_size
        h, w = bgr.shape[:2]
        scale = min(size / float(w), size / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), _PAD_VALUE, dtype=np.uint8)
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        return np.ascontiguousarray(blob), scale, pad_x, pad_y

    # ── postprocess ─────────────────────────────────────────────────────────

    def _decode(self, raw, scale, pad_x, pad_y, orig_w, orig_h):
        """Decode a YOLOv8+ head output into Roboflow-shaped detections.

        Ultralytics exports (1, 4+nc, N): rows cx,cy,w,h in letterbox pixels
        followed by nc class scores (no separate objectness). Accepts the
        transposed layout too.
        """
        out = np.squeeze(np.asarray(raw))
        if out.ndim != 2:
            return {name: [] for name in self.classes}
        nc = len(self.classes)
        if out.shape[0] == 4 + nc and out.shape[1] != 4 + nc:
            out = out.T  # → (N, 4+nc)
        if out.shape[1] != 4 + nc:
            raise ValueError(
                f"model output shape {out.shape} does not match "
                f"{nc} configured classes (expected N x {4 + nc})")

        scores = out[:, 4:4 + nc]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        keep = confidences >= self.conf_threshold
        if not np.any(keep):
            return {name: [] for name in self.classes}

        boxes = out[keep, :4].astype(np.float64)  # cx, cy, w, h (letterbox px)
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        # Letterbox → original image space.
        boxes[:, 0] = (boxes[:, 0] - pad_x) / scale
        boxes[:, 1] = (boxes[:, 1] - pad_y) / scale
        boxes[:, 2] /= scale
        boxes[:, 3] /= scale

        # Corner form for NMS, clipped to the frame.
        x1 = np.clip(boxes[:, 0] - boxes[:, 2] / 2, 0, orig_w - 1)
        y1 = np.clip(boxes[:, 1] - boxes[:, 3] / 2, 0, orig_h - 1)
        x2 = np.clip(boxes[:, 0] + boxes[:, 2] / 2, 0, orig_w - 1)
        y2 = np.clip(boxes[:, 1] + boxes[:, 3] / 2, 0, orig_h - 1)

        # Class-aware NMS via the coordinate-offset trick: shift each class
        # into its own disjoint region so boxes never cross-suppress. Only the
        # copies handed to _nms are shifted; x1..y2 stay in image space.
        offset = class_ids.astype(np.float64) * (max(orig_w, orig_h) + 1.0)
        keep_idx = _nms(x1 + offset, y1 + offset, x2 + offset, y2 + offset,
                        confidences, self.iou_threshold)

        detections = {name: [] for name in self.classes}
        for i in keep_idx:
            name = self.classes[class_ids[i]]
            detections[name].append({
                "x": float((x1[i] + x2[i]) / 2),
                "y": float((y1[i] + y2[i]) / 2),
                "width": float(x2[i] - x1[i]),
                "height": float(y2[i] - y1[i]),
                "confidence": float(confidences[i]),
                "class": name,
            })
        return detections


def _nms(x1, y1, x2, y2, scores, iou_threshold):
    """Plain numpy non-maximum suppression; returns kept indices."""
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return keep
