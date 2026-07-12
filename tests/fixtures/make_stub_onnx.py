"""Regenerate stub_yolo.onnx — a ~1 kB fake YOLO head for OnnxDetector tests.

Run once and check in the result (needs the dev-only `onnx` package):

    python tests/fixtures/make_stub_onnx.py

The graph consumes a [1, 3, 64, 64] image and returns a CONSTANT ultralytics-
style output (1, 4+nc, N) regardless of pixels, so tests exercise the real
onnxruntime session plus OnnxDetector's letterbox/decode/NMS deterministically.
(The input feeds the output via `out = const + 0 * sum(input)` because ONNX
disallows dangling graph inputs in strict checkers.)

Layout: nc = 3 classes ("a", "b", "c"), N = 6 anchors, input size 64.
Anchor columns (cx, cy, w, h in 64x64 letterbox space, then 3 class scores):

    1. class-a box, conf .90            -> kept
    2. same box shifted 1px, conf .80   -> NMS-suppressed by 1
    3. same box as 1 but class b, .85   -> kept (class-aware NMS)
    4. low-confidence box, .20          -> conf-filtered
    5. class-c corner box, .70          -> kept
    6. all zeros (background)           -> conf-filtered
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper

ANCHORS = np.array([
    # cx,  cy,   w,   h,    a,    b,    c
    [32.0, 32.0, 20.0, 10.0, 0.90, 0.05, 0.05],
    [33.0, 33.0, 20.0, 10.0, 0.80, 0.05, 0.05],
    [32.0, 32.0, 20.0, 10.0, 0.05, 0.85, 0.05],
    [40.0, 40.0, 8.0, 8.0, 0.20, 0.10, 0.05],
    [8.0, 20.0, 8.0, 6.0, 0.05, 0.05, 0.70],
    [0.0, 0.0, 0.0, 0.0, 0.00, 0.00, 0.00],
], dtype=np.float32)

INPUT_SIZE = 64


def build():
    pred = ANCHORS.T[np.newaxis]  # (1, 7, 6) — ultralytics (1, 4+nc, N)
    nodes = [
        helper.make_node("ReduceSum", ["images"], ["pix_sum"], keepdims=0),
        helper.make_node("Mul", ["pix_sum", "zero"], ["zero_scalar"]),
        helper.make_node("Add", ["pred", "zero_scalar"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes, "stub_yolo",
        inputs=[helper.make_tensor_value_info(
            "images", TensorProto.FLOAT, [1, 3, INPUT_SIZE, INPUT_SIZE])],
        outputs=[helper.make_tensor_value_info(
            "output0", TensorProto.FLOAT, list(pred.shape))],
        initializer=[
            helper.make_tensor("pred", TensorProto.FLOAT, pred.shape,
                               pred.flatten().tolist()),
            helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0]),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 12)], ir_version=8)
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "stub_yolo.onnx")
    onnx.save(build(), out_path)
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")
