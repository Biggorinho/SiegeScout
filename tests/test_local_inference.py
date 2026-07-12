"""OnnxDetector pre/post-processing against the checked-in stub model.

stub_yolo.onnx (see tests/fixtures/make_stub_onnx.py) returns a constant
6-anchor prediction for a 64x64 input, 3 classes — the anchor table there
documents which boxes must survive confidence filtering and class-aware NMS.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from local_inference import OnnxDetector, _nms

STUB_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "stub_yolo.onnx")
STUB_CLASSES = ["a", "b", "c"]


def make_detector(**kwargs):
    kwargs.setdefault("conf_threshold", 0.40)
    return OnnxDetector(STUB_PATH, STUB_CLASSES, 64, **kwargs)


# Test frame: 32x16 -> letterbox scale 2.0, pad_x 0, pad_y 16.
FRAME = np.zeros((16, 32, 3), dtype=np.uint8)


def test_missing_model_fails_soft(tmp_path):
    det = OnnxDetector(str(tmp_path / "nope.onnx"), STUB_CLASSES, 64)
    assert det.available is False
    assert det.detect(FRAME) == {}


def test_detect_none_and_empty_frames():
    det = make_detector()
    assert det.detect(None) == {}
    assert det.detect(np.zeros((0, 0, 3), np.uint8)) == {}


def test_letterbox_geometry():
    det = make_detector()
    blob, scale, pad_x, pad_y = det._letterbox(FRAME)
    assert blob.shape == (1, 3, 64, 64)
    assert blob.dtype == np.float32
    assert scale == pytest.approx(2.0)
    assert (pad_x, pad_y) == (0, 16)
    # Padded rows carry the grey fill, content rows the (black) frame.
    assert blob[0, :, 0, :] == pytest.approx(114 / 255.0)
    assert blob[0, :, 20, :] == pytest.approx(0.0)


def test_detect_decodes_filters_and_nms():
    det = make_detector()
    out = det.detect(FRAME)
    assert set(out) == set(STUB_CLASSES)

    # Anchor 1 kept, anchor 2 (1px-shifted duplicate) NMS-suppressed.
    assert len(out["a"]) == 1
    a = out["a"][0]
    assert (a["x"], a["y"]) == (pytest.approx(16.0), pytest.approx(8.0))
    assert (a["width"], a["height"]) == (pytest.approx(10.0), pytest.approx(5.0))
    assert a["confidence"] == pytest.approx(0.90)
    assert a["class"] == "a"

    # Anchor 3: same box as anchor 1 but another class — class-aware NMS
    # must NOT cross-suppress it.
    assert len(out["b"]) == 1
    assert out["b"][0]["x"] == pytest.approx(16.0)
    assert out["b"][0]["confidence"] == pytest.approx(0.85)

    # Anchor 5 maps through the y-pad into the corner.
    assert len(out["c"]) == 1
    c = out["c"][0]
    assert (c["x"], c["y"]) == (pytest.approx(4.0), pytest.approx(2.0))
    assert (c["width"], c["height"]) == (pytest.approx(4.0), pytest.approx(3.0))

    # Anchors 4 (conf .20) and 6 (zeros) fell to the confidence filter:
    # nothing else anywhere.
    assert sum(len(v) for v in out.values()) == 3


def test_conf_threshold_respected():
    det = make_detector(conf_threshold=0.95)
    out = det.detect(FRAME)
    assert set(out) == set(STUB_CLASSES)
    assert all(v == [] for v in out.values())


def test_inference_time_recorded():
    det = make_detector()
    assert det.avg_inference_ms is None
    det.detect(FRAME)
    assert det.avg_inference_ms is not None and det.avg_inference_ms >= 0.0


def test_class_count_mismatch_fails_soft(capsys):
    # Configuring the wrong class list must not raise into the bot loop.
    det = OnnxDetector(STUB_PATH, ["a", "b", "c", "d", "e"], 64)
    assert det.available is True
    assert det.detect(FRAME) == {}
    det.detect(FRAME)  # second failure: logged once only
    assert capsys.readouterr().out.count("detect error") == 1


def test_nms_basic():
    # Two heavy overlaps + one separate box.
    x1 = np.array([0.0, 1.0, 50.0])
    y1 = np.array([0.0, 1.0, 50.0])
    x2 = np.array([10.0, 11.0, 60.0])
    y2 = np.array([10.0, 11.0, 60.0])
    scores = np.array([0.9, 0.8, 0.7])
    assert _nms(x1, y1, x2, y2, scores, 0.45) == [0, 2]


def test_eval_onnx_backend_class_folding(monkeypatch):
    from eval.backends import OnnxBackend, get_backend

    backend = get_backend("onnx")
    assert isinstance(backend, OnnxBackend)

    def fake_dets(_img):
        def d(x, y, w=4.0, h=4.0):
            return {"x": x, "y": y, "width": w, "height": h,
                    "confidence": 0.9, "class": "?"}
        return {
            "oreGold": [d(1.0, 2.0)],
            "oreRuby": [d(5.0, 6.0)],
            "enter": [d(7.0, 8.0)],
            "player": [],
        }

    monkeypatch.setattr(backend._minimap, "detect", fake_dets)
    out = backend.detect_minimap(FRAME)
    assert sorted(out["ore"]) == [(1.0, 2.0), (5.0, 6.0)]
    assert out["portal"] == [(7.0, 8.0)]
    assert out["player"] == []

    monkeypatch.setattr(
        backend._screen, "detect",
        lambda _img: {"enemy": [{"x": 10.0, "y": 10.0, "width": 4.0,
                                 "height": 2.0, "confidence": 0.9,
                                 "class": "enemy"}]})
    boxes = backend.detect_screen(FRAME)
    assert boxes["enemy"] == [(8.0, 9.0, 4.0, 2.0)]

    assert backend.terrain_mask(FRAME) is None
