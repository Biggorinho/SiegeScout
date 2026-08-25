"""Read the satanic zone name off the HUD by OCR'ing its magenta line.

The HUD prints, top-right:

    Dawn's Chapel          cream   - the zone the player is in
    (skull)Prison Dungeon  MAGENTA - the satanic zone
    Hell (2)               cream   - difficulty
    Safe Zone              green

Only the middle line is magenta, and colour is what separates it from the
zone the player is standing in -- both are zone names, so text alone cannot
tell them apart.

Two things learned from real frames drive this implementation:

* Do NOT binarise the magenta before OCR. Thresholding the glyphs destroys
  the anti-aliasing the recogniser depends on; a strip that reads perfectly
  in colour came back as "11111:1:4 5" once masked. The mask is used only to
  classify which OCR'd line is the magenta one.
* The strip must stop short of the minimap. With the purple-walls game
  setting the minimap is full of magenta pixels and would win the colour
  test outright.

rapidocr-onnxruntime runs on the onnxruntime already shipped for the local
detectors, so this adds no new runtime. Everything is fail-soft: any failure
returns no name and leaves the caller on its existing code path.
"""

import sys

import cv2
import numpy as np

import satanic_name

# HUD strip as fractions of the frame, so it holds at any capture size.
# Right edge stops before the minimap (its left edge sits at ~0.878).
_STRIP_TOP, _STRIP_BOTTOM = 0.0, 0.12
_STRIP_LEFT, _STRIP_RIGHT = 0.55, 0.87

# Magenta in HSV. Tuned on dimmed pause-menu frames, which are the darkest
# the satanic line ever renders.
_MAGENTA_LO = np.array([130, 60, 40])
_MAGENTA_HI = np.array([175, 255, 255])

# Fraction of a text box that must be magenta for it to be the satanic line.
# Glyph strokes cover only a small part of their bounding box, so this is low
# by necessity; the cream and green lines score ~0.
_MIN_MAGENTA = 0.02

_engine = None

# Why the most recent read_satanic_zone call produced no name, or None after a
# successful read. Every failure path used to return an identical
# (None, 0.0, None), so "rapidocr could not load inside the exe" and "the reader
# found nothing" were indistinguishable — which is exactly the gap that left the
# live "HUD satanic line unreadable" on every reset undiagnosed while the same
# frames resolved correctly from source.
last_failure_reason = None


def _fail(reason):
    """Record why no zone could be read, and return the empty result."""
    global last_failure_reason
    last_failure_reason = reason
    return None, 0.0, None


def ocr_available():
    """True when an OCR engine can be constructed."""
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


# rapidocr resolves its three components by BARE module name:
#
#     module_part = importlib.import_module(config['Det']['module_name'])
#     return getattr(module_part, 'TextDetector')
#
# with module_name = "ch_ppocr_v3_det". That works in a normal install only
# because rapid_ocr_api does sys.path.append(root_dir) on its own package
# directory. In a Nuitka onefile build the compiled module lives at
# rapidocr_onnxruntime.ch_ppocr_v3_det, while root_dir points at the extracted
# folder holding the bundled config.yaml and NO code -- so the bare import finds
# an empty namespace package and the getattr raises
#
#     AttributeError: module 'ch_ppocr_v3_det' has no attribute 'TextDetector'
#
# which is exactly what a live exe run reported. Binding the real compiled
# submodules to their bare names first makes import_module return the right
# object. A no-op in a normal install, where the names already resolve.
_RAPIDOCR_COMPONENTS = ("ch_ppocr_v3_det", "ch_ppocr_v3_rec", "ch_ppocr_v2_cls")


def _alias_rapidocr_submodules():
    """Bind rapidocr's component packages to the bare names it imports by."""
    import importlib
    for name in _RAPIDOCR_COMPONENTS:
        if name in sys.modules:
            continue
        try:
            sys.modules[name] = importlib.import_module(
                f"rapidocr_onnxruntime.{name}")
        except Exception:
            # Leave it unbound; read_satanic_zone reports the resulting
            # failure with a reason rather than swallowing it.
            pass


def _default_engine():
    global _engine
    if _engine is None:
        _alias_rapidocr_submodules()
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def hud_strip_bounds(frame):
    """(y0, y1, x0, x1) of the HUD strip within `frame`."""
    h, w = frame.shape[:2]
    return (int(h * _STRIP_TOP), int(h * _STRIP_BOTTOM),
            int(w * _STRIP_LEFT), int(w * _STRIP_RIGHT))


def hud_strip(frame):
    """The top-right HUD band, excluding the minimap."""
    y0, y1, x0, x1 = hud_strip_bounds(frame)
    return frame[y0:y1, x0:x1]


def _magenta_fraction(hsv, box):
    """Share of magenta pixels inside an OCR text box."""
    pts = np.asarray(box, dtype=np.int32).reshape(-1, 2)
    x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())
    sub = hsv[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
    if sub.size == 0:
        return 0.0
    mask = cv2.inRange(sub, _MAGENTA_LO, _MAGENTA_HI)
    return float(np.count_nonzero(mask)) / float(mask.size)


def read_current_zone(frame, candidates, ocr=None):
    """Resolve the zone the player is STANDING IN, as ``(name, ratio, raw_text)``.

    A different question from read_satanic_zone, which hunts the magenta line
    further down the HUD. The current zone is the TOP line of the strip:

        Dawn's Chapel          cream   <- this one
        (skull)Prison Dungeon  MAGENTA <- read_satanic_zone returns this
        Hell (2)               cream
        Safe Zone              green

    When the player is inside the satanic zone the top line reads
    "Satanic <name>" instead, and the current zone is still <name> — so a
    satanic prefix is not a reason to reject the line.

    Exists because a tool that asks to travel to a specific zone cannot trust
    that it arrived: run_vote_reset_sequence falls back to the satanic colour
    scan when a zone click fails, which silently changes the destination.
    Returns ``(None, 0.0, None)`` when it cannot be read; the reason lands in
    last_failure_reason.
    """
    global last_failure_reason
    if frame is None or getattr(frame, "size", 0) == 0:
        return _fail("no_frame")
    try:
        strip = hud_strip(frame)
        if strip.size == 0:
            return _fail("empty_strip")
        engine = ocr if ocr is not None else _default_engine()
        result = engine(strip)
    except Exception as exc:
        return _fail(f"ocr_engine_error: {type(exc).__name__}: {exc}")

    lines = result[0] if isinstance(result, tuple) else result
    if not lines:
        return _fail("ocr_found_no_text")

    # Topmost first: the current zone is the line nearest the top of the strip.
    entries = []
    for entry in lines:
        try:
            box, text = entry[0], entry[1]
        except (TypeError, IndexError):
            continue
        pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        entries.append((float(pts[:, 1].min()), text))
    entries.sort(key=lambda e: e[0])

    for _top, text in entries:
        name, ratio = satanic_name.match_zone_name(text, candidates)
        if name is not None:
            last_failure_reason = None
            return name, ratio, text
    return _fail("no_zone_match")


def read_satanic_zone(frame, candidates, ocr=None, min_magenta=_MIN_MAGENTA):
    """Resolve the satanic zone from a frame as ``(name, ratio, raw_text)``.

    Returns ``(None, 0.0, None)`` whenever the zone cannot be read -- no
    magenta line, OCR unavailable, or the text matching nothing known.
    `ocr` may be any callable taking an image, which keeps this testable
    without the OCR package installed.
    """
    global last_failure_reason
    if frame is None or getattr(frame, "size", 0) == 0:
        return _fail("no_frame")
    try:
        strip = hud_strip(frame)
        if strip.size == 0:
            return _fail("empty_strip")
        engine = ocr if ocr is not None else _default_engine()
        result = engine(strip)
    except Exception as exc:
        # Includes the engine failing to construct — an ImportError or a
        # missing-DLL OSError here is the signature of rapidocr not loading
        # inside the onefile bundle, and it must not read as "found nothing".
        return _fail(f"ocr_engine_error: {type(exc).__name__}: {exc}")

    # rapidocr returns (results, elapsed); a bare list is also accepted.
    lines = result[0] if isinstance(result, tuple) else result
    if not lines:
        return _fail("ocr_found_no_text")

    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    best = (None, 0.0, None)
    saw_satanic_line = False
    for entry in lines:
        try:
            box, text = entry[0], entry[1]
        except (TypeError, IndexError):
            continue
        # Two independent signals, because neither covers every HUD variant:
        # the magenta skull line carries no "Satanic" word, and the cream
        # in-zone line carries no magenta. Requiring both would reject both.
        if not (_magenta_fraction(hsv, box) >= min_magenta
                or satanic_name.has_satanic_label(text)):
            continue          # difficulty/zone-level lines, or map labels
        saw_satanic_line = True
        name, ratio = satanic_name.match_zone_name(text, candidates)
        if name is not None and ratio > best[1]:
            best = (name, ratio, text)

    if best[0] is None:
        # Split these two: "never saw the line" points at the colour/label
        # gates, "saw it but matched nothing" points at OCR text quality.
        return _fail("no_zone_match" if saw_satanic_line else "no_satanic_line")
    last_failure_reason = None
    return best
