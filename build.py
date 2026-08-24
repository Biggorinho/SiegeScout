"""
Run this script to produce dist/SiegeScout.exe.
Usage: python build.py
"""
import subprocess
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent
ASSETS_DIR = ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
LOGO_PNG = ASSETS_DIR / "logo.png"
LOGO_ICO = ASSETS_DIR / "siegescout.ico"
DIST_DIR = ROOT / "dist"

# Packages whose imports are LAZY -- done inside functions so a missing
# install degrades gracefully instead of killing startup. --follow-imports
# reasons about module-level imports, so these have to be named explicitly or
# they are silently left out and every ML path in the build turns itself off.
LAZY_PACKAGES = ("onnxruntime", "rapidocr_onnxruntime")

# Training-only packages. They live in the same venv because the trainers run
# from it, but nothing the exe runs imports them -- inference is onnxruntime
# alone. Left to itself Nuitka drags in torch's DLLs and the build balloons by
# ~520MB, which is the difference between an exe you can post and one you
# cannot. Excluding them is safe precisely because the shipped code path never
# touches them.
TRAINING_ONLY = ("torch", "torchvision", "ultralytics")


def make_ico():
    img = Image.open(LOGO_PNG).convert("RGBA").resize((256, 256), Image.LANCZOS)
    img.save(
        LOGO_ICO,
        format="ICO",
        sizes=[(256, 256), (128, 128), (48, 48), (32, 32), (16, 16)],
    )
    print(f"[build] icon written to {LOGO_ICO}")


def check_models():
    """Fail loudly if the shipped weights are missing.

    Every detector is fail-soft: a model it cannot load simply turns itself
    off and the old heuristics run instead. That is right at runtime and
    wrong at build time -- it would produce an exe that looks fine, starts
    fine, and quietly performs like the pre-ML build.
    """
    missing = [n for n in ("minimap.onnx", "screen.onnx")
               if not (MODELS_DIR / n).is_file()]
    if missing:
        raise SystemExit(
            f"[build] ABORT: missing model weights {missing} in {MODELS_DIR}. "
            f"The build would silently ship with ML disabled.")
    total = sum((MODELS_DIR / n).stat().st_size
                for n in ("minimap.onnx", "screen.onnx"))
    print(f"[build] bundling {MODELS_DIR.name}/ ({total / 1e6:.1f} MB of weights)")


def check_dependencies():
    """Fail loudly if the interpreter running the build lacks a runtime dep.

    Nuitka bundles what IT can import, so building from an environment that is
    missing a package produces an exe silently without it. This is easy to hit
    because builds run from .venv while day-to-day work may run from a
    different interpreter -- .venv had neither onnxruntime nor rapidocr for
    months, so every exe built in that time shipped with the ML detectors dead
    and quietly fell back to the old heuristics.
    """
    import importlib
    required = ("onnxruntime", "rapidocr_onnxruntime", "cv2", "numpy",
                "PIL", "mss", "keyboard", "httpx")
    missing = []
    for name in required:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    if missing:
        raise SystemExit(
            f"[build] ABORT: {sys.executable} cannot import {missing}.\n"
            f"[build] Install them into THIS interpreter first:\n"
            f"[build]   {sys.executable} -m pip install -r requirements.txt")
    print(f"[build] dependencies present in {Path(sys.executable).parent.name}")


def build():
    make_ico()
    check_dependencies()
    check_models()
    cmd = [
        sys.executable, "-m", "nuitka",
        "--onefile",
        "--windows-console-mode=disable",
        f"--windows-icon-from-ico={LOGO_ICO}",
        f"--include-data-files={LOGO_PNG}=logo.png",
        # The ONNX weights are loaded from a path relative to config.py, so
        # they must land in the same layout inside the onefile bundle.
        f"--include-data-dir={MODELS_DIR}=assets/models",
        f"--output-dir={DIST_DIR}",
        "--output-filename=SiegeScout.exe",
        "--enable-plugin=tk-inter",
        "--assume-yes-for-downloads",
        "--follow-imports",
        str(ROOT / "gui.py"),
    ]
    # rapidocr carries ~13.7MB of its own .onnx weights plus config.yaml as
    # PACKAGE DATA; without --include-package-data the module is compiled in
    # but has no models to load and OCR fails at runtime.
    for pkg in LAZY_PACKAGES:
        cmd[-1:-1] = [f"--include-package={pkg}",
                      f"--include-package-data={pkg}"]
    for pkg in TRAINING_ONLY:
        cmd[-1:-1] = [f"--nofollow-import-to={pkg}"]
    print("[build] running Nuitka — this takes a few minutes on first run")
    subprocess.run(cmd, check=True)
    print(f"[build] done -> {DIST_DIR / 'SiegeScout.exe'}")


if __name__ == "__main__":
    build()
