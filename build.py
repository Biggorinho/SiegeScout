"""
Run this script to produce dist/SiegeScout.exe.
Usage: python build.py
"""
import subprocess
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent
LOGO_PNG = ROOT / "logo.png"
LOGO_ICO = ROOT / "siegescout.ico"
DIST_DIR = ROOT / "dist"


def make_ico():
    img = Image.open(LOGO_PNG).convert("RGBA").resize((256, 256), Image.LANCZOS)
    img.save(
        LOGO_ICO,
        format="ICO",
        sizes=[(256, 256), (128, 128), (48, 48), (32, 32), (16, 16)],
    )
    print(f"[build] icon written to {LOGO_ICO}")


def build():
    make_ico()
    cmd = [
        sys.executable, "-m", "nuitka",
        "--onefile",
        "--windows-console-mode=disable",
        f"--windows-icon-from-ico={LOGO_ICO}",
        f"--include-data-files={LOGO_PNG}=logo.png",
        f"--output-dir={DIST_DIR}",
        "--output-filename=SiegeScout.exe",
        "--enable-plugin=tk-inter",
        "--assume-yes-for-downloads",
        "--follow-imports",
        str(ROOT / "gui.py"),
    ]
    print(f"[build] running Nuitka — this takes a few minutes on first run")
    subprocess.run(cmd, check=True)
    print(f"[build] done → {DIST_DIR / 'SiegeScout.exe'}")


if __name__ == "__main__":
    build()
