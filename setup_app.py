"""py2app build config — produces dist/R3DMatch.app.

Run on macOS (not in the sandbox):
    python3 setup_app.py py2app

Or just double-click build_app.command, which sets up the venv and runs this.

Notes
-----
* QtWebEngine is excluded on purpose — it's the heaviest, most fragile thing to
  freeze, and the Results page already falls back to "Open in Browser" when it's
  absent. Re-add it later by removing the QtWebEngine entries from `excludes`.
* The scientific stack (numpy/scipy/skimage/cv2/PIL/tifffile) is listed in
  `packages` so py2app copies each package whole — the most reliable option for
  libraries that carry compiled extensions and data files.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from setuptools import setup

APP = ["R3DMatch_launch.py"]

OPTIONS = {
    "argv_emulation": False,
    "packages": [
        "r3dmatch3",
        "numpy",
        "scipy",
        "skimage",
        "cv2",
        "PIL",
        "tifffile",
        "websockets",
    ],
    "includes": [
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
    ],
    "qt_plugins": ["platforms", "styles", "imageformats"],
    "excludes": [
        "PyQt5", "PyQt6", "tkinter", "matplotlib", "pytest", "IPython",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtQuick", "PySide6.QtQml", "PySide6.Qt3DCore",
        "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    ],
    "plist": {
        "CFBundleName": "R3DMatch",
        "CFBundleDisplayName": "R3DMatch",
        "CFBundleIdentifier": "com.ilm.r3dmatch",
        "CFBundleVersion": "4.0.0",
        "CFBundleShortVersionString": "4.0.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
        "NSHumanReadableCopyright": "ILM — R3DMatch v4",
    },
}

setup(
    name="R3DMatch",
    app=APP,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
