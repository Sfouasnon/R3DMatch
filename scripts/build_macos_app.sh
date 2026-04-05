#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/_r3dmatch_env.sh"

BUILD_ROOT=${R3DMATCH_BUILD_ROOT:-"$R3DMATCH_REPO_ROOT/build/macos_app"}
DIST_ROOT=${R3DMATCH_DIST_ROOT:-"$R3DMATCH_REPO_ROOT/dist"}
APP_NAME=${R3DMATCH_APP_NAME:-R3DMatch}
BUNDLE_ID=${R3DMATCH_BUNDLE_ID:-com.r3dmatch.desktop}
APP_VERSION=${R3DMATCH_APP_VERSION:-0.0.0}
RED_RUNTIME_STAGE_ROOT="$BUILD_ROOT/resources/red_runtime/redistributable"
RED_RUNTIME_SOURCE_DIR=${RED_SDK_REDISTRIBUTABLE_DIR:-}

if [ -z "$RED_RUNTIME_SOURCE_DIR" ] && [ -n "${RED_SDK_ROOT:-}" ]; then
  RED_RUNTIME_SOURCE_DIR="$RED_SDK_ROOT/Redistributable/mac"
fi

if ! "$R3DMATCH_PYTHON_BIN" -m PyInstaller --version >/dev/null 2>&1; then
  echo "PyInstaller is not installed in the active R3DMatch runtime." >&2
  echo "Install it with:" >&2
  echo "  ./.venv/bin/pip install pyinstaller" >&2
  exit 1
fi

mkdir -p "$BUILD_ROOT" "$DIST_ROOT"
rm -rf "$RED_RUNTIME_STAGE_ROOT"
mkdir -p "$RED_RUNTIME_STAGE_ROOT"

if [ -n "$RED_RUNTIME_SOURCE_DIR" ] && [ -d "$RED_RUNTIME_SOURCE_DIR" ]; then
  echo "  bundling RED redistributable runtime from: $RED_RUNTIME_SOURCE_DIR"
  cp "$RED_RUNTIME_SOURCE_DIR"/RED*.dylib "$RED_RUNTIME_STAGE_ROOT"/
else
  echo "  RED redistributable runtime not bundled (set RED_SDK_REDISTRIBUTABLE_DIR or RED_SDK_ROOT to include it)."
fi

echo "Building $APP_NAME.app"
echo "  repo:        $R3DMATCH_REPO_ROOT"
echo "  interpreter: $R3DMATCH_PYTHON_BIN"
echo "  build root:  $BUILD_ROOT"
echo "  dist root:   $DIST_ROOT"

set -- \
  --noconfirm \
  --clean \
  --windowed \
  --name "$APP_NAME" \
  --osx-bundle-identifier "$BUNDLE_ID" \
  --distpath "$DIST_ROOT" \
  --workpath "$BUILD_ROOT/work" \
  --specpath "$BUILD_ROOT/spec" \
  --paths "$R3DMATCH_REPO_ROOT/src" \
  --add-data "$R3DMATCH_REPO_ROOT/src/r3dmatch/static:r3dmatch/static" \
  --collect-submodules weasyprint \
  --hidden-import PySide6 \
  --hidden-import PySide6.QtCore \
  --hidden-import PySide6.QtGui \
  --hidden-import PySide6.QtWidgets \
  --hidden-import r3dmatch.cli \
  --hidden-import r3dmatch.desktop_app \
  --hidden-import r3dmatch.runtime_env \
  --hidden-import r3dmatch.web_app \
  --hidden-import shiboken6

if find "$RED_RUNTIME_STAGE_ROOT" -maxdepth 1 -name 'RED*.dylib' | grep -q .; then
  set -- "$@" --add-data "$RED_RUNTIME_STAGE_ROOT:red_runtime/redistributable"
fi

if [ -f "$R3DMATCH_REPO_ROOT/src/r3dmatch/_red_sdk_bridge.so" ]; then
  set -- "$@" --add-binary "$R3DMATCH_REPO_ROOT/src/r3dmatch/_red_sdk_bridge.so:r3dmatch"
fi

sign_bundle_macos() {
  app_path=$1
  if [ ! -d "$app_path" ]; then
    echo "Expected app bundle at $app_path before signing." >&2
    exit 1
  fi
  if ! command -v /usr/bin/codesign >/dev/null 2>&1; then
    echo "codesign is required to finish the macOS bundle." >&2
    exit 1
  fi

  find "$app_path/Contents" -depth \
    \( -type f \( -name "*.so" -o -name "*.dylib" -o -perm -111 \) -o -type d \( -name "*.framework" -o -name "*.app" \) \) \
    | while IFS= read -r path; do
        /usr/bin/codesign --remove-signature "$path" >/dev/null 2>&1 || true
        /usr/bin/codesign --force --sign - --timestamp=none "$path"
      done

  /usr/bin/codesign --remove-signature "$app_path" >/dev/null 2>&1 || true
  /usr/bin/codesign --force --sign - --timestamp=none "$app_path"
}

"$R3DMATCH_PYTHON_BIN" -m PyInstaller "$@" "$R3DMATCH_REPO_ROOT/src/r3dmatch/web_launcher.py"

APP_PATH="$DIST_ROOT/$APP_NAME.app"
PLIST_PATH="$APP_PATH/Contents/Info.plist"

if [ ! -f "$PLIST_PATH" ]; then
  echo "Expected Info.plist at $PLIST_PATH after build." >&2
  exit 1
fi

PLIST_PATH="$PLIST_PATH" APP_NAME="$APP_NAME" BUNDLE_ID="$BUNDLE_ID" APP_VERSION="$APP_VERSION" "$R3DMATCH_PYTHON_BIN" - <<'PY'
from pathlib import Path
import os
import plistlib

plist_path = Path(os.environ["PLIST_PATH"])
with plist_path.open("rb") as handle:
    payload = plistlib.load(handle)

payload["CFBundleName"] = os.environ["APP_NAME"]
payload["CFBundleDisplayName"] = os.environ["APP_NAME"]
payload["CFBundleExecutable"] = os.environ["APP_NAME"]
payload["CFBundleIdentifier"] = os.environ["BUNDLE_ID"]
payload["CFBundlePackageType"] = "APPL"
payload["CFBundleShortVersionString"] = os.environ["APP_VERSION"]
payload["CFBundleVersion"] = os.environ["APP_VERSION"]
payload["NSPrincipalClass"] = "NSApplication"
payload["LSMinimumSystemVersion"] = payload.get("LSMinimumSystemVersion", "12.0")
payload["NSHighResolutionCapable"] = True
payload["LSApplicationCategoryType"] = "public.app-category.photography"

with plist_path.open("wb") as handle:
    plistlib.dump(payload, handle)
PY

sign_bundle_macos "$APP_PATH"

echo "Built and ad-hoc signed: $APP_PATH"
