#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
NATIVE_DIR="$ROOT_DIR/src/r3dmatch/native"
BUILD_DIR="$ROOT_DIR/build/red_sdk_bridge"

mkdir -p "$BUILD_DIR"

echo "Configuring RED SDK bridge in $BUILD_DIR"
echo "RED_SDK_ROOT=${RED_SDK_ROOT:-<unset>}"

cmake -S "$NATIVE_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --config Release

echo "Build complete. Copy or install the generated _red_sdk_bridge module into src/r3dmatch/ or your Python environment."
