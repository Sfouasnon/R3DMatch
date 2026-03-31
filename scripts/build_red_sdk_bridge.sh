#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
NATIVE_DIR="$ROOT_DIR/src/r3dmatch/native"
BUILD_DIR="$ROOT_DIR/build/red_sdk_bridge"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -z "${RED_SDK_ROOT:-}" ]; then
  echo "ERROR: RED_SDK_ROOT is not set."
  echo "Set RED_SDK_ROOT to your external RED SDK install, for example:"
  echo "  /Users/sfouasnon/Desktop/R3DSplat_Dependecies/RED_SDK/R3DSDKv9_2_0"
  exit 1
fi

if [ ! -d "$RED_SDK_ROOT" ]; then
  echo "ERROR: RED_SDK_ROOT does not exist: $RED_SDK_ROOT"
  exit 1
fi

RED_SDK_INCLUDE_DIR="${RED_SDK_INCLUDE_DIR:-$RED_SDK_ROOT/Include}"
case "$(uname -s)" in
  Darwin)
    RED_SDK_LIBRARY_DIR="${RED_SDK_LIBRARY_DIR:-$RED_SDK_ROOT/Lib/mac64}"
    DERIVED_REDISTRIBUTABLE_DIR="$RED_SDK_ROOT/Redistributable/mac"
    ;;
  Linux)
    RED_SDK_LIBRARY_DIR="${RED_SDK_LIBRARY_DIR:-$RED_SDK_ROOT/Lib/linux64}"
    DERIVED_REDISTRIBUTABLE_DIR="$RED_SDK_ROOT/Redistributable/linux"
    ;;
  *)
    RED_SDK_LIBRARY_DIR="${RED_SDK_LIBRARY_DIR:-$RED_SDK_ROOT/Lib/win64}"
    DERIVED_REDISTRIBUTABLE_DIR="$RED_SDK_ROOT/Redistributable/win"
    ;;
esac
RED_SDK_REDISTRIBUTABLE_DIR="${RED_SDK_REDISTRIBUTABLE_DIR:-$DERIVED_REDISTRIBUTABLE_DIR}"
PYBIND11_DIR="${PYBIND11_DIR:-}"

for required_dir in "$RED_SDK_INCLUDE_DIR" "$RED_SDK_LIBRARY_DIR" "$RED_SDK_REDISTRIBUTABLE_DIR"; do
  if [ ! -d "$required_dir" ]; then
    echo "ERROR: required RED SDK directory is missing: $required_dir"
    exit 1
  fi
done

if [ -z "$PYBIND11_DIR" ]; then
  if PYBIND11_DIR="$("$PYTHON_BIN" -m pybind11 --cmakedir 2>/dev/null)"; then
    :
  else
    echo "ERROR: pybind11 CMake package could not be resolved."
    echo "Set PYBIND11_DIR explicitly or install pybind11 for $PYTHON_BIN."
    exit 1
  fi
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "Configuring RED SDK bridge in $BUILD_DIR"
echo "RED_SDK_ROOT=$RED_SDK_ROOT"
echo "RED_SDK_INCLUDE_DIR=$RED_SDK_INCLUDE_DIR"
echo "RED_SDK_LIBRARY_DIR=$RED_SDK_LIBRARY_DIR"
echo "RED_SDK_REDISTRIBUTABLE_DIR=$RED_SDK_REDISTRIBUTABLE_DIR"
echo "PYBIND11_DIR=$PYBIND11_DIR"

cmake -S "$NATIVE_DIR" -B "$BUILD_DIR" \
  -DRED_SDK_ROOT="$RED_SDK_ROOT" \
  -DRED_SDK_INCLUDE_DIR="$RED_SDK_INCLUDE_DIR" \
  -DRED_SDK_LIBRARY_DIR="$RED_SDK_LIBRARY_DIR" \
  -DRED_SDK_REDISTRIBUTABLE_DIR="$RED_SDK_REDISTRIBUTABLE_DIR" \
  -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build "$BUILD_DIR" --config Release

echo "Build complete. Copy or install the generated _red_sdk_bridge module into src/r3dmatch/ or your Python environment."
