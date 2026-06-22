#!/bin/bash
# R3DMatch v4 — create a dedicated virtual environment and install the app.
# Double-click in Finder, or run:  bash setup_venv.command
set -e
cd "$(dirname "$0")"

echo "R3DMatch v4 — environment setup"
echo "  project: $(pwd)"

PY="${PYTHON:-python3}"
echo "  python : $($PY --version 2>&1)  ($(command -v $PY))"

if [ ! -d ".venv" ]; then
  echo "Creating .venv ..."
  "$PY" -m venv .venv
else
  echo ".venv already exists — reusing it."
fi

# Ensure stdout has an encoding even when run non-interactively (.command), and
# skip byte-compilation: PySide6 ships Jinja template .tmpl.py files that aren't
# valid Python, which crashes pip's compileall step under a piped stdout.
export PYTHONIOENCODING=utf-8

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel >/dev/null
echo "Installing R3DMatch v4 and dependencies (editable) ..."
pip install --no-compile -e .

echo ""
echo "✅ Done. To run R3DMatch v4:"
echo "    source \"$(pwd)/.venv/bin/activate\""
echo "    r3dmatch            # or:  python -m r3dmatch3.app"
echo ""
