#!/bin/zsh
# Double-click to build dist/R3DMatch.app with py2app.
# Re-run this after any code change to rebundle.
set -e
cd "$(dirname "$0")"
echo "▶ R3DMatch .app build — $(pwd)"

# 1) Venv (reuse if present)
if [[ ! -d .venv ]]; then
  echo "▶ Creating .venv"
  python3 -m venv .venv
fi
source .venv/bin/activate
export PYTHONIOENCODING=utf-8

# 2) Runtime deps + the bundler
echo "▶ Installing runtime deps + py2app"
pip install --upgrade pip wheel >/dev/null
pip install --no-compile -e . >/dev/null
pip install py2app >/dev/null

# 3) Clean previous bundle
rm -rf build dist

# 4) Freeze (alias=off → a real, distributable bundle; log everything)
echo "▶ Running py2app (this takes a few minutes)…"
if python3 setup_app.py py2app 2>&1 | tee build_app.log; then
  if [[ -d dist/R3DMatch.app ]]; then
    SIZE=$(du -sh dist/R3DMatch.app | cut -f1)
    echo "✅ Built dist/R3DMatch.app  (${SIZE})"
    echo "   First launch: right-click → Open (unsigned, Gatekeeper will warn once)."
    open dist
  else
    echo "❌ py2app finished but dist/R3DMatch.app is missing — see build_app.log"
  fi
else
  echo "❌ py2app failed — see build_app.log (send me the last ~40 lines)"
fi
