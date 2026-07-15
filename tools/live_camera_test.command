#!/bin/zsh
# Double-click to run the live camera test (Detect-on-Network + RCP2 reset).
# Edit CAMERA_IP / SCAN_CIDR below if your setup changes.
set -e
cd "$(dirname "$0")/.."
echo "▶ R3DMatch live camera test — $(pwd)"

CAMERA_IP="169.254.5.136"
SCAN_CIDR="169.254.5.0/24"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

python3 tools/live_camera_test.py "$CAMERA_IP" "$SCAN_CIDR"

echo ""
echo "▶ Done. Press return to close."
read
