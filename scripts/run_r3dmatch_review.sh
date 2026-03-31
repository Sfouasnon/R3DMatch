#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/_r3dmatch_env.sh"

if [ "$#" -lt 1 ]; then
  echo "Usage: scripts/run_r3dmatch_review.sh <review-calibration args...>" >&2
  echo "Example: scripts/run_r3dmatch_review.sh /path/to/calibration --out /path/to/run --target-type gray_sphere --processing-mode both --backend red --review-mode full_contact_sheet --preview-mode monitoring --target-strategy median" >&2
  exit 64
fi

echo "R3DMatch review launcher"
echo "  repo:        $R3DMATCH_REPO_ROOT"
echo "  interpreter: $R3DMATCH_PYTHON_BIN"
echo "  DYLD source: ${R3DMATCH_DYLD_SOURCE:-unset}"

"$R3DMATCH_PYTHON_BIN" -m r3dmatch.cli runtime-health --strict --require-red-backend

exec "$R3DMATCH_PYTHON_BIN" -m r3dmatch.cli review-calibration "$@"
