#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/_r3dmatch_env.sh"

if [ "$#" -lt 1 ]; then
  echo "Usage: scripts/run_r3dmatch_ingest.sh <ingest-ftps args...>" >&2
  echo "Example: scripts/run_r3dmatch_ingest.sh --action discover --out /path/to/ingest --ftps-reel 007 --ftps-clips 63 --ftps-camera GA" >&2
  exit 64
fi

echo "R3DMatch FTPS ingest launcher"
echo "  repo:        $R3DMATCH_REPO_ROOT"
echo "  interpreter: $R3DMATCH_PYTHON_BIN"
echo "  DYLD source: ${R3DMATCH_DYLD_SOURCE:-unset}"

exec "$R3DMATCH_PYTHON_BIN" -m r3dmatch.cli ingest-ftps "$@"
