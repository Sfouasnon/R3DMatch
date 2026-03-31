#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/_r3dmatch_env.sh"

HOST=${R3DMATCH_WEB_HOST:-127.0.0.1}
PORT=${1:-${R3DMATCH_WEB_PORT:-5000}}

echo "R3DMatch web launcher"
echo "  repo:        $R3DMATCH_REPO_ROOT"
echo "  interpreter: $R3DMATCH_PYTHON_BIN"
echo "  host:        $HOST"
echo "  port:        $PORT"
echo "  DYLD source: ${R3DMATCH_DYLD_SOURCE:-unset}"

"$R3DMATCH_PYTHON_BIN" -m r3dmatch.cli runtime-health --strict --require-red-backend

echo "Launching web UI on http://$HOST:$PORT"
exec "$R3DMATCH_PYTHON_BIN" -m r3dmatch.web_app --host "$HOST" --port "$PORT"
