#!/bin/sh

if [ -n "${R3DMATCH_ENV_READY:-}" ]; then
  return 0 2>/dev/null || exit 0
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${R3DMATCH_PYTHON:-"$REPO_ROOT/.venv/bin/python"}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "R3DMatch launcher error: expected Python interpreter at $PYTHON_BIN" >&2
  echo "Set R3DMATCH_PYTHON or create the project venv before launching." >&2
  return 1 2>/dev/null || exit 1
fi

export R3DMATCH_ENV_READY=1
export R3DMATCH_REPO_ROOT="$REPO_ROOT"
export R3DMATCH_PYTHON_BIN="$PYTHON_BIN"

if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"
else
  export PYTHONPATH="$REPO_ROOT/src"
fi

if [ -z "${DYLD_FALLBACK_LIBRARY_PATH:-}" ] && [ -d /opt/homebrew/lib ]; then
  export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
  export R3DMATCH_DYLD_SOURCE=auto_homebrew
elif [ -n "${DYLD_FALLBACK_LIBRARY_PATH:-}" ]; then
  export R3DMATCH_DYLD_SOURCE=environment
else
  export R3DMATCH_DYLD_SOURCE=unset
fi
