#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/_r3dmatch_env.sh"

APP_NAME=${R3DMATCH_APP_NAME:-R3DMatch}
DIST_ROOT=${R3DMATCH_DIST_ROOT:-"$R3DMATCH_REPO_ROOT/dist"}
APP_PATH="$DIST_ROOT/$APP_NAME.app"
ZIP_PATH="$DIST_ROOT/${APP_NAME}-macos-arm64.zip"

if [ ! -d "$APP_PATH" ]; then
  echo "Missing app bundle at $APP_PATH" >&2
  echo "Build it first with scripts/build_macos_app.sh" >&2
  exit 1
fi

rm -f "$ZIP_PATH"
echo "Packaging $APP_PATH"
echo "  zip: $ZIP_PATH"
if /usr/bin/ditto -c -k --sequesterRsrc --keepParent "$APP_PATH" "$ZIP_PATH"; then
  exit 0
fi

echo "ditto packaging failed; falling back to zip -qry for transport packaging." >&2
rm -f "$ZIP_PATH"
(
  cd "$DIST_ROOT"
  exec /usr/bin/zip -qry "$ZIP_PATH" "$APP_NAME.app"
)
