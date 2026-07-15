#!/bin/zsh
# Double-click to force macOS to re-read R3DMatch.app's icon (clears the stale
# LaunchServices/Finder icon cache that shows a generic icon after a rebuild).
set -e
cd "$(dirname "$0")/.."
APP="$(pwd)/dist/R3DMatch.app"
echo "▶ Refreshing icon for: $APP"

if [[ ! -d "$APP" ]]; then
  echo "❌ $APP not found — build it first with build_app.command"
  read; exit 1
fi

# 1) Touch the bundle so its mod-time changes
touch "$APP"

# 2) Re-register with LaunchServices (no sudo needed)
LSREG="/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister"
if [[ -x "$LSREG" ]]; then
  "$LSREG" -f "$APP"
  echo "✓ Re-registered with LaunchServices"
fi

# 3) Restart Finder + Dock to drop their in-memory icon cache
killall Finder 2>/dev/null || true
killall Dock   2>/dev/null || true
echo "✓ Finder + Dock restarted"

echo ""
echo "If the small icon is STILL generic, the system icon store is stale. Run this"
echo "once in Terminal (asks for your password), then log out/in:"
echo "   sudo rm -rf /Library/Caches/com.apple.iconservices.store"
echo "   killall Finder Dock"
echo ""
echo "▶ Done. Press return to close."
read
