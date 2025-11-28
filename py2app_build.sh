#!/usr/bin/env bash
set -euo pipefail

# Build script for macOS app using py2app.
# Place your main.py next to this script (or edit setup.py to point to it).
# Usage:
#   ./py2app_build.sh        # build only
#   ./py2app_build.sh test   # build then run a smoke test

WD="$(cd "$(dirname "$0")" && pwd)"
cd "$WD"

# Expected bundle/executable name (keep in sync with setup.py APP_NAME)
APP_BUNDLE_NAME="MyApp"

if [ ! -f "main.py" ]; then
  echo "main.py not found in $WD. Copy your main.py here or edit setup.py to point to its location."
  exit 1
fi

# Create/activate venv
python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate

pip install --upgrade pip setuptools wheel py2app

# Build a standalone app into dist/
python setup.py py2app

echo "Build complete. Find ${APP_BUNDLE_NAME}.app in dist/ (or adjust APP name in setup.py)."

# If invoked with "test", attempt to run the built app
if [ "${1:-}" = "test" ]; then
  BUNDLE_PATH="dist/${APP_BUNDLE_NAME}.app"
  EXEC_PATH="${BUNDLE_PATH}/Contents/MacOS/${APP_BUNDLE_NAME}"

  if [ -x "$EXEC_PATH" ]; then
    echo "Running app executable: $EXEC_PATH"
    "$EXEC_PATH"
    RC=$?
    echo "App exited with code $RC"
    exit $RC
  elif [ -d "$BUNDLE_PATH" ]; then
    echo "Executable not found; opening bundle with 'open -W' to wait until it exits"
    # open -W waits until the app exits. This may not reflect the actual process exit code.
    open -W "$BUNDLE_PATH"
    RC=$?
    echo "'open' returned exit code $RC (may be 0 even if app had issues)"
    exit $RC
  else
    echo "Built app not found at $BUNDLE_PATH"
    exit 2
  fi
fi
