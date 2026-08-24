#!/usr/bin/env bash
# Reproducible pybullet install for macOS SDK >= 14 (Xcode 15+).
# Upstream sdist bundles ancient zlib whose zutil.h defines
#   #define fdopen(fd, mode) NULL   <- fires when MACOS-guarded branch leaks,
# colliding with SDK _stdio.h fdopen declaration -> "error: expected identifier".
# Patch: add !defined(__APPLE__) guard. Verified working: pybullet 3.2.5, arm64, Xcode 16.
set -euo pipefail
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
uv pip download pybullet==3.2.5 --no-deps -d "$WORK" 2>/dev/null \
  || pip3 download pybullet==3.2.5 --no-deps --no-binary :all: -d "$WORK"
cd "$WORK"
tar xzf pybullet-3.2.5.tar.gz
cd pybullet-3.2.5
sed -i '' 's/#ifndef fdopen$/#if !defined(fdopen) \&\& !defined(__APPLE__)/' \
  examples/ThirdPartyLibs/zlib/zutil.h
grep -q "!defined(__APPLE__)" examples/ThirdPartyLibs/zlib/zutil.h
cd "$WORK"
uv pip install "setuptools<66" wheel
uv pip install --no-build-isolation ./pybullet-3.2.5
uv run python -c "import pybullet; print('PYBULLET OK api', pybullet.getAPIVersion())"
