#!/usr/bin/env bash
# Reproducible SurRoL install for SOMA (macOS, uv-managed py3.11).
# Upstream default branch (SR-VPPV) is a restructured research variant — the
# original package lives on branch `main`. Two patches applied:
#   1. surrol/gym/surrol_env.py: eglRenderer plugin load guarded (macOS has no
#      EGL; DIRECT-mode software rendering works without it).
#   2. gym pinned to 0.25.2 externally (old 4-tuple step API).
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -d vendor/SurRoL ] || git clone --depth 1 https://github.com/med-air/SurRoL.git vendor/SurRoL
git -C vendor/SurRoL fetch --depth 1 origin main
git -C vendor/SurRoL checkout -q -B main FETCH_HEAD
python3 - <<'EOF'
from pathlib import Path
f = Path("vendor/SurRoL/surrol/gym/surrol_env.py")
src = f.read_text()
needle = "egl = pkgutil.get_loader('eglRenderer')\n                plugin = p.loadPlugin(egl.get_filename(), \"_eglRendererPlugin\")"
patch = ("_egl = pkgutil.get_loader('eglRenderer')\n"
         "                if _egl is not None:\n"
         "                    p.loadPlugin(_egl.get_filename(), \"_eglRendererPlugin\")")
if needle in src:
    f.write_text(src.replace(needle, patch))
assert "_egl is not None" in f.read_text(), "surrol egl guard missing"
print("surrol patched")
EOF
uv pip install "gym==0.25.2" scipy pandas imageio roboticstoolbox-python pybullet_data
uv pip install -e vendor/SurRoL --no-deps
uv run python -c "from surrol.tasks.psm_env import PsmEnv; print('SURROL OK')"
