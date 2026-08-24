#!/usr/bin/env bash
# One-shot environment restore after `uv sync` prunes pip-installed deps.
# Chains: pybullet patched build -> gym stack -> SurRoL editable.
set -euo pipefail
cd "$(dirname "$0")/../.."
bash backend/tools/install_pybullet.sh
uv pip install "gym==0.25.2" scipy pandas imageio roboticstoolbox-python
uv pip install -e vendor/SurRoL --no-deps
uv run python -c "import gym, surrol, pybullet; print('ENV RESTORED')"
