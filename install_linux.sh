#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
backend="${JK_BACKEND:-cu128}"
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/ and rerun."
  exit 1
fi
if [ ! -x .venv/bin/python ]; then uv python install 3.12; uv venv --python 3.12 .venv; fi
uv pip install --python .venv/bin/python torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0 --index-url "https://download.pytorch.org/whl/$backend"
uv pip install --python .venv/bin/python -r requirements.txt -c constraints.txt
uv pip install --python .venv/bin/python --no-deps -e .
if [ "${JK_SKIP_MODELS:-0}" != 1 ]; then .venv/bin/python jk_step.py models setup; fi
.venv/bin/python jk_step.py doctor
echo "JK-Step ready: ./jk-step.sh gui"
