#!/usr/bin/env bash
# Dataset and SFT toolkit inherited from Side-Step.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python jk_step.py toolkit "$@"
