#!/usr/bin/env bash
set -euo pipefail

# Uses YAML config by default; override with:
# SWEEP_CONFIG=configs/my_sweep.yaml bash scripts/run_baseline_sweep.sh
SWEEP_CONFIG_PATH="${SWEEP_CONFIG:-configs/sweep_baseline.yaml}"

uv run python -m src.sweep_baseline \
  --config "${SWEEP_CONFIG_PATH}" \
  "$@"
