#!/usr/bin/env bash
set -euo pipefail

# Minimal entrypoint for one full default experiment run.
# Prerequisites:
# 1. Conda is installed at /home/oi/miniforge3
# 2. The `dl` environment already exists
# 3. Run this script from the repo root

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl

python summary_mmmodel_experiments.py --config configs/final_model_unified_evidence.yaml
