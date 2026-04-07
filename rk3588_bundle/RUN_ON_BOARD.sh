#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/runtime"

python3.9 demo_multimodal.py \
  --onnx-model ../models/hcaf_pcen_dualxattn.onnx \
  --config ../configs/final_model_unified_evidence.yaml \
  "$@"
