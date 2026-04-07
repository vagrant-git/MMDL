#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$SCRIPT_DIR/runtime"
BUNDLED_SENSOR_PARAMS="$SCRIPT_DIR/R_Identification/params.json"
ARGS=("$@")
HAS_SENSOR_PARAMS=0
LIST_AUDIO_ONLY=0

for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --sensor-params)
      HAS_SENSOR_PARAMS=1
      ;;
    --list-audio-devices)
      LIST_AUDIO_ONLY=1
      ;;
  esac
done

if [[ "$HAS_SENSOR_PARAMS" -eq 0 && -n "${SENSOR_PARAMS:-}" ]]; then
  ARGS=(--sensor-params "$SENSOR_PARAMS" "${ARGS[@]}")
  HAS_SENSOR_PARAMS=1
fi

if [[ "$HAS_SENSOR_PARAMS" -eq 0 && -f "$BUNDLED_SENSOR_PARAMS" ]]; then
  ARGS=(--sensor-params "$BUNDLED_SENSOR_PARAMS" "${ARGS[@]}")
  HAS_SENSOR_PARAMS=1
fi

cd "$RUNTIME_DIR"

if [[ "$LIST_AUDIO_ONLY" -eq 1 ]]; then
  python3.9 demo_multimodal.py "${ARGS[@]}"
  exit 0
fi

if [[ "$HAS_SENSOR_PARAMS" -eq 0 ]]; then
  cat <<'EOF'
Usage:
  ./RUN_ON_BOARD.sh --sensor-params /path/to/R_Identification/params.json
  SENSOR_PARAMS=/path/to/R_Identification/params.json ./RUN_ON_BOARD.sh

Helpful checks:
  ./RUN_ON_BOARD.sh --list-audio-devices
  ./RUN_ON_BOARD.sh --sensor-params /path/to/params.json --audio-device-index 1
EOF
  exit 1
fi

python3.9 preflight_check.py "${ARGS[@]}"

python3.9 demo_multimodal.py \
  --onnx-model ../models/hcaf_pcen_dualxattn.onnx \
  --config ../configs/final_model_unified_evidence.yaml \
  "${ARGS[@]}"
