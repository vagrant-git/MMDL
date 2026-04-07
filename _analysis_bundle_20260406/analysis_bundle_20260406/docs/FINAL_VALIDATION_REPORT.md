# Board Validation Report

## 1. Inference Consistency Validation

Validation data: `smoke/`

Method:
- Reference path: `audio.wav` resampled from `22050 Hz -> 16000 Hz`, DAQ kept at `100 Hz`
- Board-simulated path: audio resampled `22050 Hz -> 44100 Hz -> 16000 Hz`, sensor resampled `100 Hz -> 10 Hz -> 100 Hz`
- Model: `rk3588_bundle/models/hcaf_pcen_dualxattn.onnx`
- Window: `5 s`, hop: `5 s`

Result:
- Total windows: `362`
- Window-level prediction agreement between reference and board-simulated path: `362 / 362 = 100%`
- Max absolute probability difference: `0.02897`
- Max absolute logit difference: `2.11652`

Per-session majority result:
- `MMdata_235.00s_0320_224031_no_secretion`: expected `0`, reference `4`, board `4`
- `MMdata_1136.50s_0327_183428_2ml`: expected `2`, reference `4`, board `4`
- `MMdata_442.75s_0327_203239_4ml`: expected `4`, reference `4`, board `4`

Conclusion:
- Deployment path is consistent with the reference path.
- Current model output on this test set is dominated by class `4`; consistency passed, task accuracy on `0/2` samples did not.

## 2. Real-Time / Resource Validation

Validation data: `benchmark/`

Method:
- Board-simulated path only
- Window: `5 s`, hop: `1 s`
- Total evaluated windows: `1929`

Overall result:
- Mean session latency: `117.08 ms`
- Max session `p95` latency: `177.65 ms`
- Worst single-window latency: `218.97 ms`
- Throughput: `8.42 windows/s`
- Real-time factor vs `1 s` hop: `0.178`
- Max RSS: `284.71 MiB`

Per-session majority result:
- `MMdata_1071.50s_0327_201445_4ml`: expected `4`, predicted `4`
- `MMdata_272.75s_0327_174501_2ml`: expected `2`, predicted `4`
- `MMdata_598.25s_0322_224923_no_secretion`: expected `0`, predicted `4`

Conclusion:
- Real-time capability passed with large margin.
- On current board path, inference latency is well below the `1 s` hop budget.

## 3. Live Capture Display

For screenshot-friendly live monitoring:

```bash
python3.9 ~/rk3588_bundle/runtime/live_capture_monitor.py
```

If you want to bind to the USB audio device explicitly:

```bash
python3.9 ~/rk3588_bundle/runtime/live_capture_monitor.py \
  --audio-device-index 3 \
  --audio-input-rate 44100
```

This monitor shows:
- latest `pressure`
- latest `flow`
- `audio_rms`
- `audio_peak`
- rolling ASCII trends for `P / F / A / K`

Detailed machine-readable output:
- `board_test_sessions/validation_summary.json`
