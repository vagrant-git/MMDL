# Deployment Technical Overview

## 1. Objective

The purpose of this deployment work was to migrate the current multimodal classification model from the training-oriented experiment repository into a form that can run on an RK3588 edge board. The target system is required to:

- preserve the training-time input conventions for audio, pressure, and flow;
- support export into a deployable inference format;
- support offline verification before real-time integration;
- support real-time acquisition and inference on the edge device.

The deployment process was intentionally separated from the training workflow. Instead of copying the whole training codebase to the board, the deployment path was rebuilt as an independent inference pipeline.

## 2. Target Model and Constraints

The deployed model is the current default multimodal model defined by:

- config: `configs/final_model_unified_evidence.yaml`
- display name: `HCAF-PCEN-DualXAttn`

The deployment implementation must preserve the following key assumptions from training:

- audio sample rate: `16000 Hz`
- sensor sample rate: `100 Hz`
- inference window length: `5 s`
- audio frontend: `PCEN96 + HP80`
- multimodal inputs:
  - audio
  - pressure
  - flow

Because the model is sensitive to preprocessing consistency, the deployment work focused heavily on preserving feature construction and input layout.

## 3. Deployment Strategy

The deployment was completed in three stages.

### 3.1 PC-side offline deployment closure

The first stage was performed entirely on PC:

- load the trained checkpoint;
- reconstruct the multimodal network;
- export the model to ONNX;
- run offline inference with the original PyTorch model;
- run offline inference with ONNX Runtime;
- verify numerical consistency between the two backends.

This stage was necessary to decouple model-export issues from board runtime issues.

### 3.2 RK3588 runtime skeleton

The second stage focused on building an edge runtime skeleton for RK3588:

- real-time USB audio acquisition;
- serial acquisition of pressure/flow from `CH0/CH1`;
- calibration from voltage to physical values;
- ring-buffer management;
- fixed-window multimodal inference.

This part did not reuse the training pipeline directly. Instead, it reused only proven engineering ideas from the old RK3588 project.

### 3.3 Board validation preparation

The third stage prepared the system for on-board validation:

- board bundle generation;
- board dependency list;
- sample test-set selection for parity and runtime checks;
- runtime scripts and documentation for board-side operation.

## 4. Deployment Code Organization

All deployment-related code was consolidated under:

- `deploy/`

The main deployment files are:

- `deploy/edge_deploy_utils.py`
- `deploy/export_onnx.py`
- `deploy/offline_multimodal_infer.py`
- `deploy/rk3588_runtime/demo_multimodal.py`
- `deploy/rk3588_runtime/runtime_infer_onnx.py`
- `deploy/rk3588_runtime/sensor_serial.py`

This separation ensures that deployment tasks can proceed without depending on the full experiment workflow.

## 5. ONNX Export Path

An ONNX export script was implemented in `deploy/export_onnx.py`.

The export workflow performs:

1. load deployment config;
2. rebuild the multimodal architecture;
3. restore checkpoint weights;
4. create dummy multimodal inputs;
5. export the graph to ONNX;
6. save export metadata alongside the model.

The main export artifacts are:

- `deploy/artifacts/hcaf_pcen_dualxattn.onnx`
- `deploy/artifacts/hcaf_pcen_dualxattn.json`

To make ONNX export succeed, one model-side change was required: a data-dependent branch inside `MaskedTokenGate` was rewritten into a pure tensor form so that the exporter could trace the graph without symbolic-shape failure.

## 6. Offline Inference and Consistency Validation

An offline inference script was implemented in `deploy/offline_multimodal_infer.py`.

It supports:

- loading one `MMdata_*` session;
- constructing one multimodal inference window;
- running PyTorch inference;
- running ONNX Runtime inference;
- comparing outputs between the two backends.

This script was used to verify that ONNX deployment preserved the original model behavior.

The observed parity results showed:

- identical predicted class;
- identical top-1 ranking;
- very small logit difference;
- maximum absolute difference on the order of `1e-6`.

These results indicate that the exported ONNX model is a faithful deployment representation of the original PyTorch model.

## 7. Reuse of the Previous RK3588 Project

The previous RK3588 project `breathe_v0.3` was not reused wholesale.

Only two proven engineering ideas were retained:

- USB audio acquisition with `PyAudio`;
- serial reading of pressure/flow using the existing `CH0/CH1` protocol.

The old project’s multimode runtime was analyzed and found to be unsuitable for direct reuse because it did not feed full pressure/flow sequences into the inference model. Instead, it:

- launched `ADC_Realtime.py`;
- estimated `R/C/MP` from incoming data;
- wrote results to CSV;
- polled only the latest derived `R` value in the main process.

That behavior is useful for the old project, but not sufficient for the current multimodal model, which requires the full `pressure` and `flow` time-series as model inputs.

## 8. Real-Time RK3588 Runtime Design

To support the current model, a new runtime structure was implemented.

### 8.1 Audio path

- USB audio is captured using `PyAudio`;
- audio is stored in a ring buffer;
- each inference window reads the latest `5 s` of audio.

### 8.2 Sensor path

- serial messages are parsed from `CH0` and `CH1`;
- calibration coefficients are loaded from `params.json`;
- pressure and flow are converted from voltage to physical values;
- samples are stored in a sensor ring buffer.

### 8.3 Inference trigger

- window size is fixed to `5 s`;
- hop size is configurable;
- each hop extracts:
  - one audio segment,
  - one pressure segment,
  - one flow segment,
- then performs one ONNX Runtime inference call.

## 9. Board-Side Preprocessing Strategy

Board-side preprocessing is a critical part of the deployment because mismatched preprocessing can invalidate inference even when the model itself is correct.

To reduce this risk, a dual-path preprocessing strategy was implemented in `deploy/rk3588_runtime/runtime_infer_onnx.py`.

### 9.1 Preferred path

If the board can install:

- `torch`
- `torchaudio`

then the runtime uses the same preprocessing logic as the training pipeline:

- waveform normalization;
- high-pass filtering;
- Mel spectrogram computation;
- PCEN;
- feature normalization.

This is the recommended path because it is most faithful to training.

### 9.2 Fallback path

If `torch/torchaudio` are not available on board, the runtime falls back to:

- `librosa`
- `scipy`
- `onnxruntime`

This allows quick board-side testing, but it is only an approximation of training-time preprocessing and may lead to drift.

During PC-side validation, this fallback path was observed to produce noticeably different predictions, while the training-consistent preprocessing path restored correct behavior. This confirmed that preprocessing fidelity is a deployment-critical requirement.

## 10. Board Deployment Bundle

A self-contained board deployment package was prepared:

- `rk3588_bundle/`
- `rk3588_bundle.tar.gz`

This package contains:

- ONNX model;
- export metadata;
- config file;
- board runtime scripts;
- dependency list;
- board-side README;
- helper startup script.

Its purpose is to make board-side deployment possible without copying the full experiment repository.

## 11. Test Session Selection for Board Validation

A dedicated board-side test dataset subset was prepared under:

- `board_test_sessions/`

It is split into:

- `smoke/`
  - first-round parity checks;
- `benchmark/`
  - runtime, resource, and stability tests.

The selected sessions were chosen to:

- cover classes `0 / 2 / 4`;
- include both short and long recordings;
- prioritize samples whose offline predictions are stable and high-confidence.

This reduces the chance that board-side debugging is confused by borderline inputs.

## 12. Current Deployment Status

At the end of this deployment round, the following items have already been completed:

- deployment code was reorganized into a dedicated structure;
- ONNX export was completed successfully;
- PyTorch/ONNX parity was verified offline;
- RK3588 real-time acquisition skeleton was implemented;
- board-side ONNX inference runtime was implemented;
- board-side deployment bundle was generated;
- board-side sample sessions were prepared.

Therefore, the current project state is no longer a deployment concept draft. It has reached a stage where:

- deployment is executable;
- inference correctness has been verified offline;
- board-side real-time integration can begin.

## 13. Remaining Work

The next steps are primarily validation tasks rather than code-structure tasks:

1. run board-side parity checks using the prepared `smoke/` sessions;
2. verify that board predictions match PC predictions;
3. measure:
   - preprocessing latency,
   - inference latency,
   - end-to-end latency;
4. monitor:
   - CPU usage,
   - memory usage,
   - runtime stability;
5. if board performance is insufficient, continue toward RKNN/NPU optimization.

## 14. Conclusion

This deployment work established a complete and technically defensible deployment path for the current multimodal model:

- training and deployment were separated;
- ONNX export and offline validation were completed first;
- board runtime acquisition and inference were implemented on top of proven RK3588 engineering patterns;
- a self-contained board bundle was prepared for direct transfer and testing.

The practical conclusion is:

> The current multimodal system has already demonstrated engineering feasibility for RK3588 edge deployment, and the project is now ready to enter board-side parity, performance, and stability validation.
