This bundle contains the non-dataset materials needed to analyze the current deployment mismatch on a PC.

Included:
- `rk3588_bundle/`: deployed ONNX bundle, configs, runtime scripts, and model files
- `docs/`: current validation reports and notes
- `artifacts/`: PC/board probability CSVs and alignment sweep summaries
- `scripts/eval_prob_alignment.py`: the offline sweep script used to compare board exports against `PC_probabilities_by_window.csv`
- `scripts/mmdl_baseline/preprocessing/signals.py`: temporary torch-side preprocessing shim used during testing

Excluded on purpose:
- raw dataset contents under `board_test_sessions/smoke/`
- raw dataset contents under `board_test_sessions/benchmark/`
- any generated runtime snapshots from live capture

Current headline finding:
- normalization/backend sweeps only changed the mean absolute probability difference slightly
- the strongest evidence still points to a model/export provenance mismatch between the provided PC CSV and the deployed ONNX bundle
