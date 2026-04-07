This bundle collects the files needed to trace and reproduce the PC-side probability export path.

Requested contents:
1. A reproducible script that generates PC_probabilities_by_window.csv
2. The preprocessing code imported by that script
3. The model definition, config, and checkpoint
4. The ONNX export script
5. Environment version information

Important note:
- The original historical script that first generated
  board_test_sessions/smoke_diff_artifacts/PC_probabilities_by_window.csv
  was not found in the current workspace.
- This bundle therefore includes a current reproducible generator:
  generate_pc_probabilities_by_window.py
  It uses the same current repo inference chain:
  deploy.edge_deploy_utils.build_window_batch + deploy/artifacts/hcaf_pcen_dualxattn.onnx

Included files are listed in MANIFEST.txt.
