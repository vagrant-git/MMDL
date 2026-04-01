## summary_mmmodel/pressure_flow_5s/repeat1_fold1 | pressure_flow

- python_env: `dl`
- model: `pressure_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/pressure_flow_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8604, macro-F1=0.8574, precision=0.8874, recall=0.8443
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/pressure_flow_5s/repeat1_fold2 | pressure_flow

- python_env: `dl`
- model: `pressure_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/pressure_flow_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9906, macro-F1=0.9896, precision=0.9869, recall=0.9926
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/pressure_flow_5s/repeat1_fold3 | pressure_flow

- python_env: `dl`
- model: `pressure_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/pressure_flow_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.4565, macro-F1=0.4029, precision=0.3743, recall=0.6207
- result_session: acc=0.6667, macro-F1=0.5556, precision=0.5000, recall=0.6667

## summary_mmmodel | pressure_flow_5s

- model: `Pressure+Flow-only`
- modality: `pressure_flow`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.7692 ± 0.2274, macro-F1=0.7499 ± 0.2513, precision=0.7495 ± 0.2684, recall=0.8192 ± 0.1529
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.1571, macro-F1=0.8519 ± 0.2095, precision=0.8333 ± 0.2357, recall=0.8889 ± 0.1571

## summary_mmmodel/hcaf_normfix_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_normfix_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.7682, macro-F1=0.7490, precision=0.8246, recall=0.7731
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_normfix_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_normfix_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9953, macro-F1=0.9956, precision=0.9948, recall=0.9965
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_normfix_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_normfix_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.7486, macro-F1=0.6587, precision=0.7294, recall=0.7060
- result_session: acc=0.6667, macro-F1=0.6556, precision=0.7222, recall=0.6667

## summary_mmmodel | hcaf_normfix_5s

- model: `HCAF norm fix`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.8374 ± 0.1120, macro-F1=0.8011 ± 0.1424, precision=0.8496 ± 0.1098, recall=0.8252 ± 0.1242
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8333 ± 0.1361, macro-F1=0.8259 ± 0.1406, precision=0.8704 ± 0.1142, recall=0.8333 ± 0.1361

## summary_mmmodel/hcaf_confgate_residual_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.7004, macro-F1=0.6820, precision=0.8375, recall=0.6653
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.7409, macro-F1=0.7363, precision=0.7895, recall=0.7815
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9198, macro-F1=0.9099, precision=0.9349, recall=0.8940
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_confgate_residual_5s

- model: `HCAF confidence gate + expert residual`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.7871 ± 0.0953, macro-F1=0.7760 ± 0.0972, precision=0.8540 ± 0.0605, recall=0.7803 ± 0.0934
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.0786, macro-F1=0.8815 ± 0.0838, precision=0.9259 ± 0.0524, recall=0.8889 ± 0.0786

