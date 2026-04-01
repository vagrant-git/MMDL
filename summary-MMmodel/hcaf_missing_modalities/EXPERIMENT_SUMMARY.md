## summary_mmmodel | hcaf_confgate_residual_full_5s

- model: `HCAF full multimodal`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.8337 ± 0.1205, macro-F1=0.8065 ± 0.1489, precision=0.8699 ± 0.1110, recall=0.8066 ± 0.1522
- best_session_method: `mean_probability_pooling`
- session_mean_std: acc=0.8333 ± 0.1361, macro-F1=0.8296 ± 0.1362, precision=0.9074 ± 0.0693, recall=0.8333 ± 0.1361

## summary_mmmodel | hcaf_confgate_residual_minus_audio_5s

- model: `HCAF without audio`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.7260 ± 0.1607, macro-F1=0.7028 ± 0.2052, precision=0.7257 ± 0.2060, recall=0.7481 ± 0.1483
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8333 ± 0.2357, macro-F1=0.8042 ± 0.2769, precision=0.8222 ± 0.2514, recall=0.8333 ± 0.2357

## summary_mmmodel | hcaf_confgate_residual_minus_pressure_5s

- model: `HCAF without pressure`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.8958 ± 0.0780, macro-F1=0.8662 ± 0.0910, precision=0.9122 ± 0.0591, recall=0.8600 ± 0.0968
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.0786, macro-F1=0.8815 ± 0.0838, precision=0.9259 ± 0.0524, recall=0.8889 ± 0.0786

## summary_mmmodel/hcaf_confgate_residual_minus_flow_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_minus_flow_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.4824, macro-F1=0.4669, precision=0.6472, recall=0.5504
- result_session: acc=0.6667, macro-F1=0.6667, precision=0.8333, recall=0.6667

## summary_mmmodel/hcaf_confgate_residual_minus_flow_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_minus_flow_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8628, macro-F1=0.8547, precision=0.8496, recall=0.8674
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8333, recall=0.8889

## summary_mmmodel/hcaf_confgate_residual_minus_flow_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_minus_flow_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.7284, macro-F1=0.5798, precision=0.6667, recall=0.5288
- result_session: acc=0.8333, macro-F1=0.6190, precision=0.6667, recall=0.5833

## summary_mmmodel/hcaf_confgate_residual_minus_flow_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_minus_flow_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.7272, macro-F1=0.7454, precision=0.7914, recall=0.8148
- result_session: acc=0.6667, macro-F1=0.6667, precision=0.7778, recall=0.7778

## summary_mmmodel | hcaf_confgate_residual_minus_flow_5s

- model: `HCAF without flow`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.7728 ± 0.0636, macro-F1=0.7266 ± 0.1130, precision=0.7692 ± 0.0763, recall=0.7370 ± 0.1488
- best_session_method: `majority_voting`
- session_mean_std: acc=0.7778 ± 0.0786, macro-F1=0.7026 ± 0.0868, precision=0.7593 ± 0.0693, recall=0.7500 ± 0.1263

## summary_mmmodel/hcaf_confgate_residual_audio_only_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_audio_only_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.7628, macro-F1=0.7731, precision=0.8240, recall=0.8255
- result_session: acc=0.8333, macro-F1=0.8413, precision=0.9167, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_audio_only_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_audio_only_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.7261, macro-F1=0.5001, precision=0.5392, recall=0.5277
- result_session: acc=0.8333, macro-F1=0.5524, precision=0.5556, recall=0.5833

## summary_mmmodel/hcaf_confgate_residual_audio_only_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_audio_only_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.5717, macro-F1=0.6035, precision=0.7755, recall=0.7271
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8333, recall=0.8889

## summary_mmmodel | hcaf_confgate_residual_audio_only_5s

- model: `HCAF audio only`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.6869 ± 0.0828, macro-F1=0.6255 ± 0.1125, precision=0.7129 ± 0.1244, recall=0.6934 ± 0.1239
- best_session_method: `mean_probability_pooling`
- session_mean_std: acc=0.8333 ± 0.0000, macro-F1=0.7386 ± 0.1319, precision=0.7685 ± 0.1544, recall=0.7685 ± 0.1329

