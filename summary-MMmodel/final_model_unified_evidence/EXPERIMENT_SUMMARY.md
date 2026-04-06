## summary_mmmodel | audio_only_pcen96hp80_5s

- model: `Audio-only PCEN96 HP80`
- modality: `audio_only`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.7170 ± 0.0578, macro-F1=0.7052 ± 0.0667, precision=0.7535 ± 0.0784, recall=0.7171 ± 0.0525
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8333 ± 0.1361, macro-F1=0.8296 ± 0.1362, precision=0.9074 ± 0.0693, recall=0.8333 ± 0.1361

## summary_mmmodel | pressure_flow_5s

- model: `Pressure+Flow-only`
- modality: `pressure_flow`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.7692 ± 0.2274, macro-F1=0.7499 ± 0.2513, precision=0.7495 ± 0.2684, recall=0.8192 ± 0.1529
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.1571, macro-F1=0.8519 ± 0.2095, precision=0.8333 ± 0.2357, recall=0.8889 ± 0.1571

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.9355, macro-F1=0.9379, precision=0.9501, recall=0.9315
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.6401, macro-F1=0.5847, precision=0.6508, recall=0.6088
- result_session: acc=0.5000, macro-F1=0.4444, precision=0.5000, recall=0.5000

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9497, macro-F1=0.9447, precision=0.9476, recall=0.9445
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s

- model: `HCAF-PCEN-DualXAttn`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.8418 ± 0.1427, macro-F1=0.8225 ± 0.1681, precision=0.8495 ± 0.1405, recall=0.8283 ± 0.1553
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8333 ± 0.2357, macro-F1=0.8148 ± 0.2619, precision=0.8333 ± 0.2357, recall=0.8333 ± 0.2357

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold1 | multimodal_minus_audio

- python_env: `dl`
- model: `multimodal_minus_audio`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8604, macro-F1=0.8632, precision=0.9167, recall=0.8581
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold2 | multimodal_minus_audio

- python_env: `dl`
- model: `multimodal_minus_audio`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.8839, macro-F1=0.8743, precision=0.8977, recall=0.8670
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold3 | multimodal_minus_audio

- python_env: `dl`
- model: `multimodal_minus_audio`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.5516, macro-F1=0.5360, precision=0.7128, recall=0.6837
- result_session: acc=0.6667, macro-F1=0.5556, precision=0.5000, recall=0.6667

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s

- model: `HCAF final variant without audio`
- modality: `multimodal_minus_audio`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.7653 ± 0.1514, macro-F1=0.7578 ± 0.1570, precision=0.8424 ± 0.0920, recall=0.8029 ± 0.0844
- best_session_method: `majority_voting`
- session_mean_std: acc=0.7778 ± 0.0786, macro-F1=0.7333 ± 0.1257, precision=0.7593 ± 0.1833, recall=0.7778 ± 0.0786

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold1 | multimodal_minus_pressure

- python_env: `dl`
- model: `multimodal_minus_pressure`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.7739, macro-F1=0.7566, precision=0.8474, recall=0.7678
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold2 | multimodal_minus_pressure

- python_env: `dl`
- model: `multimodal_minus_pressure`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9613, macro-F1=0.9540, precision=0.9640, recall=0.9479
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold3 | multimodal_minus_pressure

- python_env: `dl`
- model: `multimodal_minus_pressure`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9674, macro-F1=0.9691, precision=0.9703, recall=0.9697
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s

- model: `HCAF final variant without pressure`
- modality: `multimodal_minus_pressure`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.9009 ± 0.0898, macro-F1=0.8932 ± 0.0968, precision=0.9272 ± 0.0565, recall=0.8951 ± 0.0905
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold1 | multimodal_minus_flow

- python_env: `dl`
- model: `multimodal_minus_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.9355, macro-F1=0.9339, precision=0.9390, recall=0.9354
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold2 | multimodal_minus_flow

- python_env: `dl`
- model: `multimodal_minus_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.8664, macro-F1=0.8610, precision=0.8885, recall=0.8593
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold3 | multimodal_minus_flow

- python_env: `dl`
- model: `multimodal_minus_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9484, macro-F1=0.9399, precision=0.9560, recall=0.9270
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s

- model: `HCAF final variant without flow`
- modality: `multimodal_minus_flow`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.9167 ± 0.0360, macro-F1=0.9116 ± 0.0359, precision=0.9279 ± 0.0287, recall=0.9072 ± 0.0341
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.6629, macro-F1=0.6289, precision=0.7536, recall=0.6217
- result_session: acc=0.6667, macro-F1=0.5556, precision=0.5000, recall=0.6667

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.6893, macro-F1=0.6923, precision=0.7078, recall=0.6926
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.8451, macro-F1=0.8045, precision=0.7885, recall=0.8351
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s

- model: `HCAF final variant audio only`
- modality: `multimodal`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.7324 ± 0.0804, macro-F1=0.7086 ± 0.0726, precision=0.7500 ± 0.0330, recall=0.7165 ± 0.0888
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8333 ± 0.1361, macro-F1=0.7926 ± 0.1826, precision=0.7963 ± 0.2144, recall=0.8333 ± 0.1361
