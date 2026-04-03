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
- result_window: acc=0.8539, macro-F1=0.8562, precision=0.9138, recall=0.8513
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9472, macro-F1=0.9347, precision=0.9491, recall=0.9286
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9660, macro-F1=0.9680, precision=0.9668, recall=0.9713
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s

- model: `HCAF final compressed multimodal`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.9224 ± 0.0490, macro-F1=0.9196 ± 0.0469, precision=0.9432 ± 0.0220, recall=0.9170 ± 0.0497
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.0786, macro-F1=0.8815 ± 0.0838, precision=0.9259 ± 0.0524, recall=0.8889 ± 0.0786

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold1 | multimodal_minus_audio

- python_env: `dl`
- model: `multimodal_minus_audio`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8653, macro-F1=0.8683, precision=0.9169, recall=0.8631
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold2 | multimodal_minus_audio

- python_env: `dl`
- model: `multimodal_minus_audio`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9097, macro-F1=0.8897, precision=0.9232, recall=0.8816
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold3 | multimodal_minus_audio

- python_env: `dl`
- model: `multimodal_minus_audio`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9389, macro-F1=0.9037, precision=0.8937, recall=0.9215
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s

- model: `HCAF final compressed without audio`
- modality: `multimodal_minus_audio`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.9046 ± 0.0302, macro-F1=0.8872 ± 0.0145, precision=0.9113 ± 0.0127, recall=0.8887 ± 0.0244
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.0786, macro-F1=0.8815 ± 0.0838, precision=0.9259 ± 0.0524, recall=0.8889 ± 0.0786

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold1 | multimodal_minus_pressure

- python_env: `dl`
- model: `multimodal_minus_pressure`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.9388, macro-F1=0.9428, precision=0.9559, recall=0.9373
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold2 | multimodal_minus_pressure

- python_env: `dl`
- model: `multimodal_minus_pressure`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9285, macro-F1=0.9086, precision=0.9345, recall=0.9007
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold3 | multimodal_minus_pressure

- python_env: `dl`
- model: `multimodal_minus_pressure`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9620, macro-F1=0.9621, precision=0.9590, recall=0.9680
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s

- model: `HCAF final compressed without pressure`
- modality: `multimodal_minus_pressure`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.9431 ± 0.0140, macro-F1=0.9379 ± 0.0221, precision=0.9498 ± 0.0109, recall=0.9353 ± 0.0275
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold1 | multimodal_minus_flow

- python_env: `dl`
- model: `multimodal_minus_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.6792, macro-F1=0.6540, precision=0.7552, recall=0.6515
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold2 | multimodal_minus_flow

- python_env: `dl`
- model: `multimodal_minus_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9285, macro-F1=0.9224, precision=0.9251, recall=0.9236
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold3 | multimodal_minus_flow

- python_env: `dl`
- model: `multimodal_minus_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9755, macro-F1=0.9740, precision=0.9741, recall=0.9743
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s

- model: `HCAF final compressed without flow`
- modality: `multimodal_minus_flow`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.8611 ± 0.1300, macro-F1=0.8501 ± 0.1403, precision=0.8848 ± 0.0938, recall=0.8498 ± 0.1418
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.7992, macro-F1=0.7887, precision=0.8267, recall=0.7800
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.5404, macro-F1=0.5185, precision=0.7234, recall=0.6086
- result_session: acc=0.5000, macro-F1=0.4127, precision=0.4667, recall=0.5000

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.8533, macro-F1=0.8284, precision=0.8456, recall=0.8148
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s

- model: `HCAF final compressed audio only`
- modality: `multimodal`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.7310 ± 0.1365, macro-F1=0.7119 ± 0.1377, precision=0.7986 ± 0.0537, recall=0.7345 ± 0.0901
- best_session_method: `majority_voting`
- session_mean_std: acc=0.7778 ± 0.2079, macro-F1=0.7450 ± 0.2459, precision=0.7852 ± 0.2297, recall=0.7778 ± 0.2079

