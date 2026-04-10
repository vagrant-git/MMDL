## summary_mmmodel/audio_only_pcen96hp80_5s/repeat1_fold1 | audio_only

- python_env: `dl`
- model: `audio_only`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/audio_only_pcen96hp80_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.5176, macro-F1=0.4980, precision=0.5522, recall=0.5519
- result_session: acc=0.5000, macro-F1=0.4444, precision=0.5000, recall=0.5000

## summary_mmmodel/audio_only_pcen96hp80_5s/repeat1_fold2 | audio_only

- python_env: `dl`
- model: `audio_only`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/audio_only_pcen96hp80_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.4666, macro-F1=0.4534, precision=0.6919, recall=0.5063
- result_session: acc=0.5000, macro-F1=0.4127, precision=0.4667, recall=0.5000

## summary_mmmodel/audio_only_pcen96hp80_5s/repeat1_fold3 | audio_only

- python_env: `dl`
- model: `audio_only`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/audio_only_pcen96hp80_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.7106, macro-F1=0.7245, precision=0.7808, recall=0.7542
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel | audio_only_pcen96hp80_5s

- model: `Audio-only PCEN96 HP80`
- modality: `audio_only`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.5649 ± 0.1051, macro-F1=0.5586 ± 0.1187, precision=0.6749 ± 0.0941, recall=0.6041 ± 0.1078
- best_session_method: `mean_probability_pooling`
- session_mean_std: acc=0.6111 ± 0.1571, macro-F1=0.5598 ± 0.1860, precision=0.6185 ± 0.1917, recall=0.6111 ± 0.1571

## summary_mmmodel/pressure_flow_5s/repeat1_fold1 | pressure_flow

- python_env: `dl`
- model: `pressure_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/pressure_flow_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8751, macro-F1=0.8788, precision=0.9216, recall=0.8730
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/pressure_flow_5s/repeat1_fold2 | pressure_flow

- python_env: `dl`
- model: `pressure_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/pressure_flow_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9777, macro-F1=0.9741, precision=0.9726, recall=0.9758
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/pressure_flow_5s/repeat1_fold3 | pressure_flow

- python_env: `dl`
- model: `pressure_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/pressure_flow_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9620, macro-F1=0.9542, precision=0.9467, recall=0.9653
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | pressure_flow_5s

- model: `Pressure+Flow-only`
- modality: `pressure_flow`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.9383 ± 0.0451, macro-F1=0.9357 ± 0.0410, precision=0.9470 ± 0.0208, recall=0.9380 ± 0.0462
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8131, macro-F1=0.8082, precision=0.8926, recall=0.8098
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9414, macro-F1=0.9241, precision=0.9482, recall=0.9154
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.8954, macro-F1=0.8619, precision=0.8593, recall=0.8760
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s

- model: `HCAF-PCEN-DualXAttn`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.8833 ± 0.0531, macro-F1=0.8647 ± 0.0474, precision=0.9000 ± 0.0367, recall=0.8670 ± 0.0436
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.0786, macro-F1=0.8815 ± 0.0838, precision=0.9259 ± 0.0524, recall=0.8889 ± 0.0786

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold1 | multimodal_minus_audio

- python_env: `dl`
- model: `multimodal_minus_audio`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8155, macro-F1=0.8106, precision=0.8981, recall=0.8133
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold2 | multimodal_minus_audio

- python_env: `dl`
- model: `multimodal_minus_audio`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.8148, macro-F1=0.8078, precision=0.8478, recall=0.7932
- result_session: acc=0.6667, macro-F1=0.6556, precision=0.7222, recall=0.6667

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold3 | multimodal_minus_audio

- python_env: `dl`
- model: `multimodal_minus_audio`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9688, macro-F1=0.9649, precision=0.9582, recall=0.9740
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_audio_5s

- model: `HCAF final compressed without audio`
- modality: `multimodal_minus_audio`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.8663 ± 0.0724, macro-F1=0.8611 ± 0.0734, precision=0.9014 ± 0.0451, recall=0.8601 ± 0.0809
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8333 ± 0.1361, macro-F1=0.8259 ± 0.1406, precision=0.8704 ± 0.1142, recall=0.8333 ± 0.1361

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold1 | multimodal_minus_pressure

- python_env: `dl`
- model: `multimodal_minus_pressure`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8718, macro-F1=0.8755, precision=0.9219, recall=0.8702
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold2 | multimodal_minus_pressure

- python_env: `dl`
- model: `multimodal_minus_pressure`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9097, macro-F1=0.9112, precision=0.9227, recall=0.9115
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold3 | multimodal_minus_pressure

- python_env: `dl`
- model: `multimodal_minus_pressure`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9579, macro-F1=0.9498, precision=0.9438, recall=0.9594
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_pressure_5s

- model: `HCAF final compressed without pressure`
- modality: `multimodal_minus_pressure`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.9131 ± 0.0352, macro-F1=0.9122 ± 0.0304, precision=0.9295 ± 0.0101, recall=0.9137 ± 0.0364
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold1 | multimodal_minus_flow

- python_env: `dl`
- model: `multimodal_minus_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8947, macro-F1=0.8851, precision=0.9030, recall=0.8759
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold2 | multimodal_minus_flow

- python_env: `dl`
- model: `multimodal_minus_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.8757, macro-F1=0.8732, precision=0.9026, recall=0.8700
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold3 | multimodal_minus_flow

- python_env: `dl`
- model: `multimodal_minus_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9674, macro-F1=0.9553, precision=0.9506, recall=0.9608
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_sa0_nosummary_minus_flow_5s

- model: `HCAF final compressed without flow`
- modality: `multimodal_minus_flow`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.9126 ± 0.0395, macro-F1=0.9045 ± 0.0362, precision=0.9187 ± 0.0225, recall=0.9022 ± 0.0415
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.0786, macro-F1=0.8815 ± 0.0838, precision=0.9259 ± 0.0524, recall=0.8889 ± 0.0786

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.6784, macro-F1=0.6467, precision=0.7668, recall=0.6473
- result_session: acc=0.6667, macro-F1=0.5556, precision=0.5000, recall=0.6667

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.6858, macro-F1=0.6870, precision=0.7434, recall=0.7476
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.7758, macro-F1=0.7388, precision=0.7730, recall=0.7172
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_sa0_nosummary_audio_only_5s

- model: `HCAF final compressed audio only`
- modality: `multimodal`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.7133 ± 0.0443, macro-F1=0.6908 ± 0.0377, precision=0.7611 ± 0.0127, recall=0.7040 ± 0.0420
- best_session_method: `majority_voting`
- session_mean_std: acc=0.7778 ± 0.0786, macro-F1=0.7333 ± 0.1257, precision=0.7593 ± 0.1833, recall=0.7778 ± 0.0786

