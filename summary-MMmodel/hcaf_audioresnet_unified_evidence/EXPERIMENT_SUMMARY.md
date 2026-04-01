## summary_mmmodel/hcaf_audio_r18img_audio_only_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_audio_only_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8024, macro-F1=0.7916, precision=0.8871, recall=0.8001
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_audio_r18img_audio_only_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_audio_only_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9719, macro-F1=0.9663, precision=0.9691, recall=0.9648
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_audio_r18img_audio_only_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_audio_only_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.8927, macro-F1=0.8549, precision=0.8402, recall=0.9264
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_audio_r18img_audio_only_5s

- model: `HCAF Audio R18 ImageNet audio only`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.8890 ± 0.0692, macro-F1=0.8709 ± 0.0722, precision=0.8988 ± 0.0533, recall=0.8971 ± 0.0704
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

## summary_mmmodel | pressure_flow_5s

- model: `Pressure+Flow-only`
- modality: `pressure_flow`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.7692 ± 0.2274, macro-F1=0.7499 ± 0.2513, precision=0.7495 ± 0.2684, recall=0.8192 ± 0.1529
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.1571, macro-F1=0.8519 ± 0.2095, precision=0.8333 ± 0.2357, recall=0.8889 ± 0.1571

## summary_mmmodel | hcaf_audio_r18img_pq_tcn_5s

- model: `Audio R18 ImageNet + PQ TCN`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.9176 ± 0.0722, macro-F1=0.9145 ± 0.0745, precision=0.9448 ± 0.0350, recall=0.9154 ± 0.0723
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

## summary_mmmodel/hcaf_audio_r18img_minus_audio_5s/repeat1_fold1 | multimodal_minus_audio

- python_env: `dl`
- model: `multimodal_minus_audio`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_minus_audio_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8204, macro-F1=0.8150, precision=0.8955, recall=0.8180
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_audio_r18img_minus_audio_5s/repeat1_fold2 | multimodal_minus_audio

- python_env: `dl`
- model: `multimodal_minus_audio`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_minus_audio_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9343, macro-F1=0.9158, precision=0.9376, recall=0.9084
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

