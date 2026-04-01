## summary_mmmodel/hcaf_audio_r18img_pq_tcn_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_tcn_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8155, macro-F1=0.8094, precision=0.8958, recall=0.8135
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_audio_r18img_pq_tcn_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_tcn_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9672, macro-F1=0.9599, precision=0.9639, recall=0.9576
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_audio_r18img_pq_tcn_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_tcn_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9701, macro-F1=0.9742, precision=0.9748, recall=0.9749
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_audio_r18img_pq_tcn_5s

- model: `Audio R18 ImageNet + PQ TCN (baseline)`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.9176 ± 0.0722, macro-F1=0.9145 ± 0.0745, precision=0.9448 ± 0.0350, recall=0.9154 ± 0.0723
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

## summary_mmmodel/hcaf_audio_r18img_pq_tcn_focal_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_tcn_focal_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8041, macro-F1=0.7960, precision=0.8938, recall=0.8020
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_audio_r18img_pq_tcn_focal_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_tcn_focal_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9953, macro-F1=0.9942, precision=0.9953, recall=0.9932
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_audio_r18img_pq_tcn_focal_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_tcn_focal_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9769, macro-F1=0.9794, precision=0.9815, recall=0.9780
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_audio_r18img_pq_tcn_focal_5s

- model: `Audio R18 ImageNet + PQ TCN (Focal loss)`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.9254 ± 0.0861, macro-F1=0.9232 ± 0.0902, precision=0.9569 ± 0.0450, recall=0.9244 ± 0.0868
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

