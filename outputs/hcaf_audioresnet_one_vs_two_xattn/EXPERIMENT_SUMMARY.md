## summary_mmmodel/hcaf_audio_r18img_pq_one_xattn_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_one_xattn_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8090, macro-F1=0.8025, precision=0.8956, recall=0.8067
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_audio_r18img_pq_one_xattn_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_one_xattn_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.8980, macro-F1=0.8635, precision=0.9198, recall=0.8529
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_audio_r18img_pq_one_xattn_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_one_xattn_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9715, macro-F1=0.9527, precision=0.9798, recall=0.9322
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_audio_r18img_pq_one_xattn_5s

- model: `Audio R18 + PQ one cross-attention stage 5 s`
- modality: `multimodal`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.8928 ± 0.0664, macro-F1=0.8729 ± 0.0617, precision=0.9317 ± 0.0354, recall=0.8639 ± 0.0518
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.0786, macro-F1=0.8815 ± 0.0838, precision=0.9259 ± 0.0524, recall=0.8889 ± 0.0786

## summary_mmmodel/hcaf_audio_r18img_pq_two_xattn_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_two_xattn_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8204, macro-F1=0.8166, precision=0.8970, recall=0.8180
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_audio_r18img_pq_two_xattn_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_two_xattn_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9766, macro-F1=0.9711, precision=0.9764, recall=0.9675
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_audio_r18img_pq_two_xattn_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_two_xattn_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9769, macro-F1=0.9598, precision=0.9463, recall=0.9777
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_audio_r18img_pq_two_xattn_5s

- model: `Audio R18 + PQ two cross-attention stages 5 s`
- modality: `multimodal`
- group: `baseline`
- window_sec: `5.0`
- window_mean_std: acc=0.9246 ± 0.0737, macro-F1=0.9158 ± 0.0703, precision=0.9399 ± 0.0327, recall=0.9211 ± 0.0730
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

