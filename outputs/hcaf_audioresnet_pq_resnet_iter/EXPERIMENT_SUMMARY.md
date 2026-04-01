## summary_mmmodel/hcaf_audio_r18img_pq_r18_longtokens_attn2_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_r18_longtokens_attn2_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8180, macro-F1=0.8128, precision=0.8975, recall=0.8160
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_audio_r18img_pq_r18_longtokens_attn2_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_r18_longtokens_attn2_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9320, macro-F1=0.9112, precision=0.9397, recall=0.9022
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_audio_r18img_pq_r18_longtokens_attn2_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_r18_longtokens_attn2_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9715, macro-F1=0.9739, precision=0.9781, recall=0.9706
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_audio_r18img_pq_r18_longtokens_attn2_5s

- model: `Audio R18 ImageNet + PQ R18 long tokens + attn2`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.9071 ± 0.0651, macro-F1=0.8993 ± 0.0663, precision=0.9384 ± 0.0329, recall=0.8963 ± 0.0633
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

