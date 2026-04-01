## summary_mmmodel | hcaf_audio_r18img_audio_only_5s

- model: `Audio R18 ImageNet audio only 5 s`
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

## summary_mmmodel/hcaf_audio_r18img_pq_directconcat_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_directconcat_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8057, macro-F1=0.7983, precision=0.8936, recall=0.8031
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_audio_r18img_pq_directconcat_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_directconcat_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.7644, macro-F1=0.5744, precision=0.5087, recall=0.6595
- result_session: acc=0.6667, macro-F1=0.5333, precision=0.4444, recall=0.6667

## summary_mmmodel/hcaf_audio_r18img_pq_directconcat_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_directconcat_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9823, macro-F1=0.9675, precision=0.9710, recall=0.9644
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_audio_r18img_pq_directconcat_5s

- model: `Audio R18 + PQ direct concat 5 s`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.8508 ± 0.0945, macro-F1=0.7800 ± 0.1610, precision=0.7911 ± 0.2022, recall=0.8090 ± 0.1245
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8333 ± 0.1361, macro-F1=0.7852 ± 0.1923, precision=0.7778 ± 0.2400, recall=0.8333 ± 0.1361

## summary_mmmodel | hcaf_audio_r18img_pq_xattn_5s

- model: `Audio R18 + PQ cross-attention 5 s`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.9176 ± 0.0722, macro-F1=0.9145 ± 0.0745, precision=0.9448 ± 0.0350, recall=0.9154 ± 0.0723
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

