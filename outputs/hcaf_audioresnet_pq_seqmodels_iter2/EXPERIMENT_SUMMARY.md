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

## summary_mmmodel/hcaf_audio_r18img_pq_tcn_sched_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_tcn_sched_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8033, macro-F1=0.7949, precision=0.8927, recall=0.8009
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_audio_r18img_pq_tcn_sched_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_tcn_sched_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9672, macro-F1=0.9584, precision=0.9698, recall=0.9521
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_audio_r18img_pq_tcn_sched_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_tcn_sched_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9796, macro-F1=0.9800, precision=0.9860, recall=0.9746
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_audio_r18img_pq_tcn_sched_5s

- model: `Audio R18 ImageNet + PQ TCN (ReduceLROnPlateau)`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.9167 ± 0.0804, macro-F1=0.9111 ± 0.0826, precision=0.9495 ± 0.0407, recall=0.9092 ± 0.0771
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

