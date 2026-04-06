## summary_mmmodel/hcaf_audio_r18img_pq_xattn_sa0_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_xattn_sa0_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8180, macro-F1=0.7959, precision=0.8573, recall=0.8167
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_audio_r18img_pq_xattn_sa0_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_xattn_sa0_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9379, macro-F1=0.9217, precision=0.9371, recall=0.9160
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_audio_r18img_pq_xattn_sa0_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_xattn_sa0_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9878, macro-F1=0.9879, precision=0.9889, recall=0.9872
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_audio_r18img_pq_xattn_sa0_5s

- model: `Audio R18 + PQ cross-attention 5 s without joint self-attention`
- modality: `multimodal`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.9145 ± 0.0713, macro-F1=0.9018 ± 0.0796, precision=0.9278 ± 0.0541, recall=0.9066 ± 0.0699
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.0786, macro-F1=0.8815 ± 0.0838, precision=0.9259 ± 0.0524, recall=0.8889 ± 0.0786

## summary_mmmodel/hcaf_audio_r18img_pq_xattn_sa1_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_xattn_sa1_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8604, macro-F1=0.8629, precision=0.9128, recall=0.8583
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_audio_r18img_pq_xattn_sa1_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_xattn_sa1_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9355, macro-F1=0.9197, precision=0.9454, recall=0.9109
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_audio_r18img_pq_xattn_sa1_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_audio_r18img_pq_xattn_sa1_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9633, macro-F1=0.9564, precision=0.9741, recall=0.9417
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_audio_r18img_pq_xattn_sa1_5s

- model: `Audio R18 + PQ cross-attention 5 s with joint self-attention`
- modality: `multimodal`
- group: `baseline`
- window_sec: `5.0`
- window_mean_std: acc=0.9197 ± 0.0435, macro-F1=0.9130 ± 0.0385, precision=0.9441 ± 0.0251, recall=0.9036 ± 0.0344
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.0786, macro-F1=0.8815 ± 0.0838, precision=0.9259 ± 0.0524, recall=0.8889 ± 0.0786

