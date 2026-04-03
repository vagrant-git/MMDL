## summary_mmmodel/hcaf_confgate_residual_logmel96_sa0_nosummary_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_logmel96_sa0_nosummary_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8000, macro-F1=0.7919, precision=0.8868, recall=0.7989
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_confgate_residual_logmel96_sa0_nosummary_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_logmel96_sa0_nosummary_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9590, macro-F1=0.9505, precision=0.9582, recall=0.9468
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_confgate_residual_logmel96_sa0_nosummary_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_confgate_residual_logmel96_sa0_nosummary_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9375, macro-F1=0.9025, precision=0.8853, recall=0.9471
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_confgate_residual_logmel96_sa0_nosummary_5s

- model: `HCAF-LogMel96-DualXAttn`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.8988 ± 0.0704, macro-F1=0.8816 ± 0.0664, precision=0.9101 ± 0.0340, recall=0.8976 ± 0.0698
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

