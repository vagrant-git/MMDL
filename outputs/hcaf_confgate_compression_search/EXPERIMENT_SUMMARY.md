## summary_mmmodel | hcaf_comp_sa0_base_5s

- model: `HCAF compressed base SA0 PCEN96 HP80`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.9067 ± 0.0542, macro-F1=0.8968 ± 0.0495, precision=0.9388 ± 0.0300, recall=0.8869 ± 0.0435
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.0786, macro-F1=0.8815 ± 0.0838, precision=0.9259 ± 0.0524, recall=0.8889 ± 0.0786

## summary_mmmodel | hcaf_comp_sa0_no_summary_5s

- model: `HCAF compressed SA0 without summary repr`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.9260 ± 0.0161, macro-F1=0.9155 ± 0.0133, precision=0.9435 ± 0.0139, recall=0.9028 ± 0.0079
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

## summary_mmmodel | hcaf_comp_sa0_summary_token_5s

- model: `HCAF compressed SA0 summary token attention`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.8557 ± 0.0789, macro-F1=0.8298 ± 0.0805, precision=0.8867 ± 0.0359, recall=0.8508 ± 0.0792
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.0786, macro-F1=0.8815 ± 0.0838, precision=0.9259 ± 0.0524, recall=0.8889 ± 0.0786

## summary_mmmodel | hcaf_comp_sa0_pcen64_hp80_5s

- model: `HCAF compressed SA0 PCEN64 HP80`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.8853 ± 0.0507, macro-F1=0.8773 ± 0.0458, precision=0.9062 ± 0.0301, recall=0.8667 ± 0.0507
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.0786, macro-F1=0.8815 ± 0.0838, precision=0.9259 ± 0.0524, recall=0.8889 ± 0.0786

## summary_mmmodel/hcaf_comp_sa0_pcen96_nofilter_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_comp_sa0_pcen96_nofilter_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9496, macro-F1=0.9447, precision=0.9532, recall=0.9382
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_comp_sa0_pcen96_nofilter_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_comp_sa0_pcen96_nofilter_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9321, macro-F1=0.9237, precision=0.9233, recall=0.9316
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_comp_sa0_pcen96_nofilter_5s

- model: `HCAF compressed SA0 PCEN96 no filter`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.8958 ± 0.0641, macro-F1=0.8891 ± 0.0644, precision=0.9236 ± 0.0240, recall=0.8909 ± 0.0623
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

## summary_mmmodel/hcaf_comp_sa0_simplegate_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_comp_sa0_simplegate_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8098, macro-F1=0.8040, precision=0.8820, recall=0.8054
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_comp_sa0_simplegate_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_comp_sa0_simplegate_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.7491, macro-F1=0.7679, precision=0.8490, recall=0.7778
- result_session: acc=0.6667, macro-F1=0.6667, precision=0.8333, recall=0.6667

## summary_mmmodel/hcaf_comp_sa0_simplegate_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_comp_sa0_simplegate_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9375, macro-F1=0.9202, precision=0.9300, recall=0.9144
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_comp_sa0_simplegate_5s

- model: `HCAF compressed SA0 simple gate no expert residual`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.8321 ± 0.0785, macro-F1=0.8307 ± 0.0650, precision=0.8870 ± 0.0332, recall=0.8325 ± 0.0590
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8333 ± 0.1361, macro-F1=0.8296 ± 0.1362, precision=0.9074 ± 0.0693, recall=0.8333 ± 0.1361

