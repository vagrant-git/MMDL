## summary_mmmodel/pressure_flow_5s_baseline/repeat1_fold1 | pressure_flow

- python_env: `dl`
- model: `pressure_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/pressure_flow_5s_baseline/repeat1_fold1` 设定训练评估。
- result_window: acc=0.9478, macro-F1=0.9521, precision=0.9553, recall=0.9494
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/pressure_flow_5s_baseline/repeat1_fold2 | pressure_flow

- python_env: `dl`
- model: `pressure_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/pressure_flow_5s_baseline/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9527, macro-F1=0.9577, precision=0.9667, recall=0.9535
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/pressure_flow_5s_baseline/repeat1_fold3 | pressure_flow

- python_env: `dl`
- model: `pressure_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/pressure_flow_5s_baseline/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9149, macro-F1=0.8973, precision=0.9081, recall=0.8924
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | pressure_flow_5s_baseline

- model: `Pressure+Flow (gated)`
- modality: `pressure_flow`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.9385 ± 0.0168, macro-F1=0.9357 ± 0.0272, precision=0.9433 ± 0.0254, recall=0.9318 ± 0.0279
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

## summary_mmmodel/pressure_flow_5s_softmax_gate/repeat1_fold1 | pressure_flow

- python_env: `dl`
- model: `pressure_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/pressure_flow_5s_softmax_gate/repeat1_fold1` 设定训练评估。
- result_window: acc=0.9461, macro-F1=0.9531, precision=0.9533, recall=0.9532
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/pressure_flow_5s_softmax_gate/repeat1_fold2 | pressure_flow

- python_env: `dl`
- model: `pressure_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/pressure_flow_5s_softmax_gate/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9112, macro-F1=0.9170, precision=0.9387, recall=0.9127
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/pressure_flow_5s_softmax_gate/repeat1_fold3 | pressure_flow

- python_env: `dl`
- model: `pressure_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/pressure_flow_5s_softmax_gate/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9662, macro-F1=0.9683, precision=0.9675, recall=0.9713
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | pressure_flow_5s_softmax_gate

- model: `Pressure+Flow (softmax-gate)`
- modality: `pressure_flow`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.9412 ± 0.0227, macro-F1=0.9461 ± 0.0215, precision=0.9531 ± 0.0118, recall=0.9457 ± 0.0245
- best_session_method: `majority_voting`
- session_mean_std: acc=1.0000 ± 0.0000, macro-F1=1.0000 ± 0.0000, precision=1.0000 ± 0.0000, recall=1.0000 ± 0.0000

## summary_mmmodel/pressure_flow_5s_concat_mlp/repeat1_fold1 | pressure_flow

- python_env: `dl`
- model: `pressure_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/pressure_flow_5s_concat_mlp/repeat1_fold1` 设定训练评估。
- result_window: acc=0.9114, macro-F1=0.9220, precision=0.9233, recall=0.9211
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/pressure_flow_5s_concat_mlp/repeat1_fold2 | pressure_flow

- python_env: `dl`
- model: `pressure_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/pressure_flow_5s_concat_mlp/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9608, macro-F1=0.9589, precision=0.9599, recall=0.9615
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/pressure_flow_5s_concat_mlp/repeat1_fold3 | pressure_flow

- python_env: `dl`
- model: `pressure_flow`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/pressure_flow_5s_concat_mlp/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9189, macro-F1=0.9122, precision=0.9115, recall=0.9304
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | pressure_flow_5s_concat_mlp

- model: `Pressure+Flow (concat-mlp)`
- modality: `pressure_flow`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.9304 ± 0.0217, macro-F1=0.9310 ± 0.0201, precision=0.9316 ± 0.0206, recall=0.9376 ± 0.0173
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

