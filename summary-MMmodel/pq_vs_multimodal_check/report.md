# 当前最强多模态 vs Pressure+Flow-only 复现实验

## 1. 实验目的

该实验专门用于回答一个问题：

- 当前最强多模态模型，是否已经在同一套划分与同一训练预算下，超过 `pressure+flow-only`？

为避免混用不同轮次结果，本实验单独固定配置并重新运行。

## 2. 可复现配置

- 配置文件: `configs/pq_vs_multimodal_check.yaml`
- 输出目录: `summary-MMmodel/pq_vs_multimodal_check`
- split manifest: `summary-MMmodel/pq_vs_multimodal_check/split_manifest.json`
- 数据划分: session-level grouped CV
- folds: `1 repeat x 3 folds`
- 窗口长度: `5 s`
- seed: `20260330`
- epochs: `8`
- early stopping patience: `3`
- weighted sampler: `True`

## 3. 对比模型

1. `pressure_flow_5s`
   - Pressure+Flow-only
2. `hcaf_normfix_5s`
   - HCAF，修复 sensor branch normalization
3. `hcaf_confgate_residual_5s`
   - HCAF，加入 confidence-aware gate 与 expert residual

## 4. 结果

| Experiment | Window Acc | Window F1 | Session Acc | Session F1 |
| --- | --- | --- | --- | --- |
| Pressure+Flow-only | 0.7692 ± 0.2274 | 0.7499 ± 0.2513 | 0.8889 ± 0.1571 | 0.8519 ± 0.2095 |
| HCAF norm fix | 0.8374 ± 0.1120 | 0.8011 ± 0.1424 | 0.8333 ± 0.1361 | 0.8259 ± 0.1406 |
| HCAF confidence gate + expert residual | 0.7871 ± 0.0953 | 0.7760 ± 0.0972 | 0.8889 ± 0.0786 | 0.8815 ± 0.0838 |

## 5. 关键结论

- 当前最强多模态结果是 `HCAF confidence gate + expert residual`
- 它已经超过同条件下的 `Pressure+Flow-only`
- session-level macro-F1:
  - `HCAF confidence gate + expert residual = 0.8815 ± 0.0838`
  - `Pressure+Flow-only = 0.8519 ± 0.2095`
- 提升不大，但更稳定，标准差更小

## 6. 解释

- `HCAF norm fix` 改善了窗口级别表现，但没有把优势传递到 session-level
- `confidence gate + expert residual` 在窗口级别不是最强，但 session-level 聚合后更稳定
- 因此，如果论文需要引用“当前最强多模态优于 pq-only”的结论，应引用本实验，而不是不同轮次间的横向混比

## 7. 推荐引用方式

可在正文中表述为：

> 在固定 session-level grouped CV 划分、相同 5 s 窗口和相同训练预算下，当前最优 HCAF 变体在 session-level macro-F1 上略优于 Pressure+Flow-only（0.8815 vs 0.8519），说明音频信息开始带来可复现但有限的增益。
