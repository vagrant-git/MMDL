# summary-MMmodel Experiment Index

本文件改为 `summary-MMmodel` 目录的压缩索引，用来替代此前冗长且包含大量旧模型描述的逐条拼接版本。各子目录中的原始 `EXPERIMENT_SUMMARY.md` 仍作为归档保留。

## Current Final Model

- model: `hcaf_confgate_residual_pcen96hp80_5s`
- source: `summary-MMmodel/hcaf_confgate_improve_search`
- unified_evidence: `summary-MMmodel/final_model_unified_evidence`
- window macro-F1: `0.9207 ± 0.0261`
- session macro-F1: `0.9407 ± 0.0838`
- best_session_method: `majority_voting`

## Key Evidence Chain

| purpose | source directory | key result |
| --- | --- | --- |
| 统一口径下证明 final model 高于单模态 / 双模态 | `summary-MMmodel/final_model_unified_evidence` | final multimodal=`0.9407`，高于 `audio-only` 的 `0.8296` 与 `pressure+flow-only` 的 `0.8519` |
| 证明最终多模态超过 PQ-only | `summary-MMmodel/pq_vs_multimodal_check` | `hcaf_confgate_residual_5s` session macro-F1=`0.8815`，高于 `pressure_flow_5s` 的 `0.8519` |
| 证明有效机制来自 fusion repair | `summary-MMmodel/hcaf_fusion_gate_followup` | `confidence-aware gate` 单独使用会退化；`gate + expert residual` 才稳定 |
| 证明真正刷新的提升来自音频前端 | `summary-MMmodel/hcaf_confgate_improve_search` | `PCEN96 + HP80` 将 session macro-F1 从 `0.8815` 提升到 `0.9407` |
| 证明为何保留 `HP80` | `summary-MMmodel/hcaf_confgate_filter_lowpass300` | `LP300` 与 `BP80-300` 都低于 `HP80` |
| 证明更大结构没有继续超过 | `summary-MMmodel/hcaf_arch_search` | batch size、attention 长度、ResNet encoder 均未超过当前 best |
| 证明 ResNet18 迁移学习只能追平 | `summary-MMmodel/hcaf_resnet18_imagenet_only` | `ImageNet` 初始化可到 `0.9407`，但未超过当前 best |
| 解释 gate 行为与错误结构 | `summary-MMmodel/hcaf_confgate_interpretability` | audio gate 与 audio confidence 相关系数=`0.695`；主要混淆是 `0 -> 2` |

## What Was Kept

- 保留:
  - `final_model_unified_evidence`
  - `pq_vs_multimodal_check`
  - `hcaf_fusion_gate_followup`
  - `hcaf_confgate_improve_search`
  - `hcaf_confgate_filter_lowpass300`
  - `hcaf_arch_search`
  - `hcaf_resnet18_imagenet_only`
  - `hcaf_confgate_interpretability`
- 弱化为背景或归档:
  - 早期 `summary_mmmodel` 主表
  - `audio_frontend_search` / `audio_targeted_search` 的 audio-only 长表
  - 旧 `Audio R18 ImageNet + PQ TCN` 顶层“latest model”描述

## 2026-04-01 Extra Checks

- `hcaf_moddrop_search`
  - check: `modality_dropout=0.0`
  - fold1 result: window macro-F1=`0.8344`, session macro-F1=`0.8222`
  - conclusion: 不能严格超过当前 best，提前停止
- `hcaf_loss_search`
  - check: `focal loss, gamma=1.5`
  - fold1 result: window macro-F1=`0.8213`, session macro-F1=`0.8222`
  - conclusion: 同样不能严格超过当前 best，提前停止

## Recommendation

- 论文、答辩和后续实现说明统一以 `hcaf_confgate_residual_pcen96hp80_5s` 为主模型。
- 若需要单页展示“为什么这个模型就是最终模型”，优先引用 `summary-MMmodel/final_model_unified_evidence`。
- 若需要原始逐折明细，直接查看对应子目录的 `EXPERIMENT_SUMMARY.md`、`fold_results.csv` 与 `overall_results.csv`。
