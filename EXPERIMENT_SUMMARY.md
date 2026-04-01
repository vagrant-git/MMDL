# Experiment Summary

本文件已重组为“当前最终模型索引”，旧的逐轮长列表结果保留在各子目录的 `EXPERIMENT_SUMMARY.md` 与 `report.md` 中。

- python_env: `dl`
- final_model: `hcaf_confgate_residual_pcen96hp80_5s`
- final_config: `configs/hcaf_confgate_improve_search.yaml`
- final_evidence_config: `configs/final_model_unified_evidence.yaml`
- main_report: `report.md`

## Final Status (2026-04-01)

| candidate | source | window macro-F1 | session macro-F1 | note |
| --- | --- | ---: | ---: | --- |
| `audio_only_pcen96hp80_5s` | `summary-MMmodel/final_model_unified_evidence` | `0.7052 ± 0.0667` | `0.8296 ± 0.1362` | 统一口径下的音频单模态对照 |
| `pressure_flow_5s` | `summary-MMmodel/final_model_unified_evidence` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` | 统一口径下的 PQ-only 对照 |
| `hcaf_confgate_residual_pcen96hp80_5s` | `summary-MMmodel/final_model_unified_evidence` | `0.9207 ± 0.0261` | `0.9407 ± 0.0838` | 当前最终模型 |

## What Stayed

- 保留最终模型: `HCAF + confidence-aware gate + expert residual + PCEN96 + HP80`
- 保留关键证据:
  - `summary-MMmodel/final_model_unified_evidence`: 统一 split 下证明最终模型高于 `audio-only` 与 `pressure+flow-only`，并补齐缺失模态结果
  - `summary-MMmodel/pq_vs_multimodal_check`: 证明多模态已优于 PQ-only
  - `summary-MMmodel/hcaf_fusion_gate_followup`: 证明增益来自融合机制，不是自然出现
  - `summary-MMmodel/hcaf_confgate_improve_search`: 证明真正有效的进一步提升来自 `PCEN96 + HP80`
  - `summary-MMmodel/hcaf_confgate_filter_lowpass300`: 证明 `LP300/BP80-300` 不如 `HP80`
  - `summary-MMmodel/hcaf_arch_search` 与 `summary-MMmodel/hcaf_resnet18_imagenet_only`: 证明更大编码器、batch size、attention 长度、ResNet18 迁移学习都未超过当前 best
  - `summary-MMmodel/hcaf_confgate_interpretability`: 给出 gate 行为、误差结构与边界效应证据

## 2026-04-01 Extra Iterations

- `configs/hcaf_moddrop_search.yaml`
  - change: 将最终模型的 `modality_dropout` 从 `0.1` 改为 `0.0`
  - fold1 result: window macro-F1=`0.8344`, session macro-F1=`0.8222`
  - decision: 提前停止；即使后两折满分，最终 session 均值也只能追平当前 best，不能严格超过
- `configs/hcaf_loss_search.yaml`
  - change: 在最终模型上改用 `focal loss (gamma=1.5)`
  - fold1 result: window macro-F1=`0.8213`, session macro-F1=`0.8222`
  - decision: 提前停止；同样无法严格超过当前 best

## Archived

- 旧的 `Audio R18 ImageNet + PQ TCN` 线路不再作为顶层“latest model”描述保留，只作为并行探索分支归档在 `outputs/` 与对应总结文件中。
- 顶层推荐引用口径统一为 `hcaf_confgate_residual_pcen96hp80_5s`。
- 若正文或答辩需要最简洁的一页证据，优先引用 `summary-MMmodel/final_model_unified_evidence`。
