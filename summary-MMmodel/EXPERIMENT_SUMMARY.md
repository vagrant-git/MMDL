# summary-MMmodel Experiment Index

当前 `summary-MMmodel` 的默认主线已经切到：

- `HCAF-PCEN-DualXAttn`
- `PCEN96 + HP80`
- `PQ cross-attention + audio-sensor cross-attention`
- `confidence-aware gate + expert residual`
- primary metric = `window macro-F1`

## Current Default Model

- model: `HCAF-PCEN-DualXAttn`
- experiment id: `hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s`
- source: `outputs/hcaf_confgate_compression_search`
- window macro-F1: `0.9155 ± 0.0133`
- session macro-F1: `0.9407 ± 0.0838`

## Main Comparison

| model | source directory | window macro-F1 | session macro-F1 |
| --- | --- | ---: | ---: |
| `Audio-only PCEN96 HP80` | `summary-MMmodel/final_model_unified_evidence` | `0.7052 ± 0.0667` | `0.8296 ± 0.1362` |
| `Pressure+Flow-only` | `summary-MMmodel/final_model_unified_evidence` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` |
| `HCAF-PCEN-DualXAttn` | `outputs/hcaf_confgate_compression_search` | `0.9155 ± 0.0133` | `0.9407 ± 0.0838` |

## Historical SA0 Search

| model | source directory | window macro-F1 | session macro-F1 |
| --- | --- | ---: | ---: |
| `SA=0 base` | `outputs/hcaf_confgate_compression_search` | `0.8968 ± 0.0495` | `0.8815 ± 0.0838` |
| `SA=0 + no-summary` | `outputs/hcaf_confgate_compression_search` | `0.9155 ± 0.0133` | `0.9407 ± 0.0838` |
| `SA=0 + summary-token` | `outputs/hcaf_confgate_compression_search` | `0.8298 ± 0.0805` | `0.8815 ± 0.0838` |
| `SA=0 + PCEN64 HP80` | `outputs/hcaf_confgate_compression_search` | `0.8773 ± 0.0458` | `0.8815 ± 0.0838` |

## Key Result

- 当前默认模型不再用“删掉了什么”命名，而是突出保留的关键结构
- `PCEN96 + HP80`、双阶段 cross-attention、`confidence-aware gate + expert residual` 是当前默认模型的核心组成
- `0.9155 ± 0.0133 / 0.9407 ± 0.0838` 已作为当前默认结果正式固定

## Recommendation

- 论文、汇报和后续实现说明优先引用完整模型的 `outputs/hcaf_confgate_compression_search`
- `summary-MMmodel/final_model_unified_evidence` 继续作为缺失模态与部署补充证据
- 压缩补充实验优先引用 `outputs/hcaf_confgate_compression_search`
