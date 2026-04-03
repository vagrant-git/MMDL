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
- source: `summary-MMmodel/final_model_unified_evidence`
- window macro-F1: `0.9196 ± 0.0469`
- session macro-F1: `0.8815 ± 0.0838`

## Main Comparison

| model | source directory | window macro-F1 | session macro-F1 |
| --- | --- | ---: | ---: |
| `Audio-only PCEN96 HP80` | `summary-MMmodel/final_model_unified_evidence` | `0.7052 ± 0.0667` | `0.8296 ± 0.1362` |
| `Pressure+Flow-only` | `summary-MMmodel/final_model_unified_evidence` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` |
| `HCAF-PCEN-DualXAttn` | `summary-MMmodel/final_model_unified_evidence` | `0.9196 ± 0.0469` | `0.8815 ± 0.0838` |

## Compression Follow-up

| model | source directory | window macro-F1 | session macro-F1 |
| --- | --- | ---: | ---: |
| `SA=0 base` | `outputs/hcaf_confgate_compression_search` | `0.8968 ± 0.0495` | `0.8815 ± 0.0838` |
| `SA=0 + no-summary` | `outputs/hcaf_confgate_compression_search` | `0.9155 ± 0.0133` | `0.9407 ± 0.0838` |
| `SA=0 + summary-token` | `outputs/hcaf_confgate_compression_search` | `0.8298 ± 0.0805` | `0.8815 ± 0.0838` |
| `SA=0 + PCEN64 HP80` | `outputs/hcaf_confgate_compression_search` | `0.8773 ± 0.0458` | `0.8815 ± 0.0838` |

## Key Result

- 当前默认模型不再用“删掉了什么”命名，而是突出保留的关键结构
- `PCEN96 + HP80`、双阶段 cross-attention、`confidence-aware gate + expert residual` 是当前应保留的核心
- 在统一证据表里，`HCAF-PCEN-DualXAttn` 是当前默认最佳口径

## Recommendation

- 论文、汇报和后续实现说明优先引用 `summary-MMmodel/final_model_unified_evidence`
- 压缩补充实验优先引用 `outputs/hcaf_confgate_compression_search`
