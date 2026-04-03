# summary-MMmodel Experiment Index

当前 `summary-MMmodel` 的默认主线已经切到：

- `HCAF-PCEN-XAttn`
- `audio = ResNet18(ImageNet init)`
- `PQ = TCN`
- `PQ + audio cross-attention`
- `confidence-aware gate + expert residual`
- primary metric = `window macro-F1`

## Current Final Model

- model: `HCAF-PCEN-XAttn`
- experiment id: `hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s`
- source: `outputs/hcaf_confgate_compression_search`
- window macro-F1: `0.9155 ± 0.0133`
- session macro-F1: `0.9407 ± 0.0838`
- audio encoder: `ResNet18 (ImageNet init)`

## Main Comparison

| model | source directory | window macro-F1 | session macro-F1 |
| --- | --- | ---: | ---: |
| `HCAF compressed base SA0 PCEN96 HP80` | `outputs/hcaf_confgate_compression_search` | `0.8968 ± 0.0495` | `0.8815 ± 0.0838` |
| `HCAF-PCEN-XAttn` | `outputs/hcaf_confgate_compression_search` | `0.9155 ± 0.0133` | `0.9407 ± 0.0838` |
| `HCAF compressed SA0 summary token attention` | `outputs/hcaf_confgate_compression_search` | `0.8298 ± 0.0805` | `0.8815 ± 0.0838` |
| `HCAF compressed SA0 PCEN64 HP80` | `outputs/hcaf_confgate_compression_search` | `0.8773 ± 0.0458` | `0.8815 ± 0.0838` |

## Key Result

当前最重要的正式结论是：

- 当前默认模型不再把“去掉了什么”写进展示名，而是强调保留下来的核心结构
- `PCEN96 + HP80`、双阶段 cross-attention、`confidence-aware gate + expert residual` 是当前应保留的核心
- 在已完成压缩消融中，`HCAF-PCEN-XAttn` 是当前最佳窗口级结果

也就是说，若采用窗口级判别作为主指标，当前已经实现：

> 当前默认模型是 `HCAF-PCEN-XAttn`

## What Was Kept

- 保留:
  - `hcaf_audioresnet_xattn_vs_concat`
  - `hcaf_confgate_compression_search`
  - `hcaf_audioresnet_pq_seqmodels`
  - `hcaf_audioresnet_fixed_window_smoke`
  - `breath_cycle_analysis.json`

- 作为探索归档保留，但不再作为默认口径:
  - `final_model_unified_evidence`
  - `hcaf_audioresnet_unified_evidence`
  - `hcaf_audioresnet_cycle_window_search`
  - `hcaf_audioresnet_cycle_aligned_search`
  - `hcaf_audioresnet_cycle_aligned_8s_evidence`

## Recommendation

- 论文、汇报和后续实现说明优先引用 `outputs/hcaf_confgate_compression_search`
- 若继续优化，优先方向是增强 sensor 分支，而不是继续扩大 audio encoder
