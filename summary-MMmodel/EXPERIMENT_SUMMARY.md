# summary-MMmodel Experiment Index

当前 `summary-MMmodel` 的默认主线已经切到：

- `hcaf_audio_r18img_pq_xattn_5s`
- `audio = ResNet18(ImageNet init)`
- `PQ = TCN`
- `PQ + audio cross-attention`
- primary metric = `window macro-F1`

## Current Final Model

- model: `hcaf_audio_r18img_pq_xattn_5s`
- source: `summary-MMmodel/hcaf_audioresnet_xattn_vs_concat`
- window macro-F1: `0.9145 ± 0.0745`
- session macro-F1: `0.9407 ± 0.0838`
- audio encoder: `ResNet18 (ImageNet init)`

## Main Comparison

| model | source directory | window macro-F1 | session macro-F1 |
| --- | --- | ---: | ---: |
| `pressure_flow_5s` | `summary-MMmodel/hcaf_audioresnet_xattn_vs_concat` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` |
| `hcaf_audio_r18img_audio_only_5s` | `summary-MMmodel/hcaf_audioresnet_xattn_vs_concat` | `0.8709 ± 0.0722` | `0.9407 ± 0.0838` |
| `hcaf_audio_r18img_pq_directconcat_5s` | `summary-MMmodel/hcaf_audioresnet_xattn_vs_concat` | `0.7800 ± 0.1610` | `0.7852 ± 0.1923` |
| `hcaf_audio_r18img_pq_xattn_5s` | `summary-MMmodel/hcaf_audioresnet_xattn_vs_concat` | `0.9145 ± 0.0745` | `0.9407 ± 0.0838` |

## Key Result

当前最重要的正式结论是：

- `PQ + audio cross-attention` 在 `window-level` 上强于 `PQ-only`
- `PQ + audio cross-attention` 在 `window-level` 上强于 `audio-only`
- `PQ + audio cross-attention` 在 `window-level` 上强于 `direct concat`

也就是说，若采用窗口级判别作为主指标，当前已经实现：

> 最优模型是 `PQ + audio cross-attention`

## What Was Kept

- 保留:
  - `hcaf_audioresnet_xattn_vs_concat`
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

- 论文、汇报和后续实现说明优先引用 `summary-MMmodel/hcaf_audioresnet_xattn_vs_concat`
- 若继续优化，优先方向是增强 sensor 分支，而不是继续扩大 audio encoder
