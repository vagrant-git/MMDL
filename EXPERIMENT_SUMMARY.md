# Experiment Summary

当前默认口径已经统一切到：

- final_model: `hcaf_audio_r18img_pq_xattn_5s`
- primary_metric: `window macro-F1`
- audio_encoder: `ResNet18 (ImageNet init)`
- main_config: `configs/hcaf_audioresnet_xattn_vs_concat.yaml`
- main_result_dir: `summary-MMmodel/hcaf_audioresnet_xattn_vs_concat`
- main_report: `report.md`

## Final Status

| candidate | window macro-F1 | session macro-F1 | note |
| --- | ---: | ---: | --- |
| `pressure_flow_5s` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` | PQ-only baseline |
| `hcaf_audio_r18img_audio_only_5s` | `0.8709 ± 0.0722` | `0.9407 ± 0.0838` | ResNet18 audio-only baseline |
| `hcaf_audio_r18img_pq_directconcat_5s` | `0.7800 ± 0.1610` | `0.7852 ± 0.1923` | direct concat baseline |
| `hcaf_audio_r18img_pq_xattn_5s` | `0.9145 ± 0.0745` | `0.9407 ± 0.0838` | current final model |

## Why This Model

当前选择 `hcaf_audio_r18img_pq_xattn_5s` 作为默认主模型，原因是：

- 满足 `audio encoder = ResNet18(ImageNet init)` 的约束
- 在 `window-level` 上强于 `PQ-only`
- 在 `window-level` 上强于 `audio-only`
- 在 `window-level` 上强于 `direct concat PQ+audio`

窗口级差值：

- vs `PQ-only`: `+0.1646`
- vs `audio-only`: `+0.0436`
- vs `direct concat`: `+0.1345`

## Key Evidence

- [`summary-MMmodel/hcaf_audioresnet_xattn_vs_concat`](summary-MMmodel/hcaf_audioresnet_xattn_vs_concat)  
  直接证明 `cross-attention` 强于 `PQ-only` / `audio-only` / `direct concat`

- [`summary-MMmodel/hcaf_audioresnet_pq_seqmodels`](summary-MMmodel/hcaf_audioresnet_pq_seqmodels)  
  证明 `ResNet18 + PQ TCN` 是当前最稳的 `ResNet18` 主线

- [`summary-MMmodel/hcaf_audioresnet_fixed_window_smoke`](summary-MMmodel/hcaf_audioresnet_fixed_window_smoke)  
  说明固定更长非对齐窗在当前 fold1 上没有显示出优于 `5 s` 的趋势

- [`summary-MMmodel/breath_cycle_analysis.json`](summary-MMmodel/breath_cycle_analysis.json)  
  说明呼吸周期约为 `4 s`，用于支持周期级探索的生理依据

## Current Recommendation

- 当前文档、汇报和后续实现说明统一围绕 `hcaf_audio_r18img_pq_xattn_5s`
- 当前主指标统一用 `window macro-F1`
- 旧的 `basic audio encoder` 最终模型口径已归档，不再作为默认说明
