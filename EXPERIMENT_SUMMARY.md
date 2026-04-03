# Experiment Summary

当前默认口径已经统一切到：

- display_name: `HCAF-PCEN-XAttn`
- experiment_id: `hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s`
- primary_metric: `window macro-F1`
- audio_encoder: `ResNet18 (ImageNet init)`
- main_config: `configs/hcaf_confgate_compression_search.yaml`
- main_result_dir: `outputs/hcaf_confgate_compression_search`
- main_report: `report.md`

## Final Status

| candidate | window macro-F1 | session macro-F1 | note |
| --- | ---: | ---: | --- |
| `HCAF compressed base SA0 PCEN96 HP80` | `0.8968 ± 0.0495` | `0.8815 ± 0.0838` | SA0 base |
| `HCAF-PCEN-XAttn` | `0.9155 ± 0.0133` | `0.9407 ± 0.0838` | current final model |
| `HCAF compressed SA0 summary token attention` | `0.8298 ± 0.0805` | `0.8815 ± 0.0838` | summary-token variant |
| `HCAF compressed SA0 PCEN64 HP80` | `0.8773 ± 0.0458` | `0.8815 ± 0.0838` | frontend compression |

## Why This Model

当前选择 `HCAF-PCEN-XAttn` 作为默认主模型，原因是：

- 保留了最核心的 `PCEN96 + HP80` 音频前端
- 保留了 `Pressure-Flow` 内部交互和 `audio-sensor` 双阶段 cross-attention
- 保留了 `confidence-aware gate + expert residual`
- 在当前已完成压缩消融中取得最高 `window-level macro-F1`

窗口级差值：

- vs `SA0 base`: `+0.0187`
- vs `summary token`: `+0.0857`
- vs `PCEN64 HP80`: `+0.0382`

## Key Evidence

- [`outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md`](outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md)
  当前默认模型的直接证据链

- [`summary-MMmodel/hcaf_audioresnet_xattn_vs_concat/EXPERIMENT_SUMMARY.md`](summary-MMmodel/hcaf_audioresnet_xattn_vs_concat/EXPERIMENT_SUMMARY.md)
  说明上一阶段为什么切到 `ResNet18 + PQ cross-attention` 主线

- [`EXPERIMENT_RESULTS_ALL.md`](EXPERIMENT_RESULTS_ALL.md)
  所有已整理实验结果的一页总表

- [`summary-MMmodel/breath_cycle_analysis.json`](summary-MMmodel/breath_cycle_analysis.json)  
  说明呼吸周期约为 `4 s`，用于支持周期级探索的生理依据

## Current Recommendation

- 当前文档、汇报和后续实现说明统一围绕 `HCAF-PCEN-XAttn`
- 当前主指标统一用 `window macro-F1`
- 旧的 `basic audio encoder` 最终模型口径已归档，不再作为默认说明
