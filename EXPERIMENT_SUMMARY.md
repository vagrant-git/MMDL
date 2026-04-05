# Experiment Summary

当前默认口径已经统一切到：

- display_name: `HCAF-PCEN-DualXAttn`
- experiment_id: `hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s`
- primary_metric: `window macro-F1`
- main_config: `configs/final_model_unified_evidence.yaml`
- main_result_dir: `summary-MMmodel/final_model_unified_evidence`
- main_report: `report.md`

唯一正式默认模型的短声明见 [MODEL_IDENTITY.md](/home/wangshuai/MMDL/MODEL_IDENTITY.md)。

## Final Status

| candidate | window macro-F1 | session macro-F1 | note |
| --- | ---: | ---: | --- |
| `Audio-only PCEN96 HP80` | `0.7052 ± 0.0667` | `0.8296 ± 0.1362` | audio single-modality baseline |
| `Pressure+Flow-only` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` | PQ-only baseline |
| `HCAF-PCEN-DualXAttn` | `0.9196 ± 0.0469` | `0.8815 ± 0.0838` | current default model |
| `HCAF-PCEN-DualXAttn without audio` | `0.8872 ± 0.0145` | `0.8815 ± 0.0838` | missing-audio ablation |
| `HCAF-PCEN-DualXAttn without pressure` | `0.9379 ± 0.0221` | `0.9407 ± 0.0838` | missing-pressure ablation |
| `HCAF-PCEN-DualXAttn without flow` | `0.8501 ± 0.1403` | `0.9407 ± 0.0838` | missing-flow ablation |

## Why This Model

当前选择 `HCAF-PCEN-DualXAttn` 作为默认主模型，原因是：

- 保留了最核心的 `PCEN96 + HP80` 音频前端
- 保留了 `Pressure-Flow` 内部交互和 `audio-sensor` 双阶段 cross-attention
- 保留了 `confidence-aware gate + expert residual`
- 在统一证据表中，window-level macro-F1 同时高于 `audio_only` 和 `pressure_flow`

窗口级差值：

- vs `audio_only`: `+0.2144`
- vs `pressure_flow`: `+0.1697`
- vs `minus_audio`: `+0.0324`
- vs `minus_flow`: `+0.0695`

## Key Evidence

- [`summary-MMmodel/final_model_unified_evidence/EXPERIMENT_SUMMARY.md`](summary-MMmodel/final_model_unified_evidence/EXPERIMENT_SUMMARY.md)
  当前默认模型的统一正式证据链

- [`outputs/hcaf_confgate_compression_search/PARTIAL_RESULTS.md`](outputs/hcaf_confgate_compression_search/PARTIAL_RESULTS.md)
  当前默认模型为什么采用 `SA=0 + no-summary` 的压缩补充实验结果

- [`EXPERIMENT_RESULTS_ALL.md`](EXPERIMENT_RESULTS_ALL.md)
  所有已整理实验结果的一页总表

- [`summary-MMmodel/breath_cycle_analysis.json`](summary-MMmodel/breath_cycle_analysis.json)
  呼吸周期统计，用于支持周期级探索的生理依据

## Current Recommendation

- 当前文档、汇报和后续实现说明统一围绕 `HCAF-PCEN-DualXAttn`
- 当前主指标统一用 `window macro-F1`
- 历史实验目录仍保留，但不再作为默认说明口径
