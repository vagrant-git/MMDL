# MMDL

多模态液体体积分类实验仓库。当前默认主线已经统一切到论文展示名 `HCAF-PCEN-DualXAttn`。

唯一正式默认模型说明见 [MODEL_IDENTITY.md](/home/wangshuai/MMDL/MODEL_IDENTITY.md)。如果其他文档里还有旧的 `logmel` 表述，以该文件和本页为准。

- experiment id: `hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s`
- task: `0 / 2 / 4` 三分类
- default metric: `window macro-F1`
- current default config: `configs/final_model_unified_evidence.yaml`
- current default result dir: `summary-MMmodel/final_model_unified_evidence`

## Current Default

- display name: `HCAF-PCEN-DualXAttn`
- audio frontend: `PCEN96 + HP80`
- interaction: `PQ cross-attention + audio-sensor cross-attention`
- decision: `confidence-aware gate + expert residual`
- compression choices:
  - `self_attention_layers = 0`
  - `use_summary_in_repr = false`

## Main Result

| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `Audio-only PCEN96 HP80` | `0.7052 ± 0.0667` | `0.8296 ± 0.1362` |
| `Pressure+Flow-only` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` |
| `HCAF-PCEN-DualXAttn` | `0.9196 ± 0.0469` | `0.8815 ± 0.0838` |

当前结论：

- 默认最佳模型仍明显优于 `audio_only` 和 `pressure_flow`
- `PCEN` 当前仍建议与 `HP80` 一起保留
- `confidence-aware gate + expert residual` 不建议继续删除

## Markdown Files

- [`summary.md`](summary.md)
  当前默认最佳模型的技术说明，适合先读。

- [`MODEL_IDENTITY.md`](MODEL_IDENTITY.md)
  当前唯一正式默认模型的短声明，优先级最高。

- [`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md)
  当前默认结果的最短摘要，适合快速确认主结论。

- [`EXPERIMENT_RESULTS_ALL.md`](EXPERIMENT_RESULTS_ALL.md)
  顶层实验总表，汇总当前已整理结果。

- [`report.md`](report.md)
  完整实验过程记录，适合回查“为什么这样改”和“哪些方向已被否定”。

- [`thesis_model_architecture_draft.md`](thesis_model_architecture_draft.md)
  面向论文正文的结构说明草稿。

## Repository Guide

```text
MMDL/
├── configs/
├── mmdl_baseline/
├── outputs/
├── summary-MMmodel/
├── summary.md
├── EXPERIMENT_SUMMARY.md
├── EXPERIMENT_RESULTS_ALL.md
├── report.md
└── thesis_model_architecture_draft.md
```

## Quick Start

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
python summary_mmmodel_experiments.py --config configs/final_model_unified_evidence.yaml
```
