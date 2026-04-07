# MMDL

多模态液体体积分类实验仓库。当前默认主线已经统一到论文展示名 `HCAF-PCEN-DualXAttn`。

唯一正式默认模型说明见 [MODEL_IDENTITY.md](/home/oi/MMDL/MODEL_IDENTITY.md)。如果其他文档里还有旧的 `logmel` 表述或把 `0.8225` 写成默认结果，以该文件和本页为准。

- experiment lineage: `SA=0 + no-summary`
- task: `0 / 2 / 4` 三分类
- default metric: `window macro-F1`
- aligned runtime config: `configs/final_model_unified_evidence.yaml`
- canonical full-model result dir: `outputs/hcaf_confgate_compression_search`
- supplementary evidence dir: `summary-MMmodel/final_model_unified_evidence`

## Current Default

- display name: `HCAF-PCEN-DualXAttn`
- audio frontend: `PCEN96 + HP80`
- interaction: `PQ cross-attention + audio-sensor cross-attention`
- decision: `confidence-aware gate + expert residual`
- compression: `self_attention_layers = 0`, `use_summary_in_repr = false`

## Main Result

| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `Audio-only PCEN96 HP80` | `0.7052 ± 0.0667` | `0.8296 ± 0.1362` |
| `Pressure+Flow-only` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` |
| `HCAF-PCEN-DualXAttn` | `0.9155 ± 0.0133` | `0.9407 ± 0.0838` |

当前结论：

- 当前默认模型的完整模型正式成绩统一引用 `outputs/hcaf_confgate_compression_search`
- `summary-MMmodel/final_model_unified_evidence` 继续作为缺失模态和部署对齐的补充证据目录
- `PCEN96 + HP80`、`self_attention_layers = 0`、`use_summary_in_repr = false` 仍是当前默认配置

## Quick Start

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
python summary_mmmodel_experiments.py --config configs/final_model_unified_evidence.yaml
```
