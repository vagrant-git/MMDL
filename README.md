# MMDL

多模态液体体积分类实验仓库。当前主线任务是基于 `audio + pressure + flow` 三模态，在 `0 / 2 / 4` 三分类设置下，围绕已经确定的最优 HCAF 路线做结果固化、机制验证和可复现整理，而不是继续无边界地扩展模型分支。

线上仓库当前保留的是:

- 代码与配置
- 关键实验的汇总结果、图表和报告
- 当前最终模型的证据链

线上仓库当前不保留的是:

- 原始数据 `data/`
- 大体积 checkpoint
- 原始逐折训练中间目录

## Current Status

- final model: `hcaf_confgate_residual_pcen96hp80_5s`
- task: `0 / 2 / 4` three-class classification
- split: `session`-grouped `1 repeat x 3 folds`
- window length: `5 s`
- final selection date: `2026-04-01`

当前最终采用的不是参数最多的模型，而是证据链最完整、session-level 指标最高、且后续补充搜索没有再超过的那条主线:

| model | source | window macro-F1 | session macro-F1 |
| --- | --- | ---: | ---: |
| `audio_only_pcen96hp80_5s` | [`summary-MMmodel/final_model_unified_evidence`](summary-MMmodel/final_model_unified_evidence) | `0.7052 ± 0.0667` | `0.8296 ± 0.1362` |
| `pressure_flow_5s` | [`summary-MMmodel/final_model_unified_evidence`](summary-MMmodel/final_model_unified_evidence) | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` |
| `hcaf_confgate_residual_pcen96hp80_5s` | [`summary-MMmodel/final_model_unified_evidence`](summary-MMmodel/final_model_unified_evidence) | `0.9207 ± 0.0261` | `0.9407 ± 0.0838` |

这轮整理后的核心结论很明确:

- 最终多模态模型在同一 split、同一训练预算下，同时高于 `audio-only` 和 `pressure+flow-only`
- 相对 `audio-only` 的 session macro-F1 提升为 `+0.1111`
- 相对 `pressure+flow-only` 的 session macro-F1 提升为 `+0.0889`
- 真正把最终结果继续推高的关键不是更复杂主干，而是 `PCEN + 96 mel bins + 80 Hz high-pass`
- `ResNet18 ImageNet`、更大的 PQ 编码器、更多序列模型分支目前都没有超过当前 best
- 缺失模态鲁棒性现在也已补到与最终模型同一份 split manifest 下，见 [`summary-MMmodel/final_model_unified_evidence`](summary-MMmodel/final_model_unified_evidence)

## Core Work

目前仓库的核心工作描述可以概括为 4 条:

1. 固化最终主模型口径  
   顶层结论统一围绕 `hcaf_confgate_residual_pcen96hp80_5s` 展开，不再把旧的 `Audio ResNet18 + complex PQ encoder` 路线写成“latest model”。

2. 保留关键证据链  
   当前推荐引用的主证据链是:
   - [`summary-MMmodel/final_model_unified_evidence`](summary-MMmodel/final_model_unified_evidence): 统一口径下证明最终多模态优于 `audio-only` 与 `pressure+flow-only`，并补齐缺失模态结果
   - [`summary-MMmodel/pq_vs_multimodal_check`](summary-MMmodel/pq_vs_multimodal_check): 证明上一版 HCAF 首次稳定超过 `PQ-only`
   - [`summary-MMmodel/hcaf_fusion_gate_followup`](summary-MMmodel/hcaf_fusion_gate_followup): 证明收益来自融合机制本身
   - [`summary-MMmodel/hcaf_confgate_improve_search`](summary-MMmodel/hcaf_confgate_improve_search): 证明进一步提升来自 `PCEN96 + HP80`

3. 把补充搜索收敛成“是否还能超过当前 best”的判断  
   截至 `2026-04-01`，`modality_dropout=0.0` 和 `focal loss (gamma=1.5)` 两条补充迭代都已提前停止，没有严格超过当前最终模型。

4. 让线上仓库直接可读  
   代码、配置、报告、图表和汇总结果可以直接在 GitHub 查看；原始数据和大文件训练产物继续留在本地。

## Task And Evaluation Protocol

- label set: `0 / 2 / 4`
- modalities: `audio.wav` + `daq.csv` 中的 `Pressure (cmH2O)` 和 `Flowrate (L/min)`
- split unit: `session`
- leakage rule: 先按 `session` 划分，再在各 split 内切窗
- audio sample rate: `16000 Hz`
- sensor sample rate: `100 Hz`
- main window length: `5 s`
- main hop length: `5 s`
- grouped CV: `1 repeat x 3 folds`
- excluded session: `MMdata_265.10s_0322_224132_no_secretion`

所有主流程都会先读取配置里的 `session_filter`，再做标签映射、划分和训练。

## Repository Map

```text
MMDL/
├── configs/                  # 当前保留的实验配置
├── data/                     # 本地原始数据，不上传
├── mmdl_baseline/            # 数据集、预处理、模型与训练逻辑
├── outputs/                  # 历史分支的汇总结果与图表
├── summary-MMmodel/          # 当前主线实验、搜索与证据链
├── train.py                  # 单模型训练入口
├── grouped_cv.py             # grouped cross-validation
├── session_aggregation_cv.py # session-level 聚合评估
├── summary.md                # 最终模型技术说明
├── EXPERIMENT_SUMMARY.md     # 当前最终模型索引
└── report.md                 # 主线结果的文字化复盘
```

## What To Read First

如果你是第一次看这个仓库，建议按这个顺序读:

1. [`summary.md`](summary.md)  
   当前最终模型的技术说明、数据流和结构细节。

2. [`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md)  
   当前最终模型口径、关键保留证据和最近补充迭代结论。

3. [`report.md`](report.md)  
   用于写作和复盘的主线叙述。

4. [`summary-MMmodel/hcaf_confgate_improve_search`](summary-MMmodel/hcaf_confgate_improve_search)  
   当前 best 的直接来源。

5. [`summary-MMmodel/final_model_unified_evidence`](summary-MMmodel/final_model_unified_evidence)  
   用来说明“最终多模态已经同时超过 `audio-only` 与 `pressure+flow-only`，并给出统一 split 下的缺失模态结果”。

6. [`summary-MMmodel/pq_vs_multimodal_check`](summary-MMmodel/pq_vs_multimodal_check)  
   用来说明“上一版 HCAF 首次超过 PQ-only”的关键历史对照。

7. [`summary-MMmodel/hcaf_fusion_gate_followup`](summary-MMmodel/hcaf_fusion_gate_followup)  
   用来说明最终收益来自融合机制和 reliability gate，而不是偶然波动。

## Key Configs

- [`configs/summary_mmmodel.yaml`](configs/summary_mmmodel.yaml): `summary-MMmodel` 主实验入口
- [`configs/pq_vs_multimodal_check.yaml`](configs/pq_vs_multimodal_check.yaml): `PQ-only vs 多模态` 主对照
- [`configs/hcaf_fusion_gate_followup.yaml`](configs/hcaf_fusion_gate_followup.yaml): 融合机制补充验证
- [`configs/hcaf_confgate_improve_search.yaml`](configs/hcaf_confgate_improve_search.yaml): `PCEN96 + HP80` 提升来源
- [`configs/final_model_unified_evidence.yaml`](configs/final_model_unified_evidence.yaml): 最终统一证据配置，含 `audio-only / pressure+flow-only / final multimodal / missing-modality`
- [`configs/chapter4_024.yaml`](configs/chapter4_024.yaml): 第四章 `0/2/4` 主实验

## Environment

按仓库约定，训练、评估和报告脚本都在 `dl` 环境下运行:

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
```

如果日志里出现 `torch 2.x+cpu` 或 `cuda_available=False`，通常说明当前没有正确使用 GPU 环境。

## Quick Start

运行 `summary-MMmodel` 主实验与汇总:

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
python summary_mmmodel_experiments.py --config configs/summary_mmmodel.yaml
python generate_summary_mmmodel_report.py --config configs/summary_mmmodel.yaml
```

运行 grouped CV:

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
python grouped_cv.py --config configs/baseline.yaml
python generate_grouped_cv_report.py --config configs/baseline.yaml
```

在已有 grouped CV 基础上做 session-level aggregation:

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
python session_aggregation_cv.py --config configs/baseline.yaml
python generate_session_aggregation_report.py --config configs/baseline.yaml
```

运行第四章 `0/2/4` 三分类实验:

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
python chapter4_024_experiments.py --config configs/chapter4_024.yaml
python generate_chapter4_024_report.py --config configs/chapter4_024.yaml
```

## Online Repo Notes

为了让线上仓库保留“有用结果”而不是只剩代码，目前 `.gitignore` 的策略是:

- 继续忽略 `data/`
- 继续忽略 checkpoint 和大体积训练中间目录
- 保留 `outputs/` 与 `summary-MMmodel/` 中的汇总结果、报告、图表和实验索引

所以 GitHub 上看到的 `outputs/` 和 `summary-MMmodel/` 是“可读结果层”，不是完整训练缓存。
