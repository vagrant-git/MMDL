# MMDL

多模态液体体积分类实验仓库。当前默认主线已经统一切到论文展示名 `HCAF-PCEN-XAttn`：

- `audio encoder = ResNet18 (ImageNet init)`
- `PQ + audio cross-attention`
- 固定非对齐 `5 s` 窗
- 以 `window-level macro-F1` 作为主模型选择指标

之所以把主指标切到 `window-level`，是因为当前记录级 `session` 很长、且长度不等；对当前任务来说，窗口判别更直接，也更适合比较不同融合结构本身的有效性。

对应实验配置 ID 为 `hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s`。

## Current Final Model

- display name: `HCAF-PCEN-XAttn`
- experiment id: `hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s`
- task: `0 / 2 / 4` three-class classification
- modalities: `audio + pressure + flow`
- audio frontend: `PCEN96 + HP80`
- audio encoder: `ResNet18` with `ImageNet` initialization
- PQ encoder: `TCN`
- interaction: `PQ cross-attention + audio-sensor cross-attention`
- decision: `confidence-aware gate + expert residual`
- primary metric: `window macro-F1`
- split: `session`-grouped `1 repeat x 3 folds`
- window length: `5 s`
- final selection date: `2026-04-01`

当前最终主结果目录：

- [`outputs/hcaf_confgate_compression_search`](outputs/hcaf_confgate_compression_search)

## Main Result

当前主结果以补充压缩实验为准：

| model | source | window macro-F1 | session macro-F1 |
| --- | --- | ---: | ---: |
| `HCAF compressed base SA0 PCEN96 HP80` | [`outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md`](outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md) | `0.8968 ± 0.0495` | `0.8815 ± 0.0838` |
| `HCAF-PCEN-XAttn` | [`outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md`](outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md) | `0.9155 ± 0.0133` | `0.9407 ± 0.0838` |
| `HCAF compressed SA0 summary token attention` | [`outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md`](outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md) | `0.8298 ± 0.0805` | `0.8815 ± 0.0838` |
| `HCAF compressed SA0 PCEN64 HP80` | [`outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md`](outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md) | `0.8773 ± 0.0458` | `0.8815 ± 0.0838` |

当前核心结论：

- 去掉联合 self-attention 之后，保留核心双阶段 cross-attention 仍然可以稳定达到当前最佳窗口级结果
- 去掉表示层的 `summary` 残差后，window-level 反而比同轮 `SA0 base` 更高、更稳
- `PCEN96 + HP80` 仍然比 `PCEN64` 压缩前端更能守住当前主指标

对应窗口级差值：

- vs `SA0 base`: `+0.0187`
- vs `summary token`: `+0.0857`
- vs `PCEN64 HP80`: `+0.0382`

## Why This Model

当前保留 `HCAF-PCEN-XAttn` 作为默认主模型，不是因为名字最长，而是因为它保留了最有价值的核心结构：

1. `PCEN96 + HP80` 音频前端
2. `Pressure-Flow` 内部交互 + `audio-sensor` 双阶段 cross-attention
3. `confidence-aware gate + expert residual`
4. 在当前已完成结果里，`window-level macro-F1` 最强且波动更小

## What Was Tried

在当前 `ResNet18` 主线上，已经系统尝试过：

- 更复杂 PQ encoder
- 更高 `modality_dropout`
- `direct concat PQ+audio`
- 固定更长窗：`8 s / 10 s / 12 s`
- 呼吸周期对齐切窗
- 单周期 / 双周期归一化表示

目前这些方向都没有稳定超过当前固定非对齐 `5 s` 的 `HCAF-PCEN-XAttn` 主线。

## Repository Guide

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
├── summary.md                # 当前主模型技术说明
├── EXPERIMENT_SUMMARY.md     # 当前主模型索引
├── EXPERIMENT_RESULTS_ALL.md # 所有实验结果总表
└── report.md                 # 逐轮实验记录
```

## What To Read First

1. [`summary.md`](summary.md)  
   当前主模型 `HCAF-PCEN-XAttn` 的结构、数据流和性能分析。

2. [`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md)  
   当前主模型索引和最短证据链摘要。

3. [`outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md`](outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md)
   当前默认模型的直接证据链，说明保留核心结构后压缩版仍是当前最佳。

4. [`EXPERIMENT_RESULTS_ALL.md`](EXPERIMENT_RESULTS_ALL.md)
   单个 Markdown 汇总所有已整理实验结果。

5. [`report.md`](report.md)
   逐轮实验日志，包含 `ResNet18` 路线、固定窗搜索和周期切窗探索的完整过程。

## Key Configs

- [`configs/hcaf_confgate_compression_search.yaml`](configs/hcaf_confgate_compression_search.yaml)
  当前默认模型所在的压缩消融配置。

- [`configs/final_model_unified_evidence.yaml`](configs/final_model_unified_evidence.yaml)
  当前 HCAF 完整结构版的统一证据配置。

- [`configs/hcaf_audioresnet_pq_seqmodels.yaml`](configs/hcaf_audioresnet_pq_seqmodels.yaml)  
  `ResNet18 + PQ TCN / GRU / CNN-GRU` 的结构搜索配置。

- [`configs/hcaf_audioresnet_cycle_window_search.yaml`](configs/hcaf_audioresnet_cycle_window_search.yaml)  
  固定 `4 s / 5 s` 与周期切窗探索。

- [`configs/hcaf_audioresnet_cycle_aligned_search.yaml`](configs/hcaf_audioresnet_cycle_aligned_search.yaml)  
  周期对齐窗口探索。

## Environment

按仓库约定，训练、评估和报告脚本都在 `dl` 环境下运行：

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
```

如果日志里出现 `torch 2.x+cpu` 或 `cuda_available=False`，通常说明当前没有正确使用 GPU 环境。

## Quick Start

运行当前主对照实验：

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
python summary_mmmodel_experiments.py --config configs/hcaf_confgate_compression_search.yaml
```

运行呼吸周期统计：

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
python analyze_breath_cycles.py
```

## Notes

- 旧的 `basic audio encoder` 最终模型文档口径已归档，不再作为当前默认说明
- 当前默认说明统一围绕 `HCAF-PCEN-XAttn`
- 若后续继续优化，优先方向是增强 `sensor` 分支，而不是继续改音频编码器

## Markdown Guide

仓库里的 Markdown 文件比较多，但用途大致可以分成 3 类：论文写作用、当前结果查询用、历史归档用。

### 论文写作用

- [`thesis_model_architecture_draft.md`](thesis_model_architecture_draft.md)
  - 作用：论文正文草稿
  - 适合：直接改写进“模型结构”章节
  - 特点：语言最接近正式论文叙述

- [`summary.md`](summary.md)
  - 作用：当前主模型技术说明
  - 适合：写“方法”“实验结果”“局限性”时快速抽取内容
  - 特点：结构化最完整，兼顾技术细节和结果

- [`report.md`](report.md)
  - 作用：完整实验过程记录
  - 适合：回查“为什么这么做、试过什么、为什么放弃”
  - 特点：最全，但最长

- [`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md)
  - 作用：顶层简明摘要
  - 适合：答辩首页、论文补充说明、快速总览
  - 特点：最短，先看它可以迅速知道现在主线是什么

- [`EXPERIMENT_RESULTS_ALL.md`](EXPERIMENT_RESULTS_ALL.md)
  - 作用：所有实验结果总表
  - 适合：一次性总览所有已整理结果
  - 特点：按目录汇总，便于后续继续追加

### 当前结果查询用

- [`outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md`](outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md)
  - 作用：当前默认模型压缩对照摘要
  - 适合：确认 `HCAF-PCEN-XAttn` 为什么是当前默认口径

- [`summary-MMmodel/hcaf_audioresnet_xattn_vs_concat/EXPERIMENT_SUMMARY.md`](summary-MMmodel/hcaf_audioresnet_xattn_vs_concat/EXPERIMENT_SUMMARY.md)
  - 作用：当前最关键的正式对照摘要
  - 适合：确认 `cross-attention` 是否强于 `PQ-only / audio-only / direct concat`

- [`summary-MMmodel/hcaf_audioresnet_xattn_vs_concat/overall_results.csv`](summary-MMmodel/hcaf_audioresnet_xattn_vs_concat/overall_results.csv)
  - 作用：当前主结果均值表
  - 适合：直接查最终指标

- [`summary-MMmodel/hcaf_audioresnet_xattn_vs_concat/fold_results.csv`](summary-MMmodel/hcaf_audioresnet_xattn_vs_concat/fold_results.csv)
  - 作用：当前主结果逐折明细
  - 适合：分析某一折为什么高或低

- [`summary-MMmodel/hcaf_audioresnet_xattn_vs_concat/summary.json`](summary-MMmodel/hcaf_audioresnet_xattn_vs_concat/summary.json)
  - 作用：当前主结果的结构化完整输出
  - 适合：程序化读取或查全部细节

- [`summary-MMmodel/EXPERIMENT_SUMMARY.md`](summary-MMmodel/EXPERIMENT_SUMMARY.md)
  - 作用：`summary-MMmodel` 目录索引
  - 适合：定位“现在最该看哪个实验目录”

### 历史归档 / 背景参考

- `summary-MMmodel/*/EXPERIMENT_SUMMARY.md`
  - 作用：各轮专项实验的摘要
  - 适合：回查某条旧路线、失败路线、补充验证

- `outputs/*/report.md`
  - 作用：更早期实验的归档记录
  - 适合：只在需要追溯旧实验时查看

- [`README.md`](README.md)
  - 作用：仓库导航页
  - 适合：第一次进入仓库时建立全局认识

- [`AGENT.md`](AGENT.md)
  - 作用：AI 自主迭代工作规则
  - 适合：规范研发流程，不属于论文内容

### 最短阅读路径

如果你现在只想快速定位重点，建议按这个顺序读：

1. [`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md)
2. [`summary.md`](summary.md)
3. [`thesis_model_architecture_draft.md`](thesis_model_architecture_draft.md)
4. [`outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md`](outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md)
5. [`EXPERIMENT_RESULTS_ALL.md`](EXPERIMENT_RESULTS_ALL.md)
