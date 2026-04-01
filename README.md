# MMDL

多模态液体体积分类实验仓库。当前默认主线已经统一切到：

- `audio encoder = ResNet18 (ImageNet init)`
- `PQ + audio cross-attention`
- 固定非对齐 `5 s` 窗
- 以 `window-level macro-F1` 作为主模型选择指标

之所以把主指标切到 `window-level`，是因为当前记录级 `session` 很长、且长度不等；对当前任务来说，窗口判别更直接，也更适合比较不同融合结构本身的有效性。

## Current Final Model

- final model: `hcaf_audio_r18img_pq_xattn_5s`
- task: `0 / 2 / 4` three-class classification
- modalities: `audio + pressure + flow`
- audio encoder: `ResNet18` with `ImageNet` initialization
- PQ encoder: `TCN`
- primary metric: `window macro-F1`
- split: `session`-grouped `1 repeat x 3 folds`
- window length: `5 s`
- final selection date: `2026-04-01`

当前最终主结果目录：

- [`summary-MMmodel/hcaf_audioresnet_xattn_vs_concat`](summary-MMmodel/hcaf_audioresnet_xattn_vs_concat)

## Main Result

当前主对照使用同一份 split manifest、同一训练预算、同一音频前端 `PCEN96 + HP80`：

| model | source | window macro-F1 | session macro-F1 |
| --- | --- | ---: | ---: |
| `pressure_flow_5s` | [`summary-MMmodel/hcaf_audioresnet_xattn_vs_concat`](summary-MMmodel/hcaf_audioresnet_xattn_vs_concat) | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` |
| `hcaf_audio_r18img_audio_only_5s` | [`summary-MMmodel/hcaf_audioresnet_xattn_vs_concat`](summary-MMmodel/hcaf_audioresnet_xattn_vs_concat) | `0.8709 ± 0.0722` | `0.9407 ± 0.0838` |
| `hcaf_audio_r18img_pq_directconcat_5s` | [`summary-MMmodel/hcaf_audioresnet_xattn_vs_concat`](summary-MMmodel/hcaf_audioresnet_xattn_vs_concat) | `0.7800 ± 0.1610` | `0.7852 ± 0.1923` |
| `hcaf_audio_r18img_pq_xattn_5s` | [`summary-MMmodel/hcaf_audioresnet_xattn_vs_concat`](summary-MMmodel/hcaf_audioresnet_xattn_vs_concat) | `0.9145 ± 0.0745` | `0.9407 ± 0.0838` |

当前核心结论：

- `PQ + audio cross-attention` 在 `window-level` 上强于 `PQ-only`
- `PQ + audio cross-attention` 在 `window-level` 上强于 `audio-only`
- `PQ + audio cross-attention` 在 `window-level` 上强于 `direct concat PQ+audio`
- 在当前 `ResNet18` 约束下，`session-level` 上 `cross-attention` 与 `audio-only` 打平，因此默认文档口径不再用 session 指标决定最终模型

对应窗口级差值：

- vs `audio-only`: `+0.0436`
- vs `PQ-only`: `+0.1646`
- vs `direct concat`: `+0.1345`

## Why This Model

当前保留 `hcaf_audio_r18img_pq_xattn_5s` 作为默认主模型，不是因为它在所有口径下都绝对最好，而是因为它同时满足：

1. 音频分支满足 `ResNet18(ImageNet init)` 约束
2. 在固定非对齐窗设置下，`window-level macro-F1` 最强
3. 明确强于 `PQ-only`、`audio-only` 和 `direct concat`
4. 结构清晰，便于继续做传感器增强和进一步搜索

## What Was Tried

在当前 `ResNet18` 主线上，已经系统尝试过：

- 更复杂 PQ encoder
- 更高 `modality_dropout`
- `direct concat PQ+audio`
- 固定更长窗：`8 s / 10 s / 12 s`
- 呼吸周期对齐切窗
- 单周期 / 双周期归一化表示

目前这些方向都没有稳定超过当前固定非对齐 `5 s` 的 `cross-attention` 主线。

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
└── report.md                 # 逐轮实验记录
```

## What To Read First

1. [`summary.md`](summary.md)  
   当前主模型 `hcaf_audio_r18img_pq_xattn_5s` 的结构、数据流和性能分析。

2. [`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md)  
   当前主模型索引和最短证据链摘要。

3. [`summary-MMmodel/hcaf_audioresnet_xattn_vs_concat`](summary-MMmodel/hcaf_audioresnet_xattn_vs_concat)  
   当前最重要的正式对照目录，直接证明 `cross-attention` 强于 `PQ-only` / `audio-only` / `direct concat`。

4. [`report.md`](report.md)  
   逐轮实验日志，包含 `ResNet18` 路线、固定窗搜索和周期切窗探索的完整过程。

## Key Configs

- [`configs/hcaf_audioresnet_xattn_vs_concat.yaml`](configs/hcaf_audioresnet_xattn_vs_concat.yaml)  
  当前最重要的正式主对照配置。

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
python summary_mmmodel_experiments.py --config configs/hcaf_audioresnet_xattn_vs_concat.yaml
```

运行呼吸周期统计：

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
python analyze_breath_cycles.py
```

## Notes

- 旧的 `basic audio encoder` 最终模型文档口径已归档，不再作为当前默认说明
- 当前默认说明统一围绕 `ResNet18 + PQ+audio cross-attention + fixed 5 s window`
- 若后续继续优化，优先方向是增强 `sensor` 分支，而不是继续改音频编码器
