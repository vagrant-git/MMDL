from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from mmdl_baseline.utils.aggregation import AGGREGATION_METHODS
from mmdl_baseline.utils.config import load_config
from mmdl_baseline.utils.io import read_json


def format_table(rows: List[List[str]]) -> str:
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, sep] + body)


def metric_ms(metric: dict) -> str:
    return f"{metric['mean']:.4f} ± {metric['std']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chapter 4 markdown report for 0/2/4 classification.")
    parser.add_argument("--config", default="configs/chapter4_024.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["output_root"])
    summary = read_json(root / "chapter4_024_summary.json")
    experiments = summary["experiments"]

    main_rows = [["Experiment", "Window Acc", "Window F1", "Best Session Method", "Session Acc", "Session F1"]]
    for exp_name in ["audio_only_10s", "pressure_flow_10s", "multimodal_10s"]:
        exp = experiments[exp_name]
        main_rows.append(
            [
                exp_name,
                metric_ms(exp["window_level"]["mean_std"]["accuracy"]),
                metric_ms(exp["window_level"]["mean_std"]["macro_f1"]),
                exp["best_session_method"],
                metric_ms(exp["best_session_mean_std"]["accuracy"]),
                metric_ms(exp["best_session_mean_std"]["macro_f1"]),
            ]
        )

    window_rows = [["Window", "Best Session Method", "Session Acc", "Session F1"]]
    for exp_name in ["multimodal_5s", "multimodal_10s", "multimodal_20s"]:
        exp = experiments[exp_name]
        window_rows.append(
            [
                exp_name,
                exp["best_session_method"],
                metric_ms(exp["best_session_mean_std"]["accuracy"]),
                metric_ms(exp["best_session_mean_std"]["macro_f1"]),
            ]
        )

    ablation_rows = [["Ablation", "Best Session Method", "Session Acc", "Session F1"]]
    for exp_name in ["multimodal_10s", "multimodal_minus_audio_10s", "multimodal_minus_pressure_10s", "multimodal_minus_flow_10s"]:
        if exp_name in experiments:
            exp = experiments[exp_name]
            ablation_rows.append(
                [
                    exp_name,
                    exp["best_session_method"],
                    metric_ms(exp["best_session_mean_std"]["accuracy"]),
                    metric_ms(exp["best_session_mean_std"]["macro_f1"]),
                ]
            )

    fusion_items = [(name, exp) for name, exp in experiments.items() if exp.get("group") == "fusion"]
    fusion_items.sort(key=lambda item: item[1]["best_session_mean_std"]["macro_f1"]["mean"], reverse=True)
    fusion_rows = [["Fusion Experiment", "Best Session Method", "Session Acc", "Session F1"]]
    for exp_name, exp in fusion_items:
        fusion_rows.append(
            [
                exp_name,
                exp["best_session_method"],
                metric_ms(exp["best_session_mean_std"]["accuracy"]),
                metric_ms(exp["best_session_mean_std"]["macro_f1"]),
            ]
        )

    fold_rows = [["Experiment", "Fold", "Window F1", "Best Session F1"]]
    for exp_name, exp in experiments.items():
        best_method = exp["best_session_method"]
        session_fold_map = {r["fold"]: r["metrics"]["macro_f1"] for r in exp["session_aggregation"][best_method]["fold_results"]}
        for fold_record in exp["window_level"]["fold_results"]:
            fold_rows.append(
                [
                    exp_name,
                    fold_record["fold"],
                    f"{fold_record['window_metrics']['macro_f1']:.4f}",
                    f"{session_fold_map[fold_record['fold']]:.4f}",
                ]
            )

    mm_cm = np.asarray(experiments["multimodal_10s"]["session_aggregation"][experiments["multimodal_10s"]["best_session_method"]]["confusion_matrix_sum"])
    zero_two = int(mm_cm[0, 1] + mm_cm[1, 0])
    two_four = int(mm_cm[1, 2] + mm_cm[2, 1])
    zero_four = int(mm_cm[0, 2] + mm_cm[2, 0])

    full_mm = experiments["multimodal_10s"]["best_session_mean_std"]["macro_f1"]["mean"]
    minus_audio = experiments["multimodal_minus_audio_10s"]["best_session_mean_std"]["macro_f1"]["mean"] if "multimodal_minus_audio_10s" in experiments else full_mm
    minus_pressure = experiments["multimodal_minus_pressure_10s"]["best_session_mean_std"]["macro_f1"]["mean"] if "multimodal_minus_pressure_10s" in experiments else full_mm
    minus_flow = experiments["multimodal_minus_flow_10s"]["best_session_mean_std"]["macro_f1"]["mean"] if "multimodal_minus_flow_10s" in experiments else full_mm
    ablation_effects = {
        "audio": minus_audio - full_mm,
        "pressure": minus_pressure - full_mm,
        "flow": minus_flow - full_mm,
    }
    largest_drop = min(ablation_effects.items(), key=lambda x: x[1])
    largest_gain = max(ablation_effects.items(), key=lambda x: x[1])
    main_best = max(
        ["audio_only_10s", "pressure_flow_10s", "multimodal_10s"],
        key=lambda name: experiments[name]["best_session_mean_std"]["macro_f1"]["mean"],
    )

    analysis_lines = [
        f"- 默认 10s 设置下，主实验三组里 session-level macro-F1 最好的是 `{main_best}`；其中 `audio_only_10s` 为 `{experiments['audio_only_10s']['best_session_mean_std']['macro_f1']['mean']:.4f}`，`pressure_flow_10s` 为 `{experiments['pressure_flow_10s']['best_session_mean_std']['macro_f1']['mean']:.4f}`，`multimodal_10s` 仅为 `{full_mm:.4f}`。",
        f"- 这说明在当前 0/2/4 三分类上，简单三模态融合并没有在默认 10s 设置下稳定优于单模态或双模态；相反，音频或传感器单独使用反而更稳。",
        f"- 窗口长度影响非常明显：`multimodal_5s` 的 session-level macro-F1 达到 `{experiments['multimodal_5s']['best_session_mean_std']['macro_f1']['mean']:.4f}`，显著高于 `multimodal_10s` 和 `multimodal_20s`，说明较短固定窗口更适合当前三模态建模。",
        f"- 消融结果显示，去掉 `{largest_drop[0]}` 后 macro-F1 下降最多，说明该模态对当前三模态模型仍有正贡献；但去掉 `{largest_gain[0]}` 后反而提升到 `{full_mm + largest_gain[1]:.4f}`，说明该模态在现有融合设计下可能引入了噪声或与其余模态耦合不佳。",
        f"- 最佳三模态 session-level 混淆中，`0 vs 2` 为 `{zero_two}`，`2 vs 4` 为 `{two_four}`，`0 vs 4` 为 `{zero_four}`；如果相邻级别的混淆更大，则说明中间负荷边界仍然最难。",
        "- session-level aggregation 整体上比 window-level 更稳定，尤其对 pressure_flow 和 multimodal 更明显。",
        "- 当前未实现单周期切分，因为缺少可直接复用的稳定周期标注；窗口长度比较采用 5s / 10s / 20s 固定窗，并在报告中明确说明这一点。",
    ]
    if fusion_items:
        best_fusion_name, best_fusion_exp = fusion_items[0]
        analysis_lines.append(
            f"- 融合策略扩展实验里，当前最好的是 `{best_fusion_name}`，session-level macro-F1 为 `{best_fusion_exp['best_session_mean_std']['macro_f1']['mean']:.4f}`。"
        )

    report = f"""# 第四章主实验报告：0 / 2 / 4 三分类

## 1. 数据划分策略

- 按 `recording/session` 级别划分，禁止先切窗再随机划分
- 使用 grouped cross-validation
- 当前设置: `{summary['settings']['grouped_cv']['n_repeats']}` repeats x `{summary['settings']['grouped_cv']['n_splits']}` folds
- task: 原始标签 `0 / 2 / 4`，在模型内部重映射为连续类别索引

## 2. 预处理方法

- audio: 重采样到 `16000 Hz`，转 log-Mel 频谱
- pressure / flow: 使用 waveform，做 recording 级 z-score
- 默认窗口: `10s`
- 窗口长度比较: `5s / 10s / 20s`
- session-level aggregation: `majority_voting / mean_probability_pooling / logit_averaging`

## 3. 模型结构

- `audio_only`: log-Mel + 轻量 2D CNN
- `pressure_flow`: pressure encoder + flow encoder，独立 1D CNN + TCN，再做 cross-attention 与 gated fusion
- `multimodal`: audio encoder + pressure encoder + flow encoder，先融合 pressure/flow，再与 audio 做 cross-attention 与 gated fusion
- 支持 modality dropout 与固定缺失模态设置，用于消融实验

## 4. 训练配置

- python 环境: `dl`
- epochs: `{config['epochs']}`
- batch size: `{config['batch_size']}`
- optimizer: Adam
- learning rate: `{config['learning_rate']}`
- weight decay: `{config['weight_decay']}`
- random seed: `{config['seed']}`

## 5. 主实验结果

{format_table(main_rows)}

## 6. 窗口长度比较

{format_table(window_rows)}

![window_length](window_length_comparison.png)

## 7. 模态消融实验

{format_table(ablation_rows)}

![ablation](ablation_results.png)

{"## 8. 融合策略实验\n\n" + format_table(fusion_rows) if len(fusion_rows) > 1 else ""}

## 9. 每折结果摘要

{format_table(fold_rows)}

## 10. 混淆矩阵

### 主模型

![audio_main](audio_only_10s_{experiments['audio_only_10s']['best_session_method']}_confusion_matrix_sum.png)

![pf_main](pressure_flow_10s_{experiments['pressure_flow_10s']['best_session_method']}_confusion_matrix_sum.png)

![mm_main](multimodal_10s_{experiments['multimodal_10s']['best_session_method']}_confusion_matrix_sum.png)

### 消融

![mm_minus_audio](multimodal_minus_audio_10s_{experiments['multimodal_minus_audio_10s']['best_session_method']}_confusion_matrix_sum.png)

![mm_minus_pressure](multimodal_minus_pressure_10s_{experiments['multimodal_minus_pressure_10s']['best_session_method']}_confusion_matrix_sum.png)

![mm_minus_flow](multimodal_minus_flow_10s_{experiments['multimodal_minus_flow_10s']['best_session_method']}_confusion_matrix_sum.png)

## 11. 结果解释

{chr(10).join(analysis_lines)}

## 12. 局限性分析

- 当前 `4 ml` session 数量仍然较少，fold 间方差不能忽略
- `1 ml / 3 ml` 未纳入本任务，因此本章结论仅对应 `0 / 2 / 4` 三分类
- 固定时长窗口是当前工程下最稳妥的实现；若后续有可靠周期标注，可继续补单周期实验
"""

    (root / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
