from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

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
    parser = argparse.ArgumentParser(description="Generate summary-MMmodel markdown report.")
    parser.add_argument("--config", default="configs/summary_mmmodel.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["output_root"])
    summary = read_json(root / "summary.json")
    experiments = summary["experiments"]
    report_cfg = summary["settings"].get("report", {})

    main_names = [name for name in report_cfg.get("main_experiment_names", []) if name in experiments]
    window_names = [name for name in report_cfg.get("window_experiment_names", []) if name in experiments]
    ablation_names = [name for name in report_cfg.get("ablation_experiment_names", []) if name in experiments]

    main_rows = [["Experiment", "Window Acc", "Window F1", "Best Session Method", "Session Acc", "Session F1"]]
    for name in main_names:
        exp = experiments[name]
        main_rows.append(
            [
                exp["display_name"],
                metric_ms(exp["window_level"]["mean_std"]["accuracy"]),
                metric_ms(exp["window_level"]["mean_std"]["macro_f1"]),
                exp["best_session_method"],
                metric_ms(exp["best_session_mean_std"]["accuracy"]),
                metric_ms(exp["best_session_mean_std"]["macro_f1"]),
            ]
        )

    window_rows = [["Window", "Best Session Method", "Window F1", "Session F1"]]
    for name in window_names:
        exp = experiments[name]
        window_rows.append(
            [
                f"{exp['window_sec']:.0f} s",
                exp["best_session_method"],
                metric_ms(exp["window_level"]["mean_std"]["macro_f1"]),
                metric_ms(exp["best_session_mean_std"]["macro_f1"]),
            ]
        )

    ablation_rows = [["Ablation", "Best Session Method", "Window F1", "Session F1"]]
    for name in ablation_names:
        exp = experiments[name]
        ablation_rows.append(
            [
                exp["display_name"],
                exp["best_session_method"],
                metric_ms(exp["window_level"]["mean_std"]["macro_f1"]),
                metric_ms(exp["best_session_mean_std"]["macro_f1"]),
            ]
        )

    fold_rows = [["Experiment", "Fold", "Window F1", "Session F1", "Best Epoch"]]
    for name in main_names + [n for n in window_names if n not in main_names] + [n for n in ablation_names if n not in main_names]:
        exp = experiments[name]
        best_method = exp["best_session_method"]
        session_fold_map = {item["fold"]: item["metrics"]["macro_f1"] for item in exp["session_aggregation"][best_method]["fold_results"]}
        for fold_record in exp["window_level"]["fold_results"]:
            fold_rows.append(
                [
                    exp["display_name"],
                    fold_record["fold"],
                    f"{fold_record['window_metrics']['macro_f1']:.4f}",
                    f"{session_fold_map[fold_record['fold']]:.4f}",
                    str(fold_record["best_epoch"]),
                ]
            )

    primary_name = report_cfg.get("primary_experiment")
    baseline_name = report_cfg.get("baseline_experiment")
    primary_exp = experiments[primary_name]
    baseline_exp = experiments[baseline_name]

    primary_session_f1 = primary_exp["best_session_mean_std"]["macro_f1"]["mean"]
    baseline_session_f1 = baseline_exp["best_session_mean_std"]["macro_f1"]["mean"]
    primary_window_f1 = primary_exp["window_level"]["mean_std"]["macro_f1"]["mean"]

    best_window_name = max(window_names, key=lambda name: experiments[name]["best_session_mean_std"]["macro_f1"]["mean"])
    best_window_exp = experiments[best_window_name]

    audio_name = next(name for name in main_names if "audio" in name)
    pressure_flow_name = next(name for name in main_names if "pressure_flow" in name)
    audio_exp = experiments[audio_name]
    pressure_flow_exp = experiments[pressure_flow_name]

    full_name = ablation_names[0]
    full_exp = experiments[full_name]
    full_f1 = full_exp["best_session_mean_std"]["macro_f1"]["mean"]
    ablation_effects = {}
    for name in ablation_names[1:]:
        exp = experiments[name]
        key = exp["display_name"].replace("HCAF-Net without ", "")
        ablation_effects[key] = exp["best_session_mean_std"]["macro_f1"]["mean"] - full_f1
    largest_drop_value = min(ablation_effects.values())
    largest_drop_modalities = sorted(
        [name for name, delta in ablation_effects.items() if abs(delta - largest_drop_value) <= 1e-8]
    )

    cm = np.asarray(
        primary_exp["session_aggregation"][primary_exp["best_session_method"]]["confusion_matrix_sum"],
        dtype=int,
    )
    zero_two = int(cm[0, 1] + cm[1, 0])
    two_four = int(cm[1, 2] + cm[2, 1])
    zero_four = int(cm[0, 2] + cm[2, 0])

    comparison_line = (
        f"HCAF-Net 的 session-level macro-F1 为 `{primary_session_f1:.4f}`，"
        f"{'高于' if primary_session_f1 > baseline_session_f1 else '未超过'}"
        f"现有 gated baseline 的 `{baseline_session_f1:.4f}`。"
    )
    multimodal_line = (
        f"HCAF-Net 相比 audio-only 的 `{audio_exp['best_session_mean_std']['macro_f1']['mean']:.4f}` "
        f"和 pressure+flow-only 的 `{pressure_flow_exp['best_session_mean_std']['macro_f1']['mean']:.4f}` "
        f"{'都更好' if primary_session_f1 > max(audio_exp['best_session_mean_std']['macro_f1']['mean'], pressure_flow_exp['best_session_mean_std']['macro_f1']['mean']) else '未同时超过单/双模态基线'}。"
    )
    confusion_line = (
        f"在主模型的 session-level 混淆矩阵中，`0 vs 2` 为 `{zero_two}`，`2 vs 4` 为 `{two_four}`，`0 vs 4` 为 `{zero_four}`；"
        f"{'相邻组别混淆更明显' if max(zero_two, two_four) > zero_four else '跨级组别混淆并不低于相邻组别'}。"
    )
    main_deltas = []
    for name in main_names:
        exp = experiments[name]
        delta = exp["best_session_mean_std"]["macro_f1"]["mean"] - exp["window_level"]["mean_std"]["macro_f1"]["mean"]
        main_deltas.append((exp["display_name"], delta))
    improved_models = [name for name, delta in main_deltas if delta > 0]
    degraded_models = [name for name, delta in main_deltas if delta <= 0]
    stability_line = (
        f"session 聚合对 {', '.join(improved_models)} 带来了更高的 macro-F1，"
        f"但对 {', '.join(degraded_models)} 没有提升；"
        f"HCAF-Net 自身从 `{primary_window_f1:.4f}` 变为 `{primary_session_f1:.4f}`。"
    )
    window_line = (
        f"HCAF-Net 的窗口长度比较中，最佳是 `{best_window_exp['window_sec']:.0f} s`，"
        f"session-level macro-F1 为 `{best_window_exp['best_session_mean_std']['macro_f1']['mean']:.4f}`。"
    )
    modality_line = (
        f"模态消融里，去掉 `{', '.join(largest_drop_modalities)}` 后 session-level macro-F1 下降最多，"
        f"变化量为 `{largest_drop_value:.4f}`，说明这些模态对 HCAF-Net 的贡献最大。"
    )

    report = f"""# 第四章主实验报告：0 / 2 / 4 三分类与 HCAF-Net

## 1. 数据划分策略

- 严格按 `recording/session` 级别划分，先划分后切窗，避免窗口泄漏
- 使用 grouped cross-validation，设置为 `{summary['settings']['grouped_cv']['n_repeats']}` repeat x `{summary['settings']['grouped_cv']['n_splits']}` folds
- train/val/test 都以 session 为 group，所有模型共用同一套 split manifest

## 2. 预处理方法

- audio: 重采样到 `16000 Hz`，提取 log-Mel 频谱，窗口内再做标准化
- pressure / flow: 使用 waveform 输入，按 recording 做 z-score
- 默认主比较窗口: `5 s`
- HCAF-Net 窗口长度比较: `5 s / 10 s`
- session-level 聚合: `majority_voting / mean_probability_pooling / logit_averaging`

## 3. 模型结构说明

- `audio_only`: log-Mel + 轻量 2D CNN
- `pressure_flow_only`: pressure / flow 各自 1D CNN + TCN，再做轻量 cross-attention 与 gated fusion
- `gated_baseline`: 复用现有最强 gated 融合基线，先做 pressure-flow 融合，再与 audio 融合
- `HCAF-Net`: audio 走 2D CNN token encoder；pressure 与 flow 走 1D CNN + TCN token encoder；第一层做 pressure-flow 双向 cross-attention；第二层做 audio-sensor cross-attention；之后接 1 层轻量 self-attention；分类前使用 reliability gate，并支持 modality dropout 与缺失模态 mask

## 4. 训练配置

- seed: `{config['seed']}`
- epochs: `{summary['settings']['training']['epochs']}`
- batch size: `{summary['settings']['training']['batch_size']}`
- optimizer: Adam
- learning rate: `{summary['settings']['training']['learning_rate']}`
- weight decay: `{summary['settings']['training']['weight_decay']}`
- weighted sampler: `{summary['settings']['training']['weighted_sampler']}`
- early stopping patience: `{summary['settings']['training']['early_stop_patience']}`

## 5. 主实验结果

{format_table(main_rows)}

![model_comparison](model_comparison.png)

## 6. HCAF-Net 窗口长度比较

{format_table(window_rows)}

![window_length](window_length_comparison.png)

## 7. HCAF-Net 模态消融

{format_table(ablation_rows)}

![ablation](ablation_results.png)

## 8. 每折结果

{format_table(fold_rows)}

## 9. 混淆矩阵

### Window-level

![hcaf_window](hcaf_net_5s_window_confusion_matrix_sum.png)

### Session-level

![hcaf_session](hcaf_net_5s_{primary_exp['best_session_method']}_session_confusion_matrix_sum.png)

![gated_session](gated_baseline_5s_{baseline_exp['best_session_method']}_session_confusion_matrix_sum.png)

![audio_session](audio_only_5s_{audio_exp['best_session_method']}_session_confusion_matrix_sum.png)

![pressure_flow_session](pressure_flow_5s_{pressure_flow_exp['best_session_method']}_session_confusion_matrix_sum.png)

## 10. 结果分析

- {comparison_line}
- {multimodal_line}
- {confusion_line}
- {stability_line}
- {window_line}
- {modality_line}

## 11. 局限性说明

- 当前仅覆盖 `0 / 2 / 4` 三分类，不直接外推到 `1 / 3 ml`
- 样本总量仍偏小，fold 间方差仍需关注
- 目前只比较了固定长度窗口，尚未纳入可靠的单周期标注实验
- 如果 HCAF-Net 未超过 gated baseline，也应保留这一结果，因为这说明当前数据规模下更强的层次融合未必自动转化为更高泛化性能
"""

    (root / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
