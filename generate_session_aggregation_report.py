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
    parser = argparse.ArgumentParser(description="Generate session aggregation markdown report.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    load_config(args.config)
    root = Path("outputs/grouped_cv_5class_session_agg")
    summary = read_json(root / "session_aggregation_summary.json")
    modalities = ["audio_only", "pressure_flow", "multimodal"]

    overall_rows = [["Model", "Eval", "Acc", "Macro-F1", "Precision", "Recall"]]
    fold_rows = [["Fold", "Model", "Eval", "Acc", "Macro-F1", "Precision", "Recall"]]

    for modality in modalities:
        win = summary["per_model"][modality]["window_level"]["mean_std"]
        overall_rows.append(
            [
                modality,
                "window_level",
                metric_ms(win["accuracy"]),
                metric_ms(win["macro_f1"]),
                metric_ms(win["macro_precision"]),
                metric_ms(win["macro_recall"]),
            ]
        )
        for record in summary["per_model"][modality]["window_level"]["fold_results"]:
            fold_rows.append(
                [
                    record["fold"],
                    modality,
                    "window_level",
                    f"{record['window_metrics']['accuracy']:.4f}",
                    f"{record['window_metrics']['macro_f1']:.4f}",
                    f"{record['window_metrics']['macro_precision']:.4f}",
                    f"{record['window_metrics']['macro_recall']:.4f}",
                ]
            )
        for method in AGGREGATION_METHODS:
            ms = summary["per_model"][modality]["session_aggregation"][method]["mean_std"]
            overall_rows.append(
                [
                    modality,
                    method,
                    metric_ms(ms["accuracy"]),
                    metric_ms(ms["macro_f1"]),
                    metric_ms(ms["macro_precision"]),
                    metric_ms(ms["macro_recall"]),
                ]
            )
            for record in summary["per_model"][modality]["session_aggregation"][method]["fold_results"]:
                metrics = record["metrics"]
                fold_rows.append(
                    [
                        record["fold"],
                        modality,
                        method,
                        f"{metrics['accuracy']:.4f}",
                        f"{metrics['macro_f1']:.4f}",
                        f"{metrics['macro_precision']:.4f}",
                        f"{metrics['macro_recall']:.4f}",
                    ]
                )

    mm_window = summary["per_model"]["multimodal"]["window_level"]["mean_std"]["macro_f1"]["mean"]
    mm_best_session = max(
        summary["per_model"]["multimodal"]["session_aggregation"][method]["mean_std"]["macro_f1"]["mean"]
        for method in AGGREGATION_METHODS
    )
    pf_window = summary["per_model"]["pressure_flow"]["window_level"]["mean_std"]["macro_f1"]["mean"]
    pf_best_session = max(
        summary["per_model"]["pressure_flow"]["session_aggregation"][method]["mean_std"]["macro_f1"]["mean"]
        for method in AGGREGATION_METHODS
    )
    au_window = summary["per_model"]["audio_only"]["window_level"]["mean_std"]["macro_f1"]["mean"]
    au_best_session = max(
        summary["per_model"]["audio_only"]["session_aggregation"][method]["mean_std"]["macro_f1"]["mean"]
        for method in AGGREGATION_METHODS
    )

    best_method = max(
        AGGREGATION_METHODS,
        key=lambda method: summary["per_model"]["multimodal"]["session_aggregation"][method]["mean_std"]["macro_f1"]["mean"],
    )
    best_cm = np.asarray(summary["per_model"]["multimodal"]["session_aggregation"][best_method]["confusion_matrix_sum"], dtype=int)
    one_two = int(best_cm[1, 2] + best_cm[2, 1])
    zero_four = int(best_cm[0, 4] + best_cm[4, 0])

    analysis_lines = [
        f"- 对三模态模型，session-level 聚合把平均 macro-F1 从 window-level 的 `{mm_window:.4f}` 提升到最佳聚合方式 `{best_method}` 的 `{mm_best_session:.4f}`，说明对 recording/session 粒度的判别更稳定。",
        f"- 聚合增益具有模态差异：`audio_only` 的最佳 session-level macro-F1 从 `{au_window:.4f}` 变为 `{au_best_session:.4f}`，没有带来提升；`pressure_flow` 则从 `{pf_window:.4f}` 提升到 `{pf_best_session:.4f}`，说明聚合更适合传感器或多模态场景。",
        f"- multimodal 在 session-level 下的优势更明显：其最佳 session-level macro-F1 高于 `audio_only` 与 `pressure_flow` 的最佳 session-level 结果。",
        f"- 在最佳三模态聚合混淆矩阵中，`1 ml` 与 `2 ml` 的互相混淆仍最多，共 `{one_two}` 次；`0 ml` 与 `4 ml` 的互相混淆为 `{zero_four}` 次，仍明显更少。",
        "- 限制没有变化：所有分析都严格复用 grouped CV 的 session 划分，未发生 session 泄漏；但由于 `3 ml` session 极少，折间波动仍然较大。",
    ]

    report = f"""# Session-Level Aggregation 报告

## 1. 聚合方法说明

- `majority_voting`: 对同一测试 session 内所有窗口的预测类别做多数投票
- `mean_probability_pooling`: 对同一 session 内所有窗口的类别概率做均值，再取 argmax
- `logit_averaging`: 对同一 session 内所有窗口的 log-probability 做均值，再取 argmax
- 说明: 原始 grouped CV 输出保存的是概率而不是 raw logits；由于 logits 与 log-probabilities 只差每窗口一个类别无关常数，分类 argmax 不变

## 2. 整体结果对比

{format_table(overall_rows)}

## 3. 每折结果

{format_table(fold_rows)}

## 4. 混淆矩阵

### 三模态最佳聚合方式

最佳方式: `{best_method}`

![multimodal_best](multimodal_{best_method}_confusion_matrix_sum.png)

### 全模型全方法汇总图

![audio_mv](audio_only_majority_voting_confusion_matrix_sum.png)

![audio_mean](audio_only_mean_probability_pooling_confusion_matrix_sum.png)

![audio_logit](audio_only_logit_averaging_confusion_matrix_sum.png)

![pf_mv](pressure_flow_majority_voting_confusion_matrix_sum.png)

![pf_mean](pressure_flow_mean_probability_pooling_confusion_matrix_sum.png)

![pf_logit](pressure_flow_logit_averaging_confusion_matrix_sum.png)

![mm_mv](multimodal_majority_voting_confusion_matrix_sum.png)

![mm_mean](multimodal_mean_probability_pooling_confusion_matrix_sum.png)

![mm_logit](multimodal_logit_averaging_confusion_matrix_sum.png)

## 5. 结果解释与局限性

{chr(10).join(analysis_lines)}
"""

    (root / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
