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
    parser = argparse.ArgumentParser(description="Generate grouped CV markdown report.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    cv_root = Path("outputs/grouped_cv_5class")
    summary = read_json(cv_root / "grouped_cv_summary.json")
    split_manifest = read_json(cv_root / "split_manifest.json")

    summary_rows = [["Model", "Window Acc", "Window F1", "Window P", "Window R", "Session Acc", "Session F1"]]
    per_fold_rows = [["Fold", "Model", "Window Acc", "Window F1", "Window P", "Window R", "Session Acc", "Session F1"]]
    modalities = ["audio_only", "pressure_flow", "multimodal"]

    for modality in modalities:
        model_summary = summary["per_model"][modality]
        wm = model_summary["window_mean_std"]
        sm = model_summary["session_mean_std"]
        summary_rows.append(
            [
                modality,
                metric_ms(wm["accuracy"]),
                metric_ms(wm["macro_f1"]),
                metric_ms(wm["macro_precision"]),
                metric_ms(wm["macro_recall"]),
                metric_ms(sm["accuracy"]),
                metric_ms(sm["macro_f1"]),
            ]
        )
        for record in model_summary["fold_results"]:
            per_fold_rows.append(
                [
                    record["fold"],
                    modality,
                    f"{record['window_metrics']['accuracy']:.4f}",
                    f"{record['window_metrics']['macro_f1']:.4f}",
                    f"{record['window_metrics']['macro_precision']:.4f}",
                    f"{record['window_metrics']['macro_recall']:.4f}",
                    f"{record['session_metrics']['accuracy']:.4f}",
                    f"{record['session_metrics']['macro_f1']:.4f}",
                ]
            )

    mm_f1 = summary["per_model"]["multimodal"]["window_mean_std"]["macro_f1"]["mean"]
    pf_f1 = summary["per_model"]["pressure_flow"]["window_mean_std"]["macro_f1"]["mean"]
    au_f1 = summary["per_model"]["audio_only"]["window_mean_std"]["macro_f1"]["mean"]
    mm_cm = np.asarray(summary["per_model"]["multimodal"]["window_confusion_matrix_sum"], dtype=int)
    adj_confusion = {
        "1_vs_2": int(mm_cm[1, 2] + mm_cm[2, 1]),
        "2_vs_3": int(mm_cm[2, 3] + mm_cm[3, 2]),
        "3_vs_4": int(mm_cm[3, 4] + mm_cm[4, 3]),
        "0_vs_4": int(mm_cm[0, 4] + mm_cm[4, 0]),
    }

    split_lines = [
        f"- {item['name']}: train={item['train_ids']}, val={item['val_ids']}, test={item['test_ids']}"
        for item in split_manifest
    ]

    analysis_lines = [
        f"- 三模态在 grouped CV 下的平均窗口级 macro-F1 为 `{mm_f1:.4f}`，高于 `pressure_flow` 的 `{pf_f1:.4f}` 和 `audio_only` 的 `{au_f1:.4f}`，说明优势不是单次随机划分偶然得到的。",
        f"- `pressure+flow` 在窗口级平均 macro-F1 为 `{pf_f1:.4f}`，略低于 `audio_only` 的 `{au_f1:.4f}`；但在 session 聚合后，`pressure+flow` 的平均 macro-F1 更高，说明传感器模态对 recording 级判别仍有价值，但优势不如单次 baseline 明显。",
        f"- 最佳模型的相邻类别混淆仍主要集中在 `1 ml` 与 `2 ml` 附近：1<->2 共 `{adj_confusion['1_vs_2']}`，而 2<->3 为 `{adj_confusion['2_vs_3']}`，3<->4 为 `{adj_confusion['3_vs_4']}`。",
        f"- `0 ml` 与 `4 ml` 在三模态聚合混淆矩阵中的互相混淆总数为 `{adj_confusion['0_vs_4']}`，明显少于 `1<->2` 的 `{adj_confusion['1_vs_2']}`，说明负荷两端仍比中间相邻组更容易区分，但并非完全分离。",
        "- 限制仍然明显：`3 ml` 仅有 2 个 session，因此这里只能做 2-fold grouped CV；结论是初步稳定性验证，不应解读为充分统计结论。",
    ]

    report = f"""# Grouped CV 五分类稳定性报告

## 1. 实验设置

- python 环境: `dl`
- 数据与模型: 完全复用 `baseline_5class`，不改动核心预处理与模型结构
- group 单位: `recording/session`
- grouped CV: `StratifiedGroupKFold`, `{summary['settings']['n_repeats']}` repeats x `{summary['settings']['n_splits']}` folds
- val 划分: 仅从每折训练集内部按 session 再划出一部分做早停，不与测试 group 重叠
- 限制: {summary['settings']['limitation']}

## 2. 每折 split

{chr(10).join(split_lines)}

## 3. 每折结果

{format_table(per_fold_rows)}

## 4. 平均结果表

{format_table(summary_rows)}

## 5. 混淆矩阵图

### 窗口级汇总

![audio_only_window](audio_only_confusion_matrix_window_sum.png)

![pressure_flow_window](pressure_flow_confusion_matrix_window_sum.png)

![multimodal_window](multimodal_confusion_matrix_window_sum.png)

### Session 聚合汇总

![audio_only_session](audio_only_confusion_matrix_session_sum.png)

![pressure_flow_session](pressure_flow_confusion_matrix_session_sum.png)

![multimodal_session](multimodal_confusion_matrix_session_sum.png)

各折单独混淆矩阵保存在 `outputs/grouped_cv_5class/repeat*_fold*/<model>/` 子目录下。

## 6. 稳定性与局限性总结

{chr(10).join(analysis_lines)}
"""
    (cv_root / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
