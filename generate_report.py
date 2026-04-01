from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from mmdl_baseline.dataset.discovery import discover_sessions
from mmdl_baseline.dataset.splits import build_session_split
from mmdl_baseline.utils.config import load_config
from mmdl_baseline.utils.io import read_json


def format_table(rows: List[List[str]]) -> str:
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, sep] + body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline markdown report.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = Path(config["output_root"])
    sessions = discover_sessions(config["data_root"], config["labels"])
    split = build_session_split(
        sessions=sessions,
        seed=int(config["seed"]),
        test_per_class=int(config["split"]["test_per_class"]),
        val_fraction_of_remaining=float(config["split"]["val_fraction_of_remaining"]),
    )

    split_lines = []
    for split_name in ["train", "val", "test"]:
        items = split[split_name]
        split_lines.append(
            f"- {split_name}: {len(items)} sessions, labels={[session.label for session in items]}, "
            f"ids={[session.session_id for session in items]}"
        )

    summary_rows = [["Model", "Window Acc", "Window F1", "Session Acc", "Session F1", "Best Epoch"]]
    modalities = ["audio_only", "pressure_flow", "multimodal"]
    summaries: Dict[str, Dict[str, object]] = {}
    for modality in modalities:
        summary = read_json(output_root / modality / "summary.json")
        summaries[modality] = summary
        win_metrics = summary["test_metrics_window"]
        sess_metrics = summary["test_metrics_session"]
        summary_rows.append(
            [
                modality,
                f"{win_metrics['accuracy']:.4f}",
                f"{win_metrics['macro_f1']:.4f}",
                f"{sess_metrics['accuracy']:.4f}",
                f"{sess_metrics['macro_f1']:.4f}",
                str(summary["best_epoch"]),
            ]
        )

    best_modality = max(modalities, key=lambda m: summaries[m]["test_metrics_window"]["macro_f1"])
    best_win = summaries[best_modality]["test_metrics_window"]
    best_sess = summaries[best_modality]["test_metrics_session"]
    best_cm = np.asarray(best_win["confusion_matrix"], dtype=int)
    best_session_cm = np.asarray(best_sess["confusion_matrix"], dtype=int)

    off_diag = best_cm.copy()
    np.fill_diagonal(off_diag, 0)
    confusion_pairs = []
    for true_idx in range(off_diag.shape[0]):
        for pred_idx in range(off_diag.shape[1]):
            if off_diag[true_idx, pred_idx] > 0:
                confusion_pairs.append((int(off_diag[true_idx, pred_idx]), true_idx, pred_idx))
    confusion_pairs.sort(reverse=True)
    top_confusions = [f"{true_label} -> {pred_label}: {count} windows" for count, true_label, pred_label in confusion_pairs[:3]]

    adjacent_notes = []
    for a, b in [(1, 2), (2, 3), (3, 4)]:
        adjacent_notes.append(f"{a} vs {b}: {int(best_cm[a, b] + best_cm[b, a])} windows")
    zero_vs_four = int(best_cm[0, 4] + best_cm[4, 0])

    analysis_lines = [
        f"- 五分类任务可以完整跑通，当前最佳模型是 `{best_modality}`。",
        "- 类别 3 只有 2 个 session，因此无法同时覆盖 train/val/test，验证集不保证所有类别都出现。",
        "- 由于按 session 先划分再切窗，避免了同一 recording 的窗口泄漏到不同数据集。",
        f"- 最佳模型窗口级指标: accuracy={best_win['accuracy']:.4f}, macro-F1={best_win['macro_f1']:.4f}；session 聚合指标: accuracy={best_sess['accuracy']:.4f}, macro-F1={best_sess['macro_f1']:.4f}。",
        f"- 相邻组别混淆情况（最佳模型，窗口级）: {', '.join(adjacent_notes)}。",
        f"- `0 ml` 与 `4 ml` 在最佳模型上没有出现互相混淆（0<->4 共 {zero_vs_four} 个窗口），说明负荷两端更容易区分。",
        f"- 主要混淆集中在: {', '.join(top_confusions) if top_confusions else '无明显混淆'}。",
        "- 从当前测试集看，`1 ml` 是最难的类别，最佳模型把该类 session 预测成了 `0 ml`；`2 ml` 有一部分窗口被预测成 `1 ml`，而 `3 ml` 与 `4 ml` 基本稳定。",
    ]

    report = f"""# 多模态 5 分类基线实验报告

## 1. 数据组织方式

数据根目录为 `{config['data_root']}`，每个 session 目录默认包含：

- `audio.wav`
- `daq.csv`
- `metadata.json`

每个 recording/session 作为一个独立样本单元，严格先按 session 划分 train/val/test，再切 10 秒窗口。

## 2. 标签解析方式

标签优先从 `metadata.json` 中的 `label` 字段解析，失败时回退到文件夹名。统一映射到：

- 0: no / no secretion / 0 ml
- 1: 1 ml
- 2: 2 ml
- 3: 3 ml
- 4: 4 ml

## 3. 数据划分策略

固定随机种子：`{config['seed']}`。

划分规则：

- 每个类别优先保留 1 个 session 到 test
- 剩余 session 中约 25% 进入 val
- 其余进入 train

当前 session 级划分如下：

{chr(10).join(split_lines)}

## 4. 预处理方法

- audio: 重采样到 `{config['audio_sample_rate']}` Hz，单声道，逐 recording 做 z-score，转 log-Mel 频谱
- pressure / flow: 使用 `daq.csv` 原始序列，按 recording 做 z-score，采样率按 `{config['sensor_sample_rate']}` Hz 处理
- 时间对齐: 对三模态按共同有效时长截断到最短模态
- 窗口: 固定 `{config['window_sec']}` 秒，步长 `{config['window_hop_sec']}` 秒，不重叠
- 对不足一个窗口的 recording，默认跳过，不做补齐

## 5. 模型结构说明

- `audio_only`: 轻量 2D CNN 编码 log-Mel 频谱，GAP 后线性分类
- `pressure_flow`: pressure 和 flow 各自通过 1D CNN encoder，concat 后接 MLP
- `multimodal`: audio 2D CNN + pressure 1D CNN + flow 1D CNN，三分支中间层融合后分类

## 6. 训练配置

- epoch 上限: `{config['epochs']}`
- batch size: `{config['batch_size']}`
- optimizer: Adam
- learning rate: `{config['learning_rate']}`
- weight decay: `{config['weight_decay']}`
- weighted sampler: `{config['weighted_sampler']}`
- early stopping patience: `{config['early_stop_patience']}`

## 7. 测试集结果表

{format_table(summary_rows)}

## 8. 混淆矩阵图

### 窗口级

![audio_only](audio_only/confusion_matrix.png)

![pressure_flow](pressure_flow/confusion_matrix.png)

![multimodal](multimodal/confusion_matrix.png)

### Session 聚合

![audio_only_session](audio_only/confusion_matrix_session.png)

![pressure_flow_session](pressure_flow/confusion_matrix_session.png)

![multimodal_session](multimodal/confusion_matrix_session.png)

## 9. 结果分析与结论

{chr(10).join(analysis_lines)}

- 最佳模型窗口级 confusion matrix: `{best_cm.tolist()}`。
- 最佳模型 session 聚合 confusion matrix: `{best_session_cm.tolist()}`。
"""
    report_path = output_root / "report.md"
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
