from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from mmdl_baseline.dataset.discovery import SessionRecord, discover_sessions
from mmdl_baseline.dataset.windowed_dataset import MultiModalWindowDataset
from mmdl_baseline.models.factory import build_model
from mmdl_baseline.train_eval import choose_device, move_batch_to_device
from mmdl_baseline.utils.config import load_config
from mmdl_baseline.utils.io import ensure_dir
from mmdl_baseline.utils.task import resolve_task


def _deep_update(base: Dict[str, object], updates: Dict[str, object]) -> Dict[str, object]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _find_experiment(config: Dict[str, object], experiment_name: str) -> Dict[str, object]:
    for section in ["main_experiments", "window_length_experiments", "ablation_experiments"]:
        for experiment in config.get(section, []):
            if str(experiment["name"]) == experiment_name:
                return experiment
    raise KeyError(f"Experiment not found: {experiment_name}")


def _build_experiment_config(config: Dict[str, object], experiment_name: str) -> Dict[str, object]:
    experiment = _find_experiment(config, experiment_name)
    exp_cfg = copy.deepcopy(config)
    exp_cfg["window_sec"] = float(experiment["window_sec"])
    exp_cfg["window_hop_sec"] = float(experiment["window_sec"])
    for cfg_key in ["config_overrides", "model_overrides"]:
        if experiment.get(cfg_key):
            exp_cfg = _deep_update(exp_cfg, experiment[cfg_key])
    return exp_cfg


def _session_records_from_json(items: List[Dict[str, object]]) -> List[SessionRecord]:
    return [SessionRecord(**item) for item in items]


def _confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def _save_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rank_positions(rows: List[Dict[str, object]]) -> None:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["session_id"])].append(row)
    for session_rows in grouped.values():
        session_rows.sort(key=lambda item: float(item["start_sec"]))
        total = len(session_rows)
        for idx, row in enumerate(session_rows):
            row["window_index_in_session"] = idx
            row["num_windows_in_session"] = total
            row["position_ratio"] = (idx + 0.5) / total


def _plot_error_by_position(rows: List[Dict[str, object]], output_path: Path) -> Dict[str, float]:
    bin_edges = np.linspace(0.0, 1.0, 6)
    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    error_rates = []
    counts = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        bucket = [row for row in rows if left <= float(row["position_ratio"]) < right or (right == 1.0 and float(row["position_ratio"]) <= right)]
        counts.append(len(bucket))
        if bucket:
            error_rates.append(float(np.mean([0.0 if row["correct"] else 1.0 for row in bucket])))
        else:
            error_rates.append(0.0)

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(bin_labels))
    ax1.bar(x, counts, color="#9ecae1", alpha=0.8)
    ax1.set_ylabel("Window count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels)
    ax1.set_xlabel("Relative position within session")

    ax2 = ax1.twinx()
    ax2.plot(x, error_rates, color="#d95f0e", marker="o", linewidth=2)
    ax2.set_ylabel("Error rate")
    ax2.set_ylim(0.0, max(error_rates + [0.1]) * 1.25)
    ax1.set_title("Window Error Rate by Relative Session Position")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    boundary_rows = [row for row in rows if float(row["position_ratio"]) < 0.2 or float(row["position_ratio"]) >= 0.8]
    middle_rows = [row for row in rows if 0.2 <= float(row["position_ratio"]) < 0.8]
    return {
        "boundary_error_rate": float(np.mean([0.0 if row["correct"] else 1.0 for row in boundary_rows])) if boundary_rows else 0.0,
        "middle_error_rate": float(np.mean([0.0 if row["correct"] else 1.0 for row in middle_rows])) if middle_rows else 0.0,
    }


def _plot_gate_by_class(rows: List[Dict[str, object]], class_names: List[str], output_path: Path) -> Dict[str, Dict[str, float]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["true_label"])].append(float(row["audio_gate_weight"]))

    means = [float(np.mean(grouped[idx])) if grouped[idx] else 0.0 for idx in range(len(class_names))]
    stds = [float(np.std(grouped[idx])) if grouped[idx] else 0.0 for idx in range(len(class_names))]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(class_names, means, color=["#4c78a8", "#72b7b2", "#f58518"])
    ax.errorbar(class_names, means, yerr=stds, fmt="none", ecolor="black", capsize=4, linewidth=1.2)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean audio gate weight")
    ax.set_title("Confidence Gate by True Class")
    for idx, value in enumerate(means):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    return {
        class_names[idx]: {"mean_audio_gate": means[idx], "std_audio_gate": stds[idx]}
        for idx in range(len(class_names))
    }


def _plot_gate_vs_confidence(rows: List[Dict[str, object]], output_path: Path) -> Dict[str, float]:
    audio_conf = np.asarray([float(row["audio_conf_top1"]) for row in rows], dtype=np.float32)
    sensor_conf = np.asarray([float(row["sensor_conf_top1"]) for row in rows], dtype=np.float32)
    audio_gate = np.asarray([float(row["audio_gate_weight"]) for row in rows], dtype=np.float32)
    sensor_gate = np.asarray([float(row["sensor_gate_weight"]) for row in rows], dtype=np.float32)

    audio_corr = float(np.corrcoef(audio_conf, audio_gate)[0, 1]) if len(rows) > 1 else 0.0
    sensor_corr = float(np.corrcoef(sensor_conf, sensor_gate)[0, 1]) if len(rows) > 1 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    axes[0].scatter(audio_conf, audio_gate, s=10, alpha=0.35, color="#1f77b4")
    axes[0].set_title(f"Audio expert confidence vs gate (r={audio_corr:.2f})")
    axes[0].set_xlabel("Audio expert top-1 probability")
    axes[0].set_ylabel("Audio gate weight")
    axes[0].set_ylim(0.0, 1.0)

    axes[1].scatter(sensor_conf, sensor_gate, s=10, alpha=0.35, color="#ff7f0e")
    axes[1].set_title(f"Sensor expert confidence vs gate (r={sensor_corr:.2f})")
    axes[1].set_xlabel("Sensor expert top-1 probability")
    axes[1].set_ylabel("Sensor gate weight")
    axes[1].set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return {"audio_corr": audio_corr, "sensor_corr": sensor_corr}


def _majority_vote(values: List[int]) -> int:
    counts = Counter(values)
    return counts.most_common(1)[0][0]


def _summarize_sessions(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["session_id"])].append(row)

    summaries: List[Dict[str, object]] = []
    for session_id, items in grouped.items():
        items.sort(key=lambda item: float(item["start_sec"]))
        true_label = int(items[0]["true_label"])
        pred_label = _majority_vote([int(item["pred_label"]) for item in items])
        summaries.append(
            {
                "session_id": session_id,
                "fold": items[0]["fold"],
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": int(pred_label == true_label),
                "num_windows": len(items),
                "mean_audio_gate": float(np.mean([float(item["audio_gate_weight"]) for item in items])),
                "std_audio_gate": float(np.std([float(item["audio_gate_weight"]) for item in items])),
                "mean_sensor_gate": float(np.mean([float(item["sensor_gate_weight"]) for item in items])),
                "window_error_rate": float(np.mean([0.0 if item["correct"] else 1.0 for item in items])),
                "mean_final_confidence": float(np.mean([float(item["final_confidence"]) for item in items])),
            }
        )
    summaries.sort(key=lambda item: (item["fold"], item["session_id"]))
    return summaries


def _plot_session_gate(session_id: str, rows: List[Dict[str, object]], class_names: List[str], output_path: Path) -> None:
    session_rows = [row for row in rows if str(row["session_id"]) == session_id]
    session_rows.sort(key=lambda item: float(item["start_sec"]))
    x = [float(row["start_sec"]) for row in session_rows]
    audio_gate = [float(row["audio_gate_weight"]) for row in session_rows]
    sensor_gate = [float(row["sensor_gate_weight"]) for row in session_rows]
    correctness = [int(row["correct"]) for row in session_rows]
    true_label = class_names[int(session_rows[0]["true_label"])]
    pred_label = class_names[_majority_vote([int(row["pred_label"]) for row in session_rows])]

    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    ax.plot(x, audio_gate, label="Audio gate", color="#1f77b4", linewidth=2)
    ax.plot(x, sensor_gate, label="Sensor gate", color="#ff7f0e", linewidth=2)
    wrong_x = [x[idx] for idx, value in enumerate(correctness) if value == 0]
    wrong_y = [audio_gate[idx] for idx, value in enumerate(correctness) if value == 0]
    if wrong_x:
        ax.scatter(wrong_x, wrong_y, color="#d62728", s=28, label="Wrong window")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Window start (s)")
    ax.set_ylabel("Gate weight")
    ax.set_title(f"{session_id} | true={true_label}, session-pred={pred_label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_attention_examples(
    best_examples: Dict[int, Dict[str, object]],
    class_names: List[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(class_names), 2, figsize=(10.0, 3.2 * len(class_names)))
    if len(class_names) == 1:
        axes = np.asarray([axes])

    for class_idx, class_name in enumerate(class_names):
        example = best_examples.get(class_idx)
        for axis in axes[class_idx]:
            axis.axis("off")
        if not example:
            continue

        a2s = np.asarray(example["audio_to_sensor_attn"], dtype=np.float32).mean(axis=0)
        s2a = np.asarray(example["sensor_to_audio_attn"], dtype=np.float32).mean(axis=0)

        ax_left = axes[class_idx, 0]
        ax_right = axes[class_idx, 1]

        im0 = ax_left.imshow(a2s, aspect="auto", cmap="magma")
        ax_left.set_title(
            f"Class {class_name} A->S\n{example['session_id']} @ {example['start_sec']:.1f}s, conf={example['final_confidence']:.3f}"
        )
        ax_left.set_xlabel("Sensor tokens")
        ax_left.set_ylabel("Audio tokens")
        fig.colorbar(im0, ax=ax_left, fraction=0.046, pad=0.04)

        im1 = ax_right.imshow(s2a, aspect="auto", cmap="viridis")
        ax_right.set_title(f"Class {class_name} S->A")
        ax_right.set_xlabel("Audio tokens")
        ax_right.set_ylabel("Sensor tokens")
        fig.colorbar(im1, ax=ax_right, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpretability and error analysis for the best HCAF model.")
    parser.add_argument("--config", default="configs/hcaf_confgate_improve_search.yaml")
    parser.add_argument("--experiment", default="hcaf_confgate_residual_pcen96hp80_5s")
    parser.add_argument("--output-dir", default="summary-MMmodel/hcaf_confgate_interpretability")
    args = parser.parse_args()

    base_config = load_config(args.config)
    exp_config = _build_experiment_config(base_config, args.experiment)

    sessions = discover_sessions(exp_config["data_root"], exp_config["labels"])
    sessions, task_info = resolve_task(exp_config, sessions)
    exp_config = {**exp_config, "task": {**exp_config.get("task", {}), **task_info}}

    output_dir = ensure_dir(args.output_dir)
    device = choose_device()
    class_names = [str(name) for name in task_info["class_names"]]

    all_rows: List[Dict[str, object]] = []
    best_examples: Dict[int, Dict[str, object]] = {}
    session_predictions: List[Dict[str, object]] = []

    run_root = Path(exp_config["output_root"]) / "runs" / args.experiment
    for fold_dir in sorted(run_root.glob("repeat*_fold*")):
        fold_name = fold_dir.name
        split_json = json.loads((fold_dir / "session_split.json").read_text(encoding="utf-8"))
        test_sessions = _session_records_from_json(split_json["test"])
        test_dataset = MultiModalWindowDataset(test_sessions, exp_config, modality="multimodal")
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(exp_config["batch_size"]),
            shuffle=False,
            num_workers=int(exp_config["num_workers"]),
        )

        model = build_model("multimodal", exp_config, num_classes=task_info["num_classes"]).to(device)
        state_dict = torch.load(fold_dir / "best_model.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        if hasattr(model, "capture_debug"):
            model.capture_debug = True

        with torch.no_grad():
            for batch in test_loader:
                labels = batch["label"].to(device)
                logits = model(move_batch_to_device(batch, device))
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                debug = getattr(model, "last_debug", {})

                for idx in range(labels.shape[0]):
                    row = {
                        "fold": fold_name,
                        "session_id": batch["session_id"][idx],
                        "start_sec": float(batch["start_sec"][idx].item()),
                        "true_label": int(labels[idx].item()),
                        "pred_label": int(preds[idx].item()),
                        "correct": int(preds[idx].item() == labels[idx].item()),
                        "final_confidence": float(probs[idx].max().item()),
                        "audio_gate_weight": float(debug["audio_gate_weight"][idx, 0].cpu().item()),
                        "sensor_gate_weight": float(debug["sensor_gate_weight"][idx, 0].cpu().item()),
                        "audio_conf_top1": float(debug["audio_conf_features"][idx, 0].cpu().item()),
                        "audio_conf_margin": float(debug["audio_conf_features"][idx, 1].cpu().item()),
                        "audio_conf_entropyinv": float(debug["audio_conf_features"][idx, 2].cpu().item()),
                        "sensor_conf_top1": float(debug["sensor_conf_features"][idx, 0].cpu().item()),
                        "sensor_conf_margin": float(debug["sensor_conf_features"][idx, 1].cpu().item()),
                        "sensor_conf_entropyinv": float(debug["sensor_conf_features"][idx, 2].cpu().item()),
                    }
                    all_rows.append(row)

                    class_idx = int(labels[idx].item())
                    if row["correct"]:
                        current = best_examples.get(class_idx)
                        if current is None or float(row["final_confidence"]) > float(current["final_confidence"]):
                            best_examples[class_idx] = {
                                **row,
                                "audio_to_sensor_attn": debug["audio_to_sensor_attn"][idx].cpu().tolist(),
                                "sensor_to_audio_attn": debug["sensor_to_audio_attn"][idx].cpu().tolist(),
                            }

        majority_predictions_path = fold_dir / "majority_voting" / "session_predictions.json"
        if majority_predictions_path.exists():
            fold_session_predictions = json.loads(majority_predictions_path.read_text(encoding="utf-8"))
            for item in fold_session_predictions:
                session_predictions.append({"fold": fold_name, **item})

    _rank_positions(all_rows)
    session_rows = _summarize_sessions(all_rows)

    window_fieldnames = [
        "fold",
        "session_id",
        "start_sec",
        "window_index_in_session",
        "num_windows_in_session",
        "position_ratio",
        "true_label",
        "pred_label",
        "correct",
        "final_confidence",
        "audio_gate_weight",
        "sensor_gate_weight",
        "audio_conf_top1",
        "audio_conf_margin",
        "audio_conf_entropyinv",
        "sensor_conf_top1",
        "sensor_conf_margin",
        "sensor_conf_entropyinv",
    ]
    _write_csv(output_dir / "window_debug.csv", all_rows, window_fieldnames)
    _write_csv(
        output_dir / "session_debug.csv",
        session_rows,
        [
            "session_id",
            "fold",
            "true_label",
            "pred_label",
            "correct",
            "num_windows",
            "mean_audio_gate",
            "std_audio_gate",
            "mean_sensor_gate",
            "window_error_rate",
            "mean_final_confidence",
        ],
    )

    window_cm = _confusion_matrix(
        [int(row["true_label"]) for row in all_rows],
        [int(row["pred_label"]) for row in all_rows],
        num_classes=len(class_names),
    )
    _save_confusion_matrix(window_cm, class_names, "Window-level confusion matrix", output_dir / "window_confusion_matrix.png")

    if session_predictions:
        session_cm = _confusion_matrix(
            [int(item["true_label"]) for item in session_predictions],
            [int(item["pred_label"]) for item in session_predictions],
            num_classes=len(class_names),
        )
        _save_confusion_matrix(
            session_cm,
            class_names,
            "Session-level confusion matrix (majority voting)",
            output_dir / "session_confusion_matrix_majority.png",
        )
    else:
        session_cm = np.zeros((len(class_names), len(class_names)), dtype=int)

    boundary_stats = _plot_error_by_position(all_rows, output_dir / "error_rate_by_position.png")
    gate_by_class = _plot_gate_by_class(all_rows, class_names, output_dir / "audio_gate_by_class.png")
    gate_corr = _plot_gate_vs_confidence(all_rows, output_dir / "gate_vs_expert_confidence.png")
    _plot_attention_examples(best_examples, class_names, output_dir / "attention_examples.png")

    correct_sessions = [row for row in session_rows if int(row["correct"]) == 1]
    wrong_sessions = [row for row in session_rows if int(row["correct"]) == 0]
    selected_sessions: List[Dict[str, object]] = []
    if correct_sessions:
        selected_sessions.append(min(correct_sessions, key=lambda item: float(item["mean_audio_gate"])))
        selected_sessions.append(max(correct_sessions, key=lambda item: float(item["mean_audio_gate"])))
    if wrong_sessions:
        selected_sessions.append(max(wrong_sessions, key=lambda item: float(item["window_error_rate"])))

    seen_session_ids = set()
    unique_selected_sessions: List[Dict[str, object]] = []
    for item in selected_sessions:
        if item["session_id"] in seen_session_ids:
            continue
        seen_session_ids.add(item["session_id"])
        unique_selected_sessions.append(item)

    for item in unique_selected_sessions:
        _plot_session_gate(
            str(item["session_id"]),
            all_rows,
            class_names,
            output_dir / f"gate_session_{item['session_id']}.png",
        )

    off_diagonal_window = []
    off_diagonal_session = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i == j:
                continue
            off_diagonal_window.append((int(window_cm[i, j]), class_names[i], class_names[j]))
            off_diagonal_session.append((int(session_cm[i, j]), class_names[i], class_names[j]))
    off_diagonal_window.sort(reverse=True)
    off_diagonal_session.sort(reverse=True)

    correct_audio_gate = float(np.mean([float(row["audio_gate_weight"]) for row in all_rows if int(row["correct"]) == 1]))
    wrong_audio_gate = float(np.mean([float(row["audio_gate_weight"]) for row in all_rows if int(row["correct"]) == 0])) if any(
        int(row["correct"]) == 0 for row in all_rows
    ) else 0.0

    report_lines = [
        f"# HCAF 可解释性与错误分析：`{args.experiment}`",
        "",
        "## 核心发现",
        "",
        f"- 共回放 `3` 个 fold、`{len(all_rows)}` 个测试窗口、`{len(session_rows)}` 个测试 session。",
        f"- 窗口级最主要的混淆是 `{off_diagonal_window[0][1]} -> {off_diagonal_window[0][2]}`（`{off_diagonal_window[0][0]}` 个窗口）；session 级最主要混淆是 `{off_diagonal_session[0][1]} -> {off_diagonal_session[0][2]}`（`{off_diagonal_session[0][0]}` 个 session）。",
        f"- 边界窗口（前 20% + 后 20%）错误率为 `{boundary_stats['boundary_error_rate']:.4f}`，中间 60% 错误率为 `{boundary_stats['middle_error_rate']:.4f}`。",
        f"- audio gate 与 audio expert top-1 confidence 的相关系数为 `{gate_corr['audio_corr']:.3f}`；sensor gate 与 sensor expert top-1 confidence 的相关系数为 `{gate_corr['sensor_corr']:.3f}`。",
        f"- 正确窗口的平均 audio gate 为 `{correct_audio_gate:.4f}`；错误窗口为 `{wrong_audio_gate:.4f}`。",
        "",
        "## 按类别的 gate 倾向",
        "",
    ]
    for class_name in class_names:
        report_lines.append(
            f"- 类别 `{class_name}` 的平均 audio gate 为 `{gate_by_class[class_name]['mean_audio_gate']:.4f} ± {gate_by_class[class_name]['std_audio_gate']:.4f}`。"
        )

    report_lines.extend(
        [
            "",
            "## 典型 session",
            "",
        ]
    )
    for item in unique_selected_sessions:
        tag = "correct" if int(item["correct"]) == 1 else "misclassified"
        report_lines.append(
            f"- `{item['session_id']}`: `{tag}`，true=`{class_names[int(item['true_label'])]}`，pred=`{class_names[int(item['pred_label'])]}`，mean audio gate=`{float(item['mean_audio_gate']):.4f}`，window error rate=`{float(item['window_error_rate']):.4f}`。"
        )

    report_lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 当前分析能验证 gate 是否随着各 expert 置信度变化而重新分配权重，但不能直接证明某个时段一定是“环境噪音”，因为现有数据没有逐窗噪声标注。",
            "- 边界效应分析采用的是 session 相对位置代理指标；由于缺少吞咽事件的精确起止标注，暂时无法直接判断“事件恰好被 5 s 窗截断”这一更细粒度问题。",
        ]
    )

    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "num_windows": len(all_rows),
                "num_sessions": len(session_rows),
                "window_confusion_matrix": window_cm.tolist(),
                "session_confusion_matrix": session_cm.tolist(),
                "boundary_stats": boundary_stats,
                "gate_confidence_correlation": gate_corr,
                "gate_by_class": gate_by_class,
                "selected_sessions": unique_selected_sessions,
                "best_examples": best_examples,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
