from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grouped_cv import build_grouped_cv_splits
from mmdl_baseline.dataset.discovery import discover_sessions
from mmdl_baseline.train_eval import train_and_evaluate_with_splits
from mmdl_baseline.utils.aggregation import AGGREGATION_METHODS, aggregate_predictions_by_session
from mmdl_baseline.utils.config import load_config
from mmdl_baseline.utils.io import ensure_dir, read_json, write_json
from mmdl_baseline.utils.metrics import save_confusion_matrix_figure
from mmdl_baseline.utils.reporting import append_markdown_section
from mmdl_baseline.utils.runtime import log_runtime_environment
from mmdl_baseline.utils.seed import set_seed
from mmdl_baseline.utils.task import resolve_task


def _deep_update(base: Dict[str, object], updates: Dict[str, object]) -> Dict[str, object]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _mean_std(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def _format_metric(metric: Dict[str, float]) -> str:
    return f"{metric['mean']:.4f} ± {metric['std']:.4f}"


def _make_plot(output_path: Path, title: str, labels: List[str], values: List[float], ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=["#355C7D", "#C06C84", "#6C5B7B", "#F67280"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, max(values) * 1.15 if values else 1.0)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chapter 4 experiments for 0/2/4 classification.")
    parser.add_argument("--config", default="configs/chapter4_024.yaml")
    args = parser.parse_args()

    log_runtime_environment()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    output_root = ensure_dir(config["output_root"])

    sessions = discover_sessions(config["data_root"], config["labels"])
    sessions, task_info = resolve_task(config, sessions)
    config = {**config, "task": {**config.get("task", {}), **task_info}}
    cv_cfg = config["grouped_cv"]

    if min(Counter(session.label for session in sessions).values()) < int(cv_cfg["n_splits"]):
        raise ValueError("n_splits exceeds the minimum number of sessions in at least one class.")

    split_defs = build_grouped_cv_splits(
        sessions=sessions,
        seed=int(config["seed"]),
        n_splits=int(cv_cfg["n_splits"]),
        n_repeats=int(cv_cfg["n_repeats"]),
        val_fraction=float(cv_cfg["val_fraction_of_train"]),
    )
    write_json(output_root / "split_manifest.json", [
        {
            "name": split_def["name"],
            "train_ids": [s.session_id for s in split_def["train"]],
            "val_ids": [s.session_id for s in split_def["val"]],
            "test_ids": [s.session_id for s in split_def["test"]],
        }
        for split_def in split_defs
    ])

    experiments: List[Dict[str, object]] = []
    for section in ["main_experiments", "window_length_experiments", "ablation_experiments", "fusion_experiments"]:
        experiments.extend(config.get(section, []))
    experiment_results: Dict[str, object] = {}

    for experiment in experiments:
        exp_name = experiment["name"]
        if exp_name in experiment_results:
            continue
        exp_cfg = copy.deepcopy(config)
        exp_cfg["window_sec"] = float(experiment["window_sec"])
        exp_cfg["window_hop_sec"] = float(experiment["window_sec"])
        for cfg_key in ["config_overrides", "model_overrides"]:
            if experiment.get(cfg_key):
                exp_cfg = _deep_update(exp_cfg, experiment[cfg_key])
        modality = experiment["modality"]
        fold_results: List[Dict[str, object]] = []
        agg_by_method: Dict[str, List[Dict[str, object]]] = {method: [] for method in AGGREGATION_METHODS}
        for split_def in split_defs:
            fold_name = split_def["name"]
            run_dir = output_root / "runs" / exp_name / fold_name
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                summary = read_json(summary_path)
            else:
                summary = train_and_evaluate_with_splits(
                    config=exp_cfg,
                    modality=modality,
                    run_dir=run_dir,
                    splits={"train": split_def["train"], "val": split_def["val"], "test": split_def["test"]},
                    experiment_name=f"chapter4_024/{exp_name}/{fold_name}",
                    append_summary=True,
                )
            predictions = read_json(run_dir / "test_predictions.json")
            for method in AGGREGATION_METHODS:
                session_preds, session_metrics = aggregate_predictions_by_session(
                    predictions,
                    method=method,
                    num_classes=task_info["num_classes"],
                )
                method_dir = ensure_dir(run_dir / method)
                write_json(method_dir / "session_predictions.json", session_preds)
                write_json(method_dir / "summary.json", session_metrics)
                save_confusion_matrix_figure(
                    np.asarray(session_metrics["confusion_matrix"]),
                    labels=task_info["class_names"],
                    output_path=method_dir / "confusion_matrix.png",
                    title=f"{exp_name} {method} confusion matrix",
                )
                agg_by_method[method].append({"fold": fold_name, "metrics": session_metrics})
            fold_results.append(
                {
                    "fold": fold_name,
                    "window_metrics": summary["test_metrics_window"],
                    "default_session_metrics": summary["test_metrics_session"],
                    "best_epoch": summary["best_epoch"],
                }
            )

        window_ms = {
            metric: _mean_std([record["window_metrics"][metric] for record in fold_results])
            for metric in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
        }
        agg_summary = {}
        for method, records in agg_by_method.items():
            method_ms = {
                metric: _mean_std([record["metrics"][metric] for record in records])
                for metric in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
            }
            cm_sum = np.sum([np.asarray(record["metrics"]["confusion_matrix"], dtype=int) for record in records], axis=0)
            save_confusion_matrix_figure(
                cm_sum,
                labels=task_info["class_names"],
                output_path=output_root / f"{exp_name}_{method}_confusion_matrix_sum.png",
                title=f"{exp_name} {method} confusion matrix (sum)",
            )
            agg_summary[method] = {
                "fold_results": records,
                "mean_std": method_ms,
                "confusion_matrix_sum": cm_sum.tolist(),
            }
        best_method = max(AGGREGATION_METHODS, key=lambda m: agg_summary[m]["mean_std"]["macro_f1"]["mean"])
        experiment_results[exp_name] = {
            "name": exp_name,
            "group": experiment["group"],
            "modality": modality,
            "window_sec": float(experiment["window_sec"]),
            "window_level": {"fold_results": fold_results, "mean_std": window_ms},
            "session_aggregation": agg_summary,
            "best_session_method": best_method,
            "best_session_mean_std": agg_summary[best_method]["mean_std"],
        }
        append_markdown_section(
            config["summary_markdown"],
            f"chapter4_024 | {exp_name}",
            [
                f"- python_env: `dl`",
                f"- model: `{modality}`",
                f"- task: `0/2/4` 三分类",
                f"- window_sec: `{experiment['window_sec']}`",
                f"- group: `{experiment['group']}`",
                f"- result_window_mean_std: acc={_format_metric(window_ms['accuracy'])}, macro-F1={_format_metric(window_ms['macro_f1'])}, precision={_format_metric(window_ms['macro_precision'])}, recall={_format_metric(window_ms['macro_recall'])}",
                f"- best_session_method: `{best_method}`",
                f"- result_session_mean_std: acc={_format_metric(agg_summary[best_method]['mean_std']['accuracy'])}, macro-F1={_format_metric(agg_summary[best_method]['mean_std']['macro_f1'])}, precision={_format_metric(agg_summary[best_method]['mean_std']['macro_precision'])}, recall={_format_metric(agg_summary[best_method]['mean_std']['macro_recall'])}",
            ],
        )

    summary = {
        "settings": {
            "python_env": "dl",
            "task_info": task_info,
            "grouped_cv": cv_cfg,
            "note": "未实现稳定的单周期切分，因为当前工程缺少可直接复用的周期标注；窗口长度比较采用多种固定时长。",
        },
        "experiments": experiment_results,
    }
    write_json(output_root / "chapter4_024_summary.json", summary)

    if all(name in experiment_results for name in ["multimodal_5s", "multimodal_10s", "multimodal_20s"]):
        window_labels = ["5s", "10s", "20s"]
        window_values = [
            experiment_results["multimodal_5s"]["best_session_mean_std"]["macro_f1"]["mean"],
            experiment_results["multimodal_10s"]["best_session_mean_std"]["macro_f1"]["mean"],
            experiment_results["multimodal_20s"]["best_session_mean_std"]["macro_f1"]["mean"],
        ]
        _make_plot(
            output_root / "window_length_comparison.png",
            title="Multimodal Session-Level Macro-F1 by Window Length",
            labels=window_labels,
            values=window_values,
            ylabel="Macro-F1",
        )

    if all(name in experiment_results for name in ["multimodal_10s", "multimodal_minus_audio_10s", "multimodal_minus_pressure_10s", "multimodal_minus_flow_10s"]):
        ablation_labels = ["full", "-audio", "-pressure", "-flow"]
        ablation_values = [
            experiment_results["multimodal_10s"]["best_session_mean_std"]["macro_f1"]["mean"],
            experiment_results["multimodal_minus_audio_10s"]["best_session_mean_std"]["macro_f1"]["mean"],
            experiment_results["multimodal_minus_pressure_10s"]["best_session_mean_std"]["macro_f1"]["mean"],
            experiment_results["multimodal_minus_flow_10s"]["best_session_mean_std"]["macro_f1"]["mean"],
        ]
        _make_plot(
            output_root / "ablation_results.png",
            title="Session-Level Macro-F1 for Multimodal Ablations",
            labels=ablation_labels,
            values=ablation_values,
            ylabel="Macro-F1",
        )


if __name__ == "__main__":
    main()
