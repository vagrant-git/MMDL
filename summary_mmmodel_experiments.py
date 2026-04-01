from __future__ import annotations

import argparse
import copy
import csv
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


def _load_split_defs_from_manifest(
    manifest_path: Path,
    sessions: List[object],
) -> List[Dict[str, object]]:
    session_map = {session.session_id: session for session in sessions}
    manifest = read_json(manifest_path)
    split_defs: List[Dict[str, object]] = []
    for item in manifest:
        split_defs.append(
            {
                "name": item["name"],
                "train": [session_map[session_id] for session_id in item["train_ids"]],
                "val": [session_map[session_id] for session_id in item["val_ids"]],
                "test": [session_map[session_id] for session_id in item["test_ids"]],
            }
        )
    return split_defs


def _make_bar_plot(
    output_path: Path,
    title: str,
    labels: List[str],
    values: List[float],
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#1f4e79", "#2a7f62", "#b85c38", "#7a4f9a", "#c99100"]
    ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, max(values) * 1.15 if values else 1.0)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_overall_table(output_path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "experiment",
        "display_name",
        "group",
        "modality",
        "window_sec",
        "window_accuracy_mean",
        "window_accuracy_std",
        "window_macro_f1_mean",
        "window_macro_f1_std",
        "window_precision_mean",
        "window_precision_std",
        "window_recall_mean",
        "window_recall_std",
        "best_session_method",
        "session_accuracy_mean",
        "session_accuracy_std",
        "session_macro_f1_mean",
        "session_macro_f1_std",
        "session_precision_mean",
        "session_precision_std",
        "session_recall_mean",
        "session_recall_std",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_fold_table(output_path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "experiment",
        "display_name",
        "group",
        "fold",
        "window_accuracy",
        "window_macro_f1",
        "window_precision",
        "window_recall",
        "best_session_method",
        "session_accuracy",
        "session_macro_f1",
        "session_precision",
        "session_recall",
        "best_epoch",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 0/2/4 experiments for summary-MMmodel.")
    parser.add_argument("--config", default="configs/summary_mmmodel.yaml")
    args = parser.parse_args()

    log_runtime_environment()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    output_root = ensure_dir(config["output_root"])
    summary_markdown = Path(config["summary_markdown"])
    if summary_markdown.exists():
        summary_markdown.unlink()

    sessions = discover_sessions(config["data_root"], config["labels"])
    sessions, task_info = resolve_task(config, sessions)
    config = {**config, "task": {**config.get("task", {}), **task_info}}
    cv_cfg = config["grouped_cv"]

    if min(Counter(session.label for session in sessions).values()) < int(cv_cfg["n_splits"]):
        raise ValueError("n_splits exceeds the minimum number of sessions in at least one class.")

    split_manifest_source = config.get("split_manifest_source")
    if split_manifest_source:
        split_defs = _load_split_defs_from_manifest(Path(str(split_manifest_source)), sessions)
    else:
        split_defs = build_grouped_cv_splits(
            sessions=sessions,
            seed=int(config["seed"]),
            n_splits=int(cv_cfg["n_splits"]),
            n_repeats=int(cv_cfg["n_repeats"]),
            val_fraction=float(cv_cfg["val_fraction_of_train"]),
        )
    write_json(
        output_root / "split_manifest.json",
        [
            {
                "name": split_def["name"],
                "train_ids": [s.session_id for s in split_def["train"]],
                "val_ids": [s.session_id for s in split_def["val"]],
                "test_ids": [s.session_id for s in split_def["test"]],
            }
            for split_def in split_defs
        ],
    )

    experiments: List[Dict[str, object]] = []
    for section in ["main_experiments", "window_length_experiments", "ablation_experiments"]:
        experiments.extend(config.get(section, []))

    experiment_results: Dict[str, object] = {}
    overall_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []

    for experiment in experiments:
        exp_name = str(experiment["name"])
        if exp_name in experiment_results:
            continue

        exp_cfg = copy.deepcopy(config)
        exp_cfg["window_sec"] = float(experiment["window_sec"])
        exp_cfg["window_hop_sec"] = float(experiment["window_sec"])
        for cfg_key in ["config_overrides", "model_overrides"]:
            if experiment.get(cfg_key):
                exp_cfg = _deep_update(exp_cfg, experiment[cfg_key])

        modality = str(experiment["modality"])
        display_name = str(experiment.get("display_name", exp_name))
        group = str(experiment["group"])

        fold_results: List[Dict[str, object]] = []
        agg_by_method: Dict[str, List[Dict[str, object]]] = {method: [] for method in AGGREGATION_METHODS}
        window_confusion_matrices: List[np.ndarray] = []

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
                    experiment_name=f"summary_mmmodel/{exp_name}/{fold_name}",
                    append_summary=True,
                )

            predictions = read_json(run_dir / "test_predictions.json")
            window_confusion_matrices.append(np.asarray(summary["test_metrics_window"]["confusion_matrix"], dtype=int))
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
        window_cm_sum = np.sum(window_confusion_matrices, axis=0)
        save_confusion_matrix_figure(
            window_cm_sum,
            labels=task_info["class_names"],
            output_path=output_root / f"{exp_name}_window_confusion_matrix_sum.png",
            title=f"{exp_name} window confusion matrix (sum)",
        )

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
                output_path=output_root / f"{exp_name}_{method}_session_confusion_matrix_sum.png",
                title=f"{exp_name} {method} session confusion matrix (sum)",
            )
            agg_summary[method] = {
                "fold_results": records,
                "mean_std": method_ms,
                "confusion_matrix_sum": cm_sum.tolist(),
            }

        best_method = max(AGGREGATION_METHODS, key=lambda m: agg_summary[m]["mean_std"]["macro_f1"]["mean"])
        experiment_results[exp_name] = {
            "name": exp_name,
            "display_name": display_name,
            "group": group,
            "modality": modality,
            "window_sec": float(experiment["window_sec"]),
            "window_level": {
                "fold_results": fold_results,
                "mean_std": window_ms,
                "confusion_matrix_sum": window_cm_sum.tolist(),
            },
            "session_aggregation": agg_summary,
            "best_session_method": best_method,
            "best_session_mean_std": agg_summary[best_method]["mean_std"],
        }

        overall_rows.append(
            {
                "experiment": exp_name,
                "display_name": display_name,
                "group": group,
                "modality": modality,
                "window_sec": float(experiment["window_sec"]),
                "window_accuracy_mean": window_ms["accuracy"]["mean"],
                "window_accuracy_std": window_ms["accuracy"]["std"],
                "window_macro_f1_mean": window_ms["macro_f1"]["mean"],
                "window_macro_f1_std": window_ms["macro_f1"]["std"],
                "window_precision_mean": window_ms["macro_precision"]["mean"],
                "window_precision_std": window_ms["macro_precision"]["std"],
                "window_recall_mean": window_ms["macro_recall"]["mean"],
                "window_recall_std": window_ms["macro_recall"]["std"],
                "best_session_method": best_method,
                "session_accuracy_mean": agg_summary[best_method]["mean_std"]["accuracy"]["mean"],
                "session_accuracy_std": agg_summary[best_method]["mean_std"]["accuracy"]["std"],
                "session_macro_f1_mean": agg_summary[best_method]["mean_std"]["macro_f1"]["mean"],
                "session_macro_f1_std": agg_summary[best_method]["mean_std"]["macro_f1"]["std"],
                "session_precision_mean": agg_summary[best_method]["mean_std"]["macro_precision"]["mean"],
                "session_precision_std": agg_summary[best_method]["mean_std"]["macro_precision"]["std"],
                "session_recall_mean": agg_summary[best_method]["mean_std"]["macro_recall"]["mean"],
                "session_recall_std": agg_summary[best_method]["mean_std"]["macro_recall"]["std"],
            }
        )

        session_fold_map = {record["fold"]: record["metrics"] for record in agg_summary[best_method]["fold_results"]}
        for fold_record in fold_results:
            session_metrics = session_fold_map[fold_record["fold"]]
            fold_rows.append(
                {
                    "experiment": exp_name,
                    "display_name": display_name,
                    "group": group,
                    "fold": fold_record["fold"],
                    "window_accuracy": fold_record["window_metrics"]["accuracy"],
                    "window_macro_f1": fold_record["window_metrics"]["macro_f1"],
                    "window_precision": fold_record["window_metrics"]["macro_precision"],
                    "window_recall": fold_record["window_metrics"]["macro_recall"],
                    "best_session_method": best_method,
                    "session_accuracy": session_metrics["accuracy"],
                    "session_macro_f1": session_metrics["macro_f1"],
                    "session_precision": session_metrics["macro_precision"],
                    "session_recall": session_metrics["macro_recall"],
                    "best_epoch": fold_record["best_epoch"],
                }
            )

        append_markdown_section(
            config["summary_markdown"],
            f"summary_mmmodel | {exp_name}",
            [
                f"- model: `{display_name}`",
                f"- modality: `{modality}`",
                f"- group: `{group}`",
                f"- window_sec: `{experiment['window_sec']}`",
                f"- window_mean_std: acc={_format_metric(window_ms['accuracy'])}, macro-F1={_format_metric(window_ms['macro_f1'])}, precision={_format_metric(window_ms['macro_precision'])}, recall={_format_metric(window_ms['macro_recall'])}",
                f"- best_session_method: `{best_method}`",
                f"- session_mean_std: acc={_format_metric(agg_summary[best_method]['mean_std']['accuracy'])}, macro-F1={_format_metric(agg_summary[best_method]['mean_std']['macro_f1'])}, precision={_format_metric(agg_summary[best_method]['mean_std']['macro_precision'])}, recall={_format_metric(agg_summary[best_method]['mean_std']['macro_recall'])}",
            ],
        )

    summary = {
        "settings": {
            "task_info": task_info,
            "grouped_cv": cv_cfg,
            "training": {
                "epochs": int(config["epochs"]),
                "batch_size": int(config["batch_size"]),
                "learning_rate": float(config["learning_rate"]),
                "weight_decay": float(config["weight_decay"]),
                "weighted_sampler": bool(config["weighted_sampler"]),
                "class_weight": bool(config["class_weight"]),
                "loss": str(config.get("loss", "cross_entropy")),
                "early_stop_patience": int(config["early_stop_patience"]),
            },
            "report": config.get("report", {}),
            "note": "Session-level grouped CV split is fixed before window generation to avoid leakage.",
        },
        "experiments": experiment_results,
    }
    write_json(output_root / "summary.json", summary)
    _save_overall_table(output_root / "overall_results.csv", overall_rows)
    _save_fold_table(output_root / "fold_results.csv", fold_rows)

    report_cfg = config.get("report", {})
    main_experiment_names = [name for name in report_cfg.get("main_experiment_names", []) if name in experiment_results]
    if main_experiment_names:
        _make_bar_plot(
            output_root / "model_comparison.png",
            title="Session-Level Macro-F1 by Model",
            labels=[experiment_results[name]["display_name"] for name in main_experiment_names],
            values=[experiment_results[name]["best_session_mean_std"]["macro_f1"]["mean"] for name in main_experiment_names],
            ylabel="Macro-F1",
        )

    window_experiment_names = [name for name in report_cfg.get("window_experiment_names", []) if name in experiment_results]
    if window_experiment_names:
        _make_bar_plot(
            output_root / "window_length_comparison.png",
            title="HCAF-Net Session-Level Macro-F1 by Window Length",
            labels=[f"{experiment_results[name]['window_sec']:.0f} s" for name in window_experiment_names],
            values=[experiment_results[name]["best_session_mean_std"]["macro_f1"]["mean"] for name in window_experiment_names],
            ylabel="Macro-F1",
        )

    ablation_experiment_names = [name for name in report_cfg.get("ablation_experiment_names", []) if name in experiment_results]
    if ablation_experiment_names:
        _make_bar_plot(
            output_root / "ablation_results.png",
            title="HCAF-Net Session-Level Macro-F1 by Ablation",
            labels=[experiment_results[name]["display_name"] for name in ablation_experiment_names],
            values=[experiment_results[name]["best_session_mean_std"]["macro_f1"]["mean"] for name in ablation_experiment_names],
            ylabel="Macro-F1",
        )


if __name__ == "__main__":
    main()
