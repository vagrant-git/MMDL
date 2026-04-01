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
    arr = np.asarray(values, dtype=np.float32)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _format_metric(metric: Dict[str, float]) -> str:
    return f"{metric['mean']:.4f} ± {metric['std']:.4f}"


def _make_plot(output_path: Path, labels: List[str], values: List[float]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#1f4e79", "#2a7f62", "#b85c38", "#7a4f9a", "#c99100", "#7d8b99", "#5b8c5a", "#b04a5a", "#5f6caf"]
    ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_title("Audio Frontend Search: Session-Level Macro-F1")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.0, max(values) * 1.15 if values else 1.0)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search audio frontends under grouped CV.")
    parser.add_argument("--config", default="configs/audio_frontend_search.yaml")
    args = parser.parse_args()

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

    overall_rows: List[Dict[str, object]] = []
    fold_rows: List[Dict[str, object]] = []
    results: Dict[str, object] = {}

    for experiment in config["experiments"]:
        exp_name = str(experiment["name"])
        display_name = str(experiment.get("display_name", exp_name))
        modality = str(experiment["modality"])
        exp_cfg = copy.deepcopy(config)
        exp_cfg["window_sec"] = float(experiment["window_sec"])
        exp_cfg["window_hop_sec"] = float(experiment["window_sec"])
        for cfg_key in ["config_overrides", "model_overrides"]:
            if experiment.get(cfg_key):
                exp_cfg = _deep_update(exp_cfg, experiment[cfg_key])

        fold_results = []
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
                    experiment_name=f"audio_frontend_search/{exp_name}/{fold_name}",
                    append_summary=False,
                )

            predictions = read_json(run_dir / "test_predictions.json")
            window_confusion_matrices.append(np.asarray(summary["test_metrics_window"]["confusion_matrix"], dtype=int))
            for method in AGGREGATION_METHODS:
                _, session_metrics = aggregate_predictions_by_session(
                    predictions,
                    method=method,
                    num_classes=task_info["num_classes"],
                )
                agg_by_method[method].append({"fold": fold_name, "metrics": session_metrics})
            fold_results.append(
                {
                    "fold": fold_name,
                    "window_metrics": summary["test_metrics_window"],
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
        results[exp_name] = {
            "name": exp_name,
            "display_name": display_name,
            "window_sec": float(experiment["window_sec"]),
            "window_level": {"fold_results": fold_results, "mean_std": window_ms},
            "best_session_method": best_method,
            "best_session_mean_std": agg_summary[best_method]["mean_std"],
            "session_aggregation": agg_summary,
        }

        overall_rows.append(
            {
                "experiment": exp_name,
                "display_name": display_name,
                "best_session_method": best_method,
                "window_macro_f1_mean": window_ms["macro_f1"]["mean"],
                "window_macro_f1_std": window_ms["macro_f1"]["std"],
                "session_macro_f1_mean": agg_summary[best_method]["mean_std"]["macro_f1"]["mean"],
                "session_macro_f1_std": agg_summary[best_method]["mean_std"]["macro_f1"]["std"],
                "session_accuracy_mean": agg_summary[best_method]["mean_std"]["accuracy"]["mean"],
                "session_accuracy_std": agg_summary[best_method]["mean_std"]["accuracy"]["std"],
            }
        )
        session_fold_map = {record["fold"]: record["metrics"] for record in agg_summary[best_method]["fold_results"]}
        for fold_record in fold_results:
            fold_rows.append(
                {
                    "experiment": exp_name,
                    "display_name": display_name,
                    "fold": fold_record["fold"],
                    "window_macro_f1": fold_record["window_metrics"]["macro_f1"],
                    "session_macro_f1": session_fold_map[fold_record["fold"]]["macro_f1"],
                    "best_epoch": fold_record["best_epoch"],
                }
            )

        append_markdown_section(
            config["summary_markdown"],
            f"audio_frontend_search | {exp_name}",
            [
                f"- display_name: `{display_name}`",
                f"- window_macro_f1: `{_format_metric(window_ms['macro_f1'])}`",
                f"- session_macro_f1: `{_format_metric(agg_summary[best_method]['mean_std']['macro_f1'])}`",
                f"- best_session_method: `{best_method}`",
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
                "early_stop_patience": int(config["early_stop_patience"]),
            },
        },
        "experiments": results,
    }
    write_json(output_root / "summary.json", summary)
    _write_csv(
        output_root / "overall_results.csv",
        overall_rows,
        [
            "experiment",
            "display_name",
            "best_session_method",
            "window_macro_f1_mean",
            "window_macro_f1_std",
            "session_macro_f1_mean",
            "session_macro_f1_std",
            "session_accuracy_mean",
            "session_accuracy_std",
        ],
    )
    _write_csv(
        output_root / "fold_results.csv",
        fold_rows,
        ["experiment", "display_name", "fold", "window_macro_f1", "session_macro_f1", "best_epoch"],
    )

    ordered = sorted(overall_rows, key=lambda row: row["session_macro_f1_mean"], reverse=True)
    _make_plot(
        output_root / "audio_frontend_comparison.png",
        labels=[row["display_name"] for row in ordered],
        values=[float(row["session_macro_f1_mean"]) for row in ordered],
    )


if __name__ == "__main__":
    main()
