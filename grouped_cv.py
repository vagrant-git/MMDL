from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from mmdl_baseline.dataset.discovery import SessionRecord, discover_sessions
from mmdl_baseline.dataset.splits import build_train_val_split
from mmdl_baseline.train_eval import train_and_evaluate_with_splits
from mmdl_baseline.utils.config import load_config
from mmdl_baseline.utils.io import ensure_dir, write_json
from mmdl_baseline.utils.metrics import save_confusion_matrix_figure
from mmdl_baseline.utils.reporting import append_markdown_section
from mmdl_baseline.utils.runtime import log_runtime_environment
from mmdl_baseline.utils.seed import set_seed
from mmdl_baseline.utils.task import resolve_task


def build_grouped_cv_splits(
    sessions: List[SessionRecord],
    seed: int,
    n_splits: int,
    n_repeats: int,
    val_fraction: float,
) -> List[Dict[str, List[SessionRecord]]]:
    ordered = sorted(sessions, key=lambda x: x.session_id)
    labels = np.asarray([session.label for session in ordered])
    groups = np.asarray([session.session_id for session in ordered])
    splits: List[Dict[str, List[SessionRecord]]] = []
    for repeat_idx in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + repeat_idx,
        )
        for fold_idx, (train_val_idx, test_idx) in enumerate(splitter.split(np.zeros(len(ordered)), labels, groups=groups), start=1):
            train_val_sessions = [ordered[i] for i in train_val_idx]
            test_sessions = [ordered[i] for i in test_idx]
            inner = build_train_val_split(
                sessions=train_val_sessions,
                seed=seed + repeat_idx * 100 + fold_idx,
                val_fraction=val_fraction,
            )
            splits.append(
                {
                    "name": f"repeat{repeat_idx + 1}_fold{fold_idx}",
                    "train": inner["train"],
                    "val": inner["val"],
                    "test": sorted(test_sessions, key=lambda x: x.session_id),
                }
            )
    return splits


def format_metric(mean_val: float, std_val: float) -> str:
    return f"{mean_val:.4f} ± {std_val:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grouped cross-validation for multimodal baseline.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    log_runtime_environment()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    cv_root = ensure_dir("outputs/grouped_cv_5class")
    modalities = ["audio_only", "pressure_flow", "multimodal"]
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
    split_manifest = []
    for split_def in split_defs:
        split_manifest.append(
            {
                "name": split_def["name"],
                "train_ids": [s.session_id for s in split_def["train"]],
                "val_ids": [s.session_id for s in split_def["val"]],
                "test_ids": [s.session_id for s in split_def["test"]],
                "train_labels": [s.label for s in split_def["train"]],
                "val_labels": [s.label for s in split_def["val"]],
                "test_labels": [s.label for s in split_def["test"]],
            }
        )
    write_json(cv_root / "split_manifest.json", split_manifest)

    all_results: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for split_def in split_defs:
        fold_name = split_def["name"]
        print(f"[grouped_cv] start {fold_name}", flush=True)
        for modality in modalities:
            run_dir = cv_root / fold_name / modality
            summary = train_and_evaluate_with_splits(
                config=config,
                modality=modality,
                run_dir=run_dir,
                splits={
                    "train": split_def["train"],
                    "val": split_def["val"],
                    "test": split_def["test"],
                },
                experiment_name=f"grouped_cv/{fold_name}",
                append_summary=True,
            )
            record = {
                "fold": fold_name,
                "modality": modality,
                "window_metrics": summary["test_metrics_window"],
                "session_metrics": summary["test_metrics_session"],
                "best_epoch": summary["best_epoch"],
            }
            all_results[modality].append(record)

    aggregate_summary: Dict[str, object] = {
        "settings": {
            "python_env": "dl",
            "n_splits": int(cv_cfg["n_splits"]),
            "n_repeats": int(cv_cfg["n_repeats"]),
            "val_fraction_of_train": float(cv_cfg["val_fraction_of_train"]),
            "limitation": "3 ml 仅有 2 个 session，因此只能使用 2-fold grouped CV；更高 fold 数不成立。",
            "task_info": task_info,
        },
        "per_model": {},
    }

    for modality, records in all_results.items():
        window_acc = np.asarray([r["window_metrics"]["accuracy"] for r in records], dtype=np.float32)
        window_f1 = np.asarray([r["window_metrics"]["macro_f1"] for r in records], dtype=np.float32)
        window_p = np.asarray([r["window_metrics"]["macro_precision"] for r in records], dtype=np.float32)
        window_r = np.asarray([r["window_metrics"]["macro_recall"] for r in records], dtype=np.float32)
        session_acc = np.asarray([r["session_metrics"]["accuracy"] for r in records], dtype=np.float32)
        session_f1 = np.asarray([r["session_metrics"]["macro_f1"] for r in records], dtype=np.float32)
        session_p = np.asarray([r["session_metrics"]["macro_precision"] for r in records], dtype=np.float32)
        session_r = np.asarray([r["session_metrics"]["macro_recall"] for r in records], dtype=np.float32)
        cm_window = np.sum([np.asarray(r["window_metrics"]["confusion_matrix"], dtype=int) for r in records], axis=0)
        cm_session = np.sum([np.asarray(r["session_metrics"]["confusion_matrix"], dtype=int) for r in records], axis=0)
        save_confusion_matrix_figure(
            cm_window,
            labels=task_info["class_names"],
            output_path=cv_root / f"{modality}_confusion_matrix_window_sum.png",
            title=f"{modality} grouped CV window confusion matrix (sum)",
        )
        save_confusion_matrix_figure(
            cm_session,
            labels=task_info["class_names"],
            output_path=cv_root / f"{modality}_confusion_matrix_session_sum.png",
            title=f"{modality} grouped CV session confusion matrix (sum)",
        )
        aggregate_summary["per_model"][modality] = {
            "fold_results": records,
            "window_mean_std": {
                "accuracy": {"mean": float(window_acc.mean()), "std": float(window_acc.std(ddof=0))},
                "macro_f1": {"mean": float(window_f1.mean()), "std": float(window_f1.std(ddof=0))},
                "macro_precision": {"mean": float(window_p.mean()), "std": float(window_p.std(ddof=0))},
                "macro_recall": {"mean": float(window_r.mean()), "std": float(window_r.std(ddof=0))},
            },
            "session_mean_std": {
                "accuracy": {"mean": float(session_acc.mean()), "std": float(session_acc.std(ddof=0))},
                "macro_f1": {"mean": float(session_f1.mean()), "std": float(session_f1.std(ddof=0))},
                "macro_precision": {"mean": float(session_p.mean()), "std": float(session_p.std(ddof=0))},
                "macro_recall": {"mean": float(session_r.mean()), "std": float(session_r.std(ddof=0))},
            },
            "window_confusion_matrix_sum": cm_window.tolist(),
            "session_confusion_matrix_sum": cm_session.tolist(),
        }
        mean_std = aggregate_summary["per_model"][modality]["window_mean_std"]
        append_markdown_section(
            config["summary_markdown"],
            f"grouped_cv aggregate | {modality}",
            [
                f"- python_env: `dl`",
                f"- model: `{modality}`",
                f"- method: grouped CV (`{cv_cfg['n_repeats']}` repeats x `{cv_cfg['n_splits']}` folds), group=session。",
                f"- result_window_mean_std: acc={format_metric(mean_std['accuracy']['mean'], mean_std['accuracy']['std'])}, macro-F1={format_metric(mean_std['macro_f1']['mean'], mean_std['macro_f1']['std'])}, precision={format_metric(mean_std['macro_precision']['mean'], mean_std['macro_precision']['std'])}, recall={format_metric(mean_std['macro_recall']['mean'], mean_std['macro_recall']['std'])}",
            ],
        )

    write_json(cv_root / "grouped_cv_summary.json", aggregate_summary)


if __name__ == "__main__":
    main()
