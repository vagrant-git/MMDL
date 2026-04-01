from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from mmdl_baseline.utils.aggregation import AGGREGATION_METHODS, aggregate_predictions_by_session
from mmdl_baseline.utils.config import load_config
from mmdl_baseline.utils.io import ensure_dir, read_json, write_json
from mmdl_baseline.utils.metrics import save_confusion_matrix_figure
from mmdl_baseline.utils.reporting import append_markdown_section


def format_metric(mean_val: float, std_val: float) -> str:
    return f"{mean_val:.4f} ± {std_val:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate session-level aggregation on grouped CV outputs.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    src_root = Path("outputs/grouped_cv_5class")
    out_root = ensure_dir("outputs/grouped_cv_5class_session_agg")
    grouped_summary = read_json(src_root / "grouped_cv_summary.json")
    split_manifest = read_json(src_root / "split_manifest.json")
    modalities = ["audio_only", "pressure_flow", "multimodal"]

    aggregate_summary: Dict[str, object] = {
        "settings": {
            "python_env": "dl",
            "source_grouped_cv_dir": str(src_root),
            "aggregation_methods": AGGREGATION_METHODS,
            "note": "logit averaging uses averaged log-probabilities because raw logits were not persisted; this preserves the same argmax decision as logits up to per-window additive constants.",
            "split_manifest": split_manifest,
        },
        "per_model": {},
    }

    for modality in modalities:
        model_window_fold_results = grouped_summary["per_model"][modality]["fold_results"]
        model_output: Dict[str, object] = {
            "window_level": {
                "fold_results": model_window_fold_results,
                "mean_std": grouped_summary["per_model"][modality]["window_mean_std"],
                "confusion_matrix_sum": grouped_summary["per_model"][modality]["window_confusion_matrix_sum"],
            },
            "session_aggregation": {},
        }
        for method in AGGREGATION_METHODS:
            fold_records: List[Dict[str, object]] = []
            cm_sum = np.zeros((5, 5), dtype=int)
            for fold_entry in split_manifest:
                fold_name = fold_entry["name"]
                prediction_path = src_root / fold_name / modality / "test_predictions.json"
                predictions = read_json(prediction_path)
                session_predictions, metrics = aggregate_predictions_by_session(predictions, method=method, num_classes=5)
                fold_dir = ensure_dir(out_root / fold_name / modality / method)
                write_json(fold_dir / "session_predictions.json", session_predictions)
                write_json(fold_dir / "summary.json", metrics)
                save_confusion_matrix_figure(
                    np.asarray(metrics["confusion_matrix"], dtype=int),
                    labels=[str(i) for i in range(5)],
                    output_path=fold_dir / "confusion_matrix.png",
                    title=f"{modality} {method} session confusion matrix",
                )
                fold_records.append(
                    {
                        "fold": fold_name,
                        "metrics": metrics,
                    }
                )
                cm_sum += np.asarray(metrics["confusion_matrix"], dtype=int)

            acc = np.asarray([record["metrics"]["accuracy"] for record in fold_records], dtype=np.float32)
            f1 = np.asarray([record["metrics"]["macro_f1"] for record in fold_records], dtype=np.float32)
            precision = np.asarray([record["metrics"]["macro_precision"] for record in fold_records], dtype=np.float32)
            recall = np.asarray([record["metrics"]["macro_recall"] for record in fold_records], dtype=np.float32)
            mean_std = {
                "accuracy": {"mean": float(acc.mean()), "std": float(acc.std(ddof=0))},
                "macro_f1": {"mean": float(f1.mean()), "std": float(f1.std(ddof=0))},
                "macro_precision": {"mean": float(precision.mean()), "std": float(precision.std(ddof=0))},
                "macro_recall": {"mean": float(recall.mean()), "std": float(recall.std(ddof=0))},
            }
            save_confusion_matrix_figure(
                cm_sum,
                labels=[str(i) for i in range(5)],
                output_path=out_root / f"{modality}_{method}_confusion_matrix_sum.png",
                title=f"{modality} {method} confusion matrix (sum)",
            )
            model_output["session_aggregation"][method] = {
                "fold_results": fold_records,
                "mean_std": mean_std,
                "confusion_matrix_sum": cm_sum.tolist(),
            }
            append_markdown_section(
                config["summary_markdown"],
                f"session_agg grouped_cv | {modality} | {method}",
                [
                    f"- python_env: `dl`",
                    f"- model: `{modality}`",
                    f"- method: grouped CV 结果上做 session-level aggregation，聚合方式 `{method}`。",
                    f"- result_session_mean_std: acc={format_metric(mean_std['accuracy']['mean'], mean_std['accuracy']['std'])}, macro-F1={format_metric(mean_std['macro_f1']['mean'], mean_std['macro_f1']['std'])}, precision={format_metric(mean_std['macro_precision']['mean'], mean_std['macro_precision']['std'])}, recall={format_metric(mean_std['macro_recall']['mean'], mean_std['macro_recall']['std'])}",
                ],
            )
        aggregate_summary["per_model"][modality] = model_output

    write_json(out_root / "session_aggregation_summary.json", aggregate_summary)


if __name__ == "__main__":
    main()
