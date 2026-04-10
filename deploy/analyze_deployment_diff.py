from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mmdl_baseline.utils.io import ensure_dir, read_json, write_json


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)


def _record_key(item: Dict[str, object], decimals: int) -> Tuple[str, float]:
    session_id = str(item["session_id"])
    start_sec = round(float(item["start_sec"]), decimals)
    return session_id, start_sec


def _load_predictions(path: Path) -> List[Dict[str, object]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must be a JSON list.")
    if not payload:
        raise ValueError(f"{path} is empty.")
    return payload


def _extract_vectors(
    items: Sequence[Dict[str, object]],
    field: str,
) -> np.ndarray | None:
    if not all(field in item for item in items):
        return None
    values = np.asarray([item[field] for item in items], dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Field '{field}' must be a 2D array-like per item.")
    return values


def _align_predictions(
    baseline_items: Sequence[Dict[str, object]],
    deployed_items: Sequence[Dict[str, object]],
    decimals: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    baseline_map = {_record_key(item, decimals): item for item in baseline_items}
    deployed_map = {_record_key(item, decimals): item for item in deployed_items}
    common_keys = sorted(set(baseline_map) & set(deployed_map))
    if not common_keys:
        raise ValueError("No overlapping (session_id, start_sec) windows were found.")
    aligned_baseline = [baseline_map[key] for key in common_keys]
    aligned_deployed = [deployed_map[key] for key in common_keys]
    return aligned_baseline, aligned_deployed


def _percentiles(values: np.ndarray, probs: Sequence[float]) -> Dict[str, float]:
    return {f"p{int(p)}": float(np.percentile(values, p)) for p in probs}


def _plot_histogram(values: np.ndarray, output_path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(values, bins=60, color="#4e79a7", alpha=0.9, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_cdf(values: np.ndarray, output_path: Path, title: str, xlabel: str) -> None:
    sorted_values = np.sort(values)
    y = np.linspace(0.0, 1.0, len(sorted_values), endpoint=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(sorted_values, y, color="#e15759", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_top1_shift(
    baseline_probs: np.ndarray,
    deployed_probs: np.ndarray,
    output_path: Path,
) -> Dict[str, float]:
    baseline_top1 = baseline_probs.max(axis=1)
    deployed_top1 = deployed_probs.max(axis=1)
    delta = deployed_top1 - baseline_top1
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(delta, bins=50, color="#59a14f", alpha=0.9, edgecolor="white")
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title("Top-1 probability shift")
    ax.set_xlabel("deployed top1 prob - baseline top1 prob")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    abs_delta = np.abs(delta)
    return {
        "mean_signed_shift": float(delta.mean()),
        "mean_abs_shift": float(abs_delta.mean()),
        "max_abs_shift": float(abs_delta.max()),
        **_percentiles(abs_delta, [50, 95, 99]),
    }


def _summarize_diff(name: str, diff: np.ndarray) -> Dict[str, object]:
    abs_diff = np.abs(diff)
    flat_signed = diff.reshape(-1)
    flat_abs = abs_diff.reshape(-1)
    per_window_linf = abs_diff.max(axis=1)
    per_window_l1 = abs_diff.mean(axis=1)
    return {
        "name": name,
        "num_windows": int(diff.shape[0]),
        "num_classes": int(diff.shape[1]),
        "mean_signed_diff": float(flat_signed.mean()),
        "mean_abs_diff": float(flat_abs.mean()),
        "std_abs_diff": float(flat_abs.std()),
        "max_abs_diff": float(flat_abs.max()),
        "per_window_max_abs_mean": float(per_window_linf.mean()),
        "per_window_max_abs_max": float(per_window_linf.max()),
        "per_window_mean_abs_mean": float(per_window_l1.mean()),
        **_percentiles(flat_abs, [50, 90, 95, 99, 100]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline and deployed prediction JSON files and plot logit/probability differences."
    )
    parser.add_argument("--baseline", required=True, help="Reference prediction JSON path.")
    parser.add_argument("--deployed", required=True, help="Deployed prediction JSON path.")
    parser.add_argument("--output-dir", required=True, help="Directory to write plots and summary.")
    parser.add_argument(
        "--key-start-sec-decimals",
        type=int,
        default=3,
        help="Decimals used when aligning start_sec between files.",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    deployed_path = Path(args.deployed)
    output_dir = ensure_dir(args.output_dir)

    baseline_items = _load_predictions(baseline_path)
    deployed_items = _load_predictions(deployed_path)
    baseline_items, deployed_items = _align_predictions(
        baseline_items,
        deployed_items,
        decimals=args.key_start_sec_decimals,
    )

    baseline_logits = _extract_vectors(baseline_items, "logits")
    deployed_logits = _extract_vectors(deployed_items, "logits")
    baseline_probs = _extract_vectors(baseline_items, "probabilities")
    deployed_probs = _extract_vectors(deployed_items, "probabilities")

    if baseline_logits is not None and baseline_probs is None:
        baseline_probs = _softmax(baseline_logits)
    if deployed_logits is not None and deployed_probs is None:
        deployed_probs = _softmax(deployed_logits)

    if baseline_probs is None or deployed_probs is None:
        raise ValueError("At least one of logits/probabilities must be available in both files.")

    if baseline_probs.shape != deployed_probs.shape:
        raise ValueError("Probability shapes do not match between baseline and deployed files.")

    summary: Dict[str, object] = {
        "baseline_path": str(baseline_path),
        "deployed_path": str(deployed_path),
        "aligned_num_windows": len(baseline_items),
        "alignment_key": ["session_id", f"round(start_sec, {args.key_start_sec_decimals})"],
    }

    prob_diff = deployed_probs - baseline_probs
    prob_stats = _summarize_diff("probabilities", prob_diff)
    _plot_histogram(
        np.abs(prob_diff).reshape(-1),
        output_dir / "probability_abs_diff_hist.png",
        "Probability absolute difference histogram",
        "|p_deployed - p_baseline|",
    )
    _plot_cdf(
        np.abs(prob_diff).reshape(-1),
        output_dir / "probability_abs_diff_cdf.png",
        "Probability absolute difference CDF",
        "|p_deployed - p_baseline|",
    )
    prob_top1_stats = _plot_top1_shift(
        baseline_probs,
        deployed_probs,
        output_dir / "top1_probability_shift_hist.png",
    )
    summary["probability_diff"] = prob_stats
    summary["top1_probability_shift"] = prob_top1_stats

    baseline_pred = baseline_probs.argmax(axis=1)
    deployed_pred = deployed_probs.argmax(axis=1)
    summary["top1_consistency"] = {
        "same_top1_ratio": float((baseline_pred == deployed_pred).mean()),
        "num_top1_changed": int((baseline_pred != deployed_pred).sum()),
    }

    if baseline_logits is not None and deployed_logits is not None:
        if baseline_logits.shape != deployed_logits.shape:
            raise ValueError("Logit shapes do not match between baseline and deployed files.")
        logit_diff = deployed_logits - baseline_logits
        logit_stats = _summarize_diff("logits", logit_diff)
        _plot_histogram(
            np.abs(logit_diff).reshape(-1),
            output_dir / "logit_abs_diff_hist.png",
            "Logit absolute difference histogram",
            "|logit_deployed - logit_baseline|",
        )
        _plot_cdf(
            np.abs(logit_diff).reshape(-1),
            output_dir / "logit_abs_diff_cdf.png",
            "Logit absolute difference CDF",
            "|logit_deployed - logit_baseline|",
        )
        summary["logit_diff"] = logit_stats
    else:
        summary["logit_diff"] = {
            "available": False,
            "note": "Skipped because at least one file does not contain logits.",
        }

    write_json(output_dir / "summary.json", summary)
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
