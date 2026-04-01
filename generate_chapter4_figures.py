from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


ROOT = Path("summary-MMmodel")
FIG_DIR = ROOT / "figures"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _metric_map(path: Path) -> Dict[str, Dict[str, float | str]]:
    rows = _read_csv(path)
    out: Dict[str, Dict[str, float | str]] = {}
    for row in rows:
        out[row["experiment"]] = {
            key: (float(value) if key not in {"experiment", "display_name", "group", "modality", "best_session_method"} else value)
            for key, value in row.items()
        }
    return out


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_primary_performance() -> Dict[str, float]:
    data = _metric_map(ROOT / "pq_vs_multimodal_check" / "overall_results.csv")
    order = ["pressure_flow_5s", "hcaf_confgate_residual_5s"]
    labels = ["PQ-only", "Prev HCAF best"]
    means = [float(data[name]["session_macro_f1_mean"]) for name in order]
    stds = [float(data[name]["session_macro_f1_std"]) for name in order]

    current = _metric_map(ROOT / "hcaf_confgate_improve_search" / "overall_results.csv")["hcaf_confgate_residual_pcen96hp80_5s"]
    labels.append("Current best\nPCEN96+HP80")
    means.append(float(current["session_macro_f1_mean"]))
    stds.append(float(current["session_macro_f1_std"]))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    colors = ["#4e79a7", "#59a14f", "#e15759"]
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors, width=0.62)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", capsize=5, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Session-level macro-F1")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Primary Performance Progression")
    for idx, value in enumerate(means):
        ax.text(idx, value + 0.02, f"{value:.4f}", ha="center", va="bottom")
    _style_ax(ax)
    _save(fig, FIG_DIR / "primary_performance_progression.png")
    return {
        "pq_only": means[0],
        "prev_best": means[1],
        "current_best": means[2],
        "gain_over_prev": means[2] - means[1],
        "gain_over_pq": means[2] - means[0],
    }


def plot_frontend_improvement() -> None:
    data = _metric_map(ROOT / "hcaf_confgate_improve_search" / "overall_results.csv")
    order = [
        "hcaf_confgate_residual_base_5s",
        "hcaf_confgate_residual_preemphasis16k_5s",
        "hcaf_confgate_residual_preemphasis12k_5s",
        "hcaf_confgate_residual_pcen96hp80_5s",
    ]
    labels = ["Base", "Preemph 16k", "Preemph 12k", "PCEN96+HP80"]
    window_means = [float(data[name]["window_macro_f1_mean"]) for name in order]
    session_means = [float(data[name]["session_macro_f1_mean"]) for name in order]
    session_stds = [float(data[name]["session_macro_f1_std"]) for name in order]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=False)
    x = np.arange(len(labels))
    axes[0].bar(x, window_means, color="#4e79a7")
    axes[0].set_title("Window-level")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_ylim(0.0, 1.0)
    _style_ax(axes[0])

    axes[1].bar(x, session_means, color="#e15759")
    axes[1].errorbar(x, session_means, yerr=session_stds, fmt="none", ecolor="black", capsize=4, linewidth=1.1)
    axes[1].set_title("Session-level")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_ylim(0.0, 1.0)
    _style_ax(axes[1])
    fig.suptitle("Same-Split Audio Frontend Comparison")
    _save(fig, FIG_DIR / "frontend_same_split_comparison.png")


def plot_filter_comparison() -> None:
    data = _metric_map(ROOT / "hcaf_confgate_filter_lowpass300" / "overall_results.csv")
    ref = _metric_map(ROOT / "hcaf_confgate_improve_search" / "overall_results.csv")["hcaf_confgate_residual_pcen96hp80_5s"]
    labels = ["HP80", "LP300", "BP80-300"]
    means = [
        float(ref["session_macro_f1_mean"]),
        float(data["hcaf_confgate_residual_pcen96lp300_5s"]["session_macro_f1_mean"]),
        float(data["hcaf_confgate_residual_pcen96hp80lp300_5s"]["session_macro_f1_mean"]),
    ]
    stds = [
        float(ref["session_macro_f1_std"]),
        float(data["hcaf_confgate_residual_pcen96lp300_5s"]["session_macro_f1_std"]),
        float(data["hcaf_confgate_residual_pcen96hp80lp300_5s"]["session_macro_f1_std"]),
    ]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    x = np.arange(len(labels))
    ax.bar(x, means, color=["#59a14f", "#f28e2b", "#9c755f"])
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", capsize=4, linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Session-level macro-F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Filter Strategy Comparison")
    _style_ax(ax)
    _save(fig, FIG_DIR / "filter_strategy_comparison.png")


def plot_fusion_ablation() -> None:
    data = _metric_map(ROOT / "hcaf_fusion_gate_followup" / "overall_results.csv")
    order = [
        "hcaf_legacy_sharednorm_5s",
        "hcaf_normfix_5s",
        "hcaf_confgate_5s",
        "hcaf_confgate_residual_5s",
    ]
    labels = ["Legacy\nshared norm", "Norm fix", "Conf-gate", "Conf-gate\n+ residual"]
    window = [float(data[name]["window_macro_f1_mean"]) for name in order]
    session = [float(data[name]["session_macro_f1_mean"]) for name in order]
    session_std = [float(data[name]["session_macro_f1_std"]) for name in order]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    x = np.arange(len(labels))
    axes[0].plot(x, window, marker="o", linewidth=2.2, color="#4e79a7")
    axes[0].set_title("Window-level effect")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Macro-F1")
    _style_ax(axes[0])

    axes[1].bar(x, session, color="#e15759")
    axes[1].errorbar(x, session, yerr=session_std, fmt="none", ecolor="black", capsize=4, linewidth=1.1)
    axes[1].set_title("Session-level effect")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.0, 1.0)
    _style_ax(axes[1])
    fig.suptitle("Fusion Mechanism Ablation")
    _save(fig, FIG_DIR / "fusion_mechanism_ablation.png")


def plot_missing_modalities() -> None:
    data = _metric_map(ROOT / "hcaf_missing_modalities" / "overall_results.csv")
    order = [
        "hcaf_confgate_residual_full_5s",
        "hcaf_confgate_residual_minus_audio_5s",
        "hcaf_confgate_residual_minus_pressure_5s",
        "hcaf_confgate_residual_minus_flow_5s",
        "hcaf_confgate_residual_audio_only_5s",
    ]
    labels = ["Full", "Missing\nAudio", "Missing\nPressure", "Missing\nFlow", "Audio\nOnly"]
    means = [float(data[name]["session_macro_f1_mean"]) for name in order]
    stds = [float(data[name]["session_macro_f1_std"]) for name in order]

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means, color=["#4e79a7", "#59a14f", "#76b7b2", "#e15759", "#f28e2b"])
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", capsize=4, linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Session-level macro-F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Missing-Modality Robustness")
    _style_ax(ax)
    _save(fig, FIG_DIR / "missing_modality_robustness.png")


def plot_window_tradeoff() -> None:
    first = _metric_map(ROOT / "hcaf_confgate_window_lengths" / "overall_results.csv")
    second = _metric_map(ROOT / "hcaf_confgate_window_lengths_6_8_15" / "overall_results.csv")
    merged = {**first, **second}
    order = [
        ("hcaf_confgate_residual_5s", 5),
        ("hcaf_confgate_residual_6s", 6),
        ("hcaf_confgate_residual_8s", 8),
        ("hcaf_confgate_residual_10s", 10),
        ("hcaf_confgate_residual_15s", 15),
        ("hcaf_confgate_residual_20s", 20),
    ]
    x = [window for _, window in order]
    session = [float(merged[name]["session_macro_f1_mean"]) for name, _ in order]
    stds = [float(merged[name]["session_macro_f1_std"]) for name, _ in order]
    window = [float(merged[name]["window_macro_f1_mean"]) for name, _ in order]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(x, session, marker="o", linewidth=2.2, color="#e15759", label="Session-level")
    ax.fill_between(x, np.array(session) - np.array(stds), np.array(session) + np.array(stds), color="#e15759", alpha=0.16)
    ax.plot(x, window, marker="s", linewidth=2.0, color="#4e79a7", label="Window-level")
    ax.set_xlabel("Window length (s)")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Window-Length Tradeoff")
    ax.legend()
    _style_ax(ax)
    _save(fig, FIG_DIR / "window_length_tradeoff.png")


def plot_interpretability_summary() -> None:
    import json

    summary = json.loads((ROOT / "hcaf_confgate_interpretability" / "summary.json").read_text(encoding="utf-8"))
    gate_by_class = summary["gate_by_class"]
    labels = list(gate_by_class.keys())
    means = [float(gate_by_class[label]["mean_audio_gate"]) for label in labels]
    stds = [float(gate_by_class[label]["std_audio_gate"]) for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    x = np.arange(len(labels))
    axes[0].bar(x, means, color=["#4e79a7", "#59a14f", "#e15759"])
    axes[0].errorbar(x, means, yerr=stds, fmt="none", ecolor="black", capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Audio gate by class")
    axes[0].set_ylabel("Mean audio gate weight")
    _style_ax(axes[0])

    boundary = float(summary["boundary_stats"]["boundary_error_rate"])
    middle = float(summary["boundary_stats"]["middle_error_rate"])
    axes[1].bar(["Boundary\n(0-20%,80-100%)", "Middle\n(20-80%)"], [boundary, middle], color=["#f28e2b", "#76b7b2"])
    axes[1].set_ylim(0.0, max(boundary, middle) * 1.5)
    axes[1].set_title("Error rate by position")
    axes[1].set_ylabel("Window error rate")
    _style_ax(axes[1])
    fig.suptitle("Interpretability and Error Summary")
    _save(fig, FIG_DIR / "interpretability_error_summary.png")


def plot_model_diagram() -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.4,
            edgecolor="#2f2f2f",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, wrap=True)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.6, color="#444"))

    box(0.4, 4.2, 2.0, 1.0, "Audio\nPCEN96 + HP80", "#d6e9ff")
    box(0.4, 2.5, 2.0, 1.0, "Pressure\n1D waveform", "#d9f2e6")
    box(0.4, 0.8, 2.0, 1.0, "Flow\n1D waveform", "#d9f2e6")
    box(3.0, 4.2, 2.2, 1.0, "AudioTokenEncoder\n2D CNN -> 12 tokens", "#bcd7ff")
    box(3.0, 2.5, 2.2, 1.0, "SensorTemporalEncoder\n1D CNN + TCN", "#bfe6d1")
    box(3.0, 0.8, 2.2, 1.0, "SensorTemporalEncoder\n1D CNN + TCN", "#bfe6d1")
    box(5.9, 1.65, 2.2, 1.5, "Pressure-Flow\ncross-attention\n+ token gate", "#f9e2b8")
    box(8.7, 1.65, 2.3, 1.5, "Audio-Sensor\ncross-attention\n+ self-attention", "#f6d3e0")
    box(11.7, 3.45, 1.8, 1.0, "Audio expert", "#dde5f7")
    box(11.7, 2.1, 1.8, 1.0, "Sensor expert", "#dde5f7")
    box(11.5, 0.55, 2.2, 1.1, "Confidence-aware\nreliability gate", "#fbe3b0")
    box(8.8, 4.2, 1.8, 0.95, "Expert residual\nscale = 0.3", "#e6d8f5")
    box(8.7, 4.95, 2.3, 0.7, "Fused repr classifier", "#d7f0d0")
    box(11.7, 4.85, 1.7, 0.8, "Final logits", "#ffd7c2")

    arrow(2.4, 4.7, 3.0, 4.7)
    arrow(2.4, 3.0, 3.0, 3.0)
    arrow(2.4, 1.3, 3.0, 1.3)
    arrow(5.2, 3.0, 5.9, 2.6)
    arrow(5.2, 1.3, 5.9, 2.2)
    arrow(5.2, 4.7, 8.7, 2.8)
    arrow(8.1, 2.4, 8.7, 2.4)
    arrow(11.0, 2.4, 11.5, 1.1)
    arrow(10.95, 2.9, 11.7, 2.6)
    arrow(10.95, 2.9, 11.7, 3.95)
    arrow(12.6, 1.65, 12.6, 4.95)
    arrow(10.6, 5.3, 11.7, 5.25)
    arrow(13.4, 5.25, 13.4, 5.25)

    ax.text(9.9, 3.62, "audio repr", fontsize=9, ha="center")
    ax.text(9.9, 2.22, "sensor repr", fontsize=9, ha="center")
    ax.text(12.6, 1.95, "weights from repr + confidence", fontsize=8, ha="center")
    ax.text(9.65, 5.55, "main classifier", fontsize=8, ha="center")
    ax.set_title("Architecture Sketch of hcaf_confgate_residual_pcen96hp80_5s", fontsize=14, pad=12)
    _save(fig, FIG_DIR / "hcaf_current_best_architecture.png")


def main() -> None:
    _ensure_dir(FIG_DIR)
    primary = plot_primary_performance()
    plot_frontend_improvement()
    plot_filter_comparison()
    plot_fusion_ablation()
    plot_missing_modalities()
    plot_window_tradeoff()
    plot_interpretability_summary()
    plot_model_diagram()
    (FIG_DIR / "figure_summary.txt").write_text(
        "\n".join(
            [
                "Generated chapter-4 figures.",
                f"Current best vs previous HCAF gain: {primary['gain_over_prev']:.4f}",
                f"Current best vs PQ-only gain: {primary['gain_over_pq']:.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
