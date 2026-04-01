from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

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
    parser = argparse.ArgumentParser(description="Generate audio frontend search report.")
    parser.add_argument("--config", default="configs/audio_frontend_search.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["output_root"])
    summary = read_json(root / "summary.json")
    experiments = summary["experiments"]
    baseline_name = str(config.get("experiments", [{}])[0].get("name", "")) if config.get("experiments") else ""

    ordered = sorted(
        experiments.values(),
        key=lambda exp: exp["best_session_mean_std"]["macro_f1"]["mean"],
        reverse=True,
    )

    main_rows = [["Frontend", "Window F1", "Best Session Method", "Session F1", "Session Acc"]]
    for exp in ordered:
        main_rows.append(
            [
                exp["display_name"],
                metric_ms(exp["window_level"]["mean_std"]["macro_f1"]),
                exp["best_session_method"],
                metric_ms(exp["best_session_mean_std"]["macro_f1"]),
                metric_ms(exp["best_session_mean_std"]["accuracy"]),
            ]
        )

    top_exp = ordered[0]
    bottom_exp = ordered[-1]
    baseline = experiments[baseline_name] if baseline_name in experiments else ordered[0]
    delta_vs_base = top_exp["best_session_mean_std"]["macro_f1"]["mean"] - baseline["best_session_mean_std"]["macro_f1"]["mean"]

    report = f"""# Audio Frontend Search Report

## 1. Setup

- task: `0 / 2 / 4` 三分类
- split: grouped CV，先按 session 划分再切窗
- window: `5 s`
- model: `audio_only`
- goal: 在不改分类器主干的前提下，比较不同音频前端的判别能力

## 2. Results

{format_table(main_rows)}

![audio_frontend_comparison](audio_frontend_comparison.png)

## 3. Key Findings

- 当前最优前端是 `{top_exp['display_name']}`，session-level macro-F1 为 `{top_exp['best_session_mean_std']['macro_f1']['mean']:.4f}`
- 相比基线 `log-Mel 64 base`，最优前端的提升为 `{delta_vs_base:.4f}`
- 当前最差前端是 `{bottom_exp['display_name']}`，session-level macro-F1 为 `{bottom_exp['best_session_mean_std']['macro_f1']['mean']:.4f}`

## 4. Next Step

- 将最优的 1 到 2 组音频前端回灌到 HCAF-Net，再验证多模态是否同步提升
- 如果 audio-only 提升明显但 HCAF 不升，问题更可能在融合层而不是音频表征
"""
    (root / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
