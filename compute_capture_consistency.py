from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from pathlib import Path
from statistics import mean, median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute post-hoc audio/DAQ duration consistency for all sessions.",
    )
    parser.add_argument("--data-root", default="data", help="Directory containing MMdata_* sessions.")
    parser.add_argument(
        "--output-dir",
        default="summary-MMmodel",
        help="Directory where summary artifacts will be written.",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=0.01,
        help="Warn when abs(rho - 1) exceeds this threshold.",
    )
    return parser.parse_args()


def compute_audio_duration_sec(audio_path: Path) -> tuple[float, float, int]:
    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = float(wav_file.getframerate())
        num_frames = int(wav_file.getnframes())
    if sample_rate <= 0:
        raise ValueError(f"Invalid audio sample rate in {audio_path}")
    return num_frames / sample_rate, sample_rate, num_frames


def infer_daq_sample_rate_from_time(daq_path: Path) -> float | None:
    previous_time: float | None = None
    deltas: list[float] = []
    with daq_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            current_time = float(row["Time (s)"])
            if previous_time is not None:
                delta = current_time - previous_time
                if delta > 0:
                    deltas.append(delta)
            previous_time = current_time
    if not deltas:
        return None
    median_delta = median(deltas)
    if median_delta <= 0:
        return None
    return 1.0 / median_delta


def compute_daq_duration_sec(daq_path: Path, metadata: dict[str, object]) -> tuple[float, float, int]:
    daq_meta = metadata.get("daq", {}) if isinstance(metadata.get("daq"), dict) else {}
    num_samples = daq_meta.get("num_samples")
    sample_rate = daq_meta.get("sample_rate_hz")
    if isinstance(num_samples, (int, float)) and isinstance(sample_rate, (int, float)) and float(sample_rate) > 0:
        return float(num_samples) / float(sample_rate), float(sample_rate), int(num_samples)

    count = 0
    with daq_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for _ in reader:
            count += 1
    inferred_sr = infer_daq_sample_rate_from_time(daq_path)
    if inferred_sr is None or inferred_sr <= 0:
        raise ValueError(f"Unable to infer DAQ sample rate for {daq_path}")
    return count / inferred_sr, inferred_sr, count


def classify_warning(rho: float, warn_threshold: float) -> str | None:
    if math.isnan(rho) or math.isinf(rho):
        return "invalid_ratio"
    if abs(rho - 1.0) > warn_threshold:
        return "duration_mismatch"
    return None


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for session_dir in sorted(data_root.glob("MMdata_*")):
        if not session_dir.is_dir():
            continue
        metadata_path = session_dir / "metadata.json"
        audio_path = session_dir / "audio.wav"
        daq_path = session_dir / "daq.csv"
        if not (metadata_path.exists() and audio_path.exists() and daq_path.exists()):
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        audio_duration_sec, audio_sample_rate_hz, audio_num_frames = compute_audio_duration_sec(audio_path)
        daq_duration_sec, daq_sample_rate_hz, daq_num_samples = compute_daq_duration_sec(daq_path, metadata)
        rho = audio_duration_sec / daq_duration_sec if daq_duration_sec > 0 else float("nan")
        rows.append(
            {
                "session_id": session_dir.name,
                "label": metadata.get("label", ""),
                "stop_reason": metadata.get("stop_reason", ""),
                "audio_duration_sec": audio_duration_sec,
                "daq_duration_sec": daq_duration_sec,
                "rho_audio_to_daq": rho,
                "abs_deviation": abs(rho - 1.0),
                "warning": classify_warning(rho, args.warn_threshold),
                "audio_sample_rate_hz": audio_sample_rate_hz,
                "audio_num_frames": audio_num_frames,
                "daq_sample_rate_hz": daq_sample_rate_hz,
                "daq_num_samples": daq_num_samples,
            }
        )

    if not rows:
        raise FileNotFoundError(f"No complete sessions found under {data_root}")

    valid_rows = [row for row in rows if isinstance(row["rho_audio_to_daq"], float)]
    valid_rhos = [float(row["rho_audio_to_daq"]) for row in valid_rows]
    abs_devs = [float(row["abs_deviation"]) for row in valid_rows]
    flagged_rows = [row for row in rows if row["warning"]]

    summary = {
        "equation": "rho = T_audio / T_daq",
        "warn_threshold": args.warn_threshold,
        "num_sessions": len(rows),
        "num_flagged": len(flagged_rows),
        "rho_mean": mean(valid_rhos),
        "rho_median": median(valid_rhos),
        "rho_min": min(valid_rhos),
        "rho_max": max(valid_rhos),
        "abs_deviation_mean": mean(abs_devs),
        "abs_deviation_max": max(abs_devs),
        "count_abs_deviation_gt_0p001": sum(dev > 0.001 for dev in abs_devs),
        "count_abs_deviation_gt_0p005": sum(dev > 0.005 for dev in abs_devs),
        "count_abs_deviation_gt_0p01": sum(dev > 0.01 for dev in abs_devs),
        "top_deviation_sessions": sorted(rows, key=lambda row: float(row["abs_deviation"]), reverse=True)[:10],
    }

    csv_path = output_dir / "capture_consistency.csv"
    json_path = output_dir / "capture_consistency_summary.json"
    md_path = output_dir / "capture_consistency_report.md"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session_id",
                "label",
                "stop_reason",
                "audio_duration_sec",
                "daq_duration_sec",
                "rho_audio_to_daq",
                "abs_deviation",
                "warning",
                "audio_sample_rate_hz",
                "audio_num_frames",
                "daq_sample_rate_hz",
                "daq_num_samples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    top_rows = sorted(rows, key=lambda row: float(row["abs_deviation"]), reverse=True)[:10]
    table_lines = [
        "| Session | Label | T_audio (s) | T_daq (s) | rho | |rho-1| | Warning |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in top_rows:
        table_lines.append(
            "| "
            f"{row['session_id']} | {row['label']} | {float(row['audio_duration_sec']):.6f} | "
            f"{float(row['daq_duration_sec']):.6f} | {float(row['rho_audio_to_daq']):.6f} | "
            f"{float(row['abs_deviation']):.6f} | {row['warning'] or ''} |"
        )

    report = "\n".join(
        [
            "# 采后音频-DAQ一致性检查",
            "",
            "定义: `rho = T_audio / T_daq`",
            "",
            f"- session 总数: `{len(rows)}`",
            f"- 预警阈值: `|rho - 1| > {args.warn_threshold:.3f}`",
            f"- rho 均值: `{summary['rho_mean']:.6f}`",
            f"- rho 中位数: `{summary['rho_median']:.6f}`",
            f"- rho 范围: `[{summary['rho_min']:.6f}, {summary['rho_max']:.6f}]`",
            f"- 平均偏差 `mean(|rho-1|)`: `{summary['abs_deviation_mean']:.6f}`",
            f"- 最大偏差 `max(|rho-1|)`: `{summary['abs_deviation_max']:.6f}`",
            f"- `|rho-1| > 0.001` 的 session 数: `{summary['count_abs_deviation_gt_0p001']}`",
            f"- `|rho-1| > 0.005` 的 session 数: `{summary['count_abs_deviation_gt_0p005']}`",
            f"- `|rho-1| > 0.01` 的 session 数: `{summary['count_abs_deviation_gt_0p01']}`",
            f"- 触发预警的 session 数: `{len(flagged_rows)}`",
            "",
            "偏差最大的前 10 个 session:",
            "",
            *table_lines,
            "",
            "说明:",
            "- `T_audio` 直接由 `audio.wav` 的 `num_frames / sample_rate` 计算。",
            "- `T_daq` 优先由 `metadata.json` 中的 `num_samples / sample_rate_hz` 计算；若元数据缺失，则回退到 `daq.csv` 行数与时间戳估计。",
            "- 若 `rho` 明显偏离 1，可视为采集链路存在提前终止、回调阻塞或时钟漂移等风险。",
        ]
    )
    md_path.write_text(report, encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
