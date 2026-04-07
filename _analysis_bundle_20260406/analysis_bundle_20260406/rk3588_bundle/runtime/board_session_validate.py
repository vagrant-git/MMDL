from __future__ import annotations

import argparse
import json
import math
import resource
import statistics
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from runtime_infer_onnx import RuntimeOnnxInfer, resample_1d_np

LABEL_MAP = {
    "no secretion": "0",
    "none": "0",
    "0ml": "0",
    "0 ml": "0",
    "2ml": "2",
    "2 ml": "2",
    "4ml": "4",
    "4 ml": "4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate deployment behavior using board_test_sessions.")
    parser.add_argument("--sessions-root", default="/home/firefly/board_test_sessions")
    parser.add_argument("--onnx-model", default="/home/firefly/rk3588_bundle/models/hcaf_pcen_dualxattn.onnx")
    parser.add_argument("--config", default="/home/firefly/rk3588_bundle/configs/final_model_unified_evidence.yaml")
    parser.add_argument("--smoke-hop-sec", type=float, default=5.0)
    parser.add_argument("--benchmark-hop-sec", type=float, default=1.0)
    parser.add_argument("--simulate-audio-input-rate", type=int, default=44100)
    parser.add_argument("--simulate-sensor-rate", type=float, default=10.0)
    parser.add_argument("--max-benchmark-windows-per-session", type=int, default=None)
    parser.add_argument("--output-json", default="/home/firefly/board_test_sessions/validation_summary.json")
    return parser.parse_args()


def print_status(message: str) -> None:
    print(message, flush=True)


def canonical_label(label_text: str) -> str:
    normalized = label_text.strip().lower()
    if normalized not in LABEL_MAP:
        raise ValueError(f"Unsupported label text: {label_text}")
    return LABEL_MAP[normalized]


def read_audio_int16(wav_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16)
    if channels != 1:
        audio = audio.reshape(-1, channels)[:, 0].copy()
    return audio, int(sample_rate)


def read_daq_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1, usecols=(1, 2), dtype=np.float32)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data[:, 0].astype(np.float32, copy=False), data[:, 1].astype(np.float32, copy=False)


def resample_audio_int16(audio: np.ndarray, input_rate: int, target_rate: int, target_length: int) -> np.ndarray:
    if input_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive.")
    if audio.shape[0] < 2:
        raise ValueError(f"Need at least 2 audio samples to resample, got {audio.shape[0]}")
    if input_rate == target_rate and audio.shape[0] == target_length:
        return audio.astype(np.int16, copy=False)
    src_pos = np.linspace(0.0, 1.0, num=audio.shape[0], dtype=np.float32)
    dst_pos = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    resampled = np.interp(dst_pos, src_pos, audio.astype(np.float32, copy=False))
    return np.clip(np.round(resampled), -32768, 32767).astype(np.int16)


def stats_from_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def iter_window_starts(duration_sec: float, window_sec: float, hop_sec: float, limit: int | None = None) -> list[float]:
    if duration_sec < window_sec:
        return []
    count = int(math.floor((duration_sec - window_sec) / hop_sec)) + 1
    if limit is not None:
        count = min(count, limit)
    return [idx * hop_sec for idx in range(count)]


def extract_fixed_window(signal_values: np.ndarray, sample_rate: float, start_sec: float, window_sec: float) -> np.ndarray:
    start = int(round(start_sec * sample_rate))
    length = max(2, int(round(window_sec * sample_rate)))
    end = start + length
    if end > signal_values.shape[0]:
        end = signal_values.shape[0]
        start = max(0, end - length)
    return signal_values[start:end].copy()


def evaluate_consistency_session(
    session_dir: Path,
    infer_engine: RuntimeOnnxInfer,
    hop_sec: float,
    simulate_audio_input_rate: int,
    simulate_sensor_rate: float,
) -> dict[str, Any]:
    metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
    expected_label = canonical_label(str(metadata["label"]))
    audio, audio_rate = read_audio_int16(session_dir / "audio.wav")
    pressure, flow = read_daq_csv(session_dir / "daq.csv")
    daq_rate = float(metadata["daq"]["sample_rate_hz"])

    duration_sec = min(audio.shape[0] / float(audio_rate), pressure.shape[0] / daq_rate, flow.shape[0] / daq_rate)
    starts = iter_window_starts(duration_sec, infer_engine.window_sec, hop_sec)
    reference_preds: list[str] = []
    board_preds: list[str] = []
    logit_diffs: list[float] = []
    prob_diffs: list[float] = []

    for start_sec in starts:
        audio_window = extract_fixed_window(audio, audio_rate, start_sec, infer_engine.window_sec)
        pressure_window = extract_fixed_window(pressure, daq_rate, start_sec, infer_engine.window_sec)
        flow_window = extract_fixed_window(flow, daq_rate, start_sec, infer_engine.window_sec)

        audio_reference = resample_audio_int16(
            audio_window,
            audio_rate,
            infer_engine.audio_rate,
            infer_engine.audio_samples,
        )
        reference_result = infer_engine.infer(audio_reference, pressure_window, flow_window)

        audio_captured = resample_audio_int16(
            audio_window,
            audio_rate,
            simulate_audio_input_rate,
            max(2, int(round(simulate_audio_input_rate * infer_engine.window_sec))),
        )
        audio_board = resample_audio_int16(
            audio_captured,
            simulate_audio_input_rate,
            infer_engine.audio_rate,
            infer_engine.audio_samples,
        )
        target_sensor_samples = max(2, int(round(simulate_sensor_rate * infer_engine.window_sec)))
        pressure_board_source = resample_1d_np(pressure_window, target_sensor_samples)
        flow_board_source = resample_1d_np(flow_window, target_sensor_samples)
        board_result = infer_engine.infer(audio_board, pressure_board_source, flow_board_source)

        reference_preds.append(str(reference_result["predicted_label"]))
        board_preds.append(str(board_result["predicted_label"]))
        ref_logits = np.asarray(reference_result["logits"], dtype=np.float32)
        board_logits = np.asarray(board_result["logits"], dtype=np.float32)
        ref_probs = np.asarray(reference_result["probabilities"], dtype=np.float32)
        board_probs = np.asarray(board_result["probabilities"], dtype=np.float32)
        logit_diffs.append(float(np.max(np.abs(ref_logits - board_logits))))
        prob_diffs.append(float(np.max(np.abs(ref_probs - board_probs))))

    reference_majority = Counter(reference_preds).most_common(1)[0][0] if reference_preds else None
    board_majority = Counter(board_preds).most_common(1)[0][0] if board_preds else None
    agreement = sum(1 for ref_pred, board_pred in zip(reference_preds, board_preds) if ref_pred == board_pred)

    return {
        "session": session_dir.name,
        "expected_label": expected_label,
        "windows": len(starts),
        "reference_majority_label": reference_majority,
        "board_majority_label": board_majority,
        "reference_matches_expected": reference_majority == expected_label,
        "board_matches_expected": board_majority == expected_label,
        "window_prediction_agreement": agreement,
        "window_prediction_agreement_rate": (agreement / len(starts)) if starts else 0.0,
        "max_abs_logit_diff": max(logit_diffs) if logit_diffs else 0.0,
        "mean_abs_logit_diff": float(np.mean(logit_diffs)) if logit_diffs else 0.0,
        "max_abs_prob_diff": max(prob_diffs) if prob_diffs else 0.0,
        "mean_abs_prob_diff": float(np.mean(prob_diffs)) if prob_diffs else 0.0,
    }


def evaluate_benchmark_session(
    session_dir: Path,
    infer_engine: RuntimeOnnxInfer,
    hop_sec: float,
    simulate_audio_input_rate: int,
    simulate_sensor_rate: float,
    max_windows: int | None,
) -> dict[str, Any]:
    metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
    expected_label = canonical_label(str(metadata["label"]))
    audio, audio_rate = read_audio_int16(session_dir / "audio.wav")
    pressure, flow = read_daq_csv(session_dir / "daq.csv")
    daq_rate = float(metadata["daq"]["sample_rate_hz"])

    duration_sec = min(audio.shape[0] / float(audio_rate), pressure.shape[0] / daq_rate, flow.shape[0] / daq_rate)
    starts = iter_window_starts(duration_sec, infer_engine.window_sec, hop_sec, limit=max_windows)
    latencies_ms: list[float] = []
    predictions: list[str] = []
    target_sensor_samples = max(2, int(round(simulate_sensor_rate * infer_engine.window_sec)))
    target_audio_capture_samples = max(2, int(round(simulate_audio_input_rate * infer_engine.window_sec)))

    for index, start_sec in enumerate(starts, start=1):
        audio_window = extract_fixed_window(audio, audio_rate, start_sec, infer_engine.window_sec)
        pressure_window = extract_fixed_window(pressure, daq_rate, start_sec, infer_engine.window_sec)
        flow_window = extract_fixed_window(flow, daq_rate, start_sec, infer_engine.window_sec)

        start_ts = time.perf_counter()
        audio_captured = resample_audio_int16(
            audio_window,
            audio_rate,
            simulate_audio_input_rate,
            target_audio_capture_samples,
        )
        audio_board = resample_audio_int16(
            audio_captured,
            simulate_audio_input_rate,
            infer_engine.audio_rate,
            infer_engine.audio_samples,
        )
        pressure_board_source = resample_1d_np(pressure_window, target_sensor_samples)
        flow_board_source = resample_1d_np(flow_window, target_sensor_samples)
        result = infer_engine.infer(audio_board, pressure_board_source, flow_board_source)
        latencies_ms.append((time.perf_counter() - start_ts) * 1000.0)
        predictions.append(str(result["predicted_label"]))

        if index % 100 == 0:
            print_status(
                f"[benchmark] {session_dir.name}: processed {index}/{len(starts)} windows, "
                f"latest_latency_ms={latencies_ms[-1]:.2f}"
            )

    majority_label = Counter(predictions).most_common(1)[0][0] if predictions else None
    return {
        "session": session_dir.name,
        "expected_label": expected_label,
        "windows": len(starts),
        "majority_label": majority_label,
        "matches_expected": majority_label == expected_label,
        "latency_ms": stats_from_values(latencies_ms),
    }


def main() -> int:
    args = parse_args()
    infer_engine = RuntimeOnnxInfer(args.onnx_model, args.config)
    sessions_root = Path(args.sessions_root)
    smoke_sessions = sorted((sessions_root / "smoke").glob("MMdata_*"))
    benchmark_sessions = sorted((sessions_root / "benchmark").glob("MMdata_*"))

    print_status(
        "[config] "
        f"preprocess_backend={infer_engine.preprocess_backend} "
        f"audio_rate={infer_engine.audio_rate} "
        f"sensor_rate={infer_engine.sensor_rate} "
        f"window_sec={infer_engine.window_sec}"
    )
    print_status(
        "[config] "
        f"simulate_audio_input_rate={args.simulate_audio_input_rate} "
        f"simulate_sensor_rate={args.simulate_sensor_rate}"
    )

    consistency_results: list[dict[str, Any]] = []
    for session_dir in smoke_sessions:
        print_status(f"[consistency] evaluating {session_dir.name}")
        consistency_results.append(
            evaluate_consistency_session(
                session_dir,
                infer_engine,
                args.smoke_hop_sec,
                args.simulate_audio_input_rate,
                args.simulate_sensor_rate,
            )
        )

    consistency_windows = sum(item["windows"] for item in consistency_results)
    consistency_agreement = sum(item["window_prediction_agreement"] for item in consistency_results)
    consistency_summary = {
        "sessions": consistency_results,
        "total_windows": consistency_windows,
        "all_reference_match_expected": all(item["reference_matches_expected"] for item in consistency_results),
        "all_board_match_expected": all(item["board_matches_expected"] for item in consistency_results),
        "window_prediction_agreement_rate": (
            consistency_agreement / consistency_windows if consistency_windows else 0.0
        ),
        "max_abs_logit_diff": max((item["max_abs_logit_diff"] for item in consistency_results), default=0.0),
        "max_abs_prob_diff": max((item["max_abs_prob_diff"] for item in consistency_results), default=0.0),
    }

    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    wall_start = time.perf_counter()
    benchmark_results: list[dict[str, Any]] = []
    for session_dir in benchmark_sessions:
        print_status(f"[benchmark] evaluating {session_dir.name}")
        benchmark_results.append(
            evaluate_benchmark_session(
                session_dir,
                infer_engine,
                args.benchmark_hop_sec,
                args.simulate_audio_input_rate,
                args.simulate_sensor_rate,
                args.max_benchmark_windows_per_session,
            )
        )
    wall_sec = time.perf_counter() - wall_start
    usage_end = resource.getrusage(resource.RUSAGE_SELF)

    all_latencies = [item["latency_ms"]["mean"] for item in benchmark_results if item["latency_ms"]["count"] > 0]
    latency_values_p95 = [item["latency_ms"]["p95"] for item in benchmark_results if item["latency_ms"]["count"] > 0]
    total_benchmark_windows = sum(item["windows"] for item in benchmark_results)
    benchmark_summary = {
        "sessions": benchmark_results,
        "total_windows": total_benchmark_windows,
        "all_match_expected": all(item["matches_expected"] for item in benchmark_results),
        "mean_session_latency_ms": float(np.mean(all_latencies)) if all_latencies else 0.0,
        "max_session_p95_latency_ms": max(latency_values_p95) if latency_values_p95 else 0.0,
        "wall_time_sec": wall_sec,
        "windows_per_sec": (total_benchmark_windows / wall_sec) if wall_sec > 0 else 0.0,
        "realtime_factor_vs_1s_hop": (
            (max(latency_values_p95) / 1000.0) if latency_values_p95 else 0.0
        ),
        "max_rss_mb": float(usage_end.ru_maxrss) / 1024.0,
        "cpu_time_sec": (usage_end.ru_utime + usage_end.ru_stime) - (usage_start.ru_utime + usage_start.ru_stime),
    }

    summary = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {
            "onnx_model": str(Path(args.onnx_model)),
            "config": str(Path(args.config)),
            "preprocess_backend": infer_engine.preprocess_backend,
        },
        "simulation": {
            "audio_input_rate_hz": args.simulate_audio_input_rate,
            "sensor_input_rate_hz": args.simulate_sensor_rate,
            "smoke_hop_sec": args.smoke_hop_sec,
            "benchmark_hop_sec": args.benchmark_hop_sec,
            "max_benchmark_windows_per_session": args.max_benchmark_windows_per_session,
        },
        "consistency": consistency_summary,
        "benchmark": benchmark_summary,
    }

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print_status(f"[done] wrote summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
