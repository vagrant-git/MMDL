from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--backend", choices=("librosa", "torch"), required=True)
    parser.add_argument("--normalization-mode", choices=("window", "rolling"), required=True)
    parser.add_argument("--normalization-buffer-sec", type=float, default=30.0)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


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


def iter_window_starts(duration_sec: float, window_sec: float, hop_sec: float) -> list[float]:
    if duration_sec < window_sec:
        return []
    count = int(math.floor((duration_sec - window_sec) / hop_sec)) + 1
    return [idx * hop_sec for idx in range(count)]


def extract_fixed_window(signal_values: np.ndarray, sample_rate: float, start_sec: float, duration_sec: float) -> np.ndarray:
    start = int(round(start_sec * sample_rate))
    length = max(2, int(round(duration_sec * sample_rate)))
    end = start + length
    if end > signal_values.shape[0]:
        end = signal_values.shape[0]
        start = max(0, end - length)
    return signal_values[start:end].copy()


def normalization_stats_from_values(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 1.0}
    values = values.astype(np.float32, copy=False)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-6:
        std = 1.0
    return {"mean": mean, "std": std}


def stats(values: list[float]) -> dict[str, float]:
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


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root)
    if args.backend == "torch":
        sys.path.insert(0, "/tmp/torch_py39")
    sys.path.insert(0, str(repo_root / "rk3588_bundle" / "runtime"))

    from runtime_infer_onnx import RuntimeOnnxInfer, resample_1d_np

    smoke_root = repo_root / "board_test_sessions" / "smoke"
    onnx_model = repo_root / "rk3588_bundle" / "models" / "hcaf_pcen_dualxattn.onnx"
    config_path = repo_root / "rk3588_bundle" / "configs" / "final_model_unified_evidence.yaml"
    pc_csv = repo_root / "board_test_sessions" / "smoke_diff_artifacts" / "PC_probabilities_by_window.csv"

    simulate_audio_input_rate = 44100
    simulate_sensor_rate = 10.0
    hop_sec = 5.0

    infer = RuntimeOnnxInfer(onnx_model, config_path)
    class_names = infer.class_names[:] if infer.class_names else ["0", "2", "4"]

    rows: list[dict[str, object]] = []
    normalization_buffer_sec = max(infer.window_sec, args.normalization_buffer_sec)

    for session_dir in sorted(smoke_root.glob("MMdata_*")):
        meta = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
        expected_label = canonical_label(str(meta["label"]))
        audio, audio_rate = read_audio_int16(session_dir / "audio.wav")
        pressure, flow = read_daq_csv(session_dir / "daq.csv")
        daq_rate = float(meta["daq"]["sample_rate_hz"])
        duration_sec = min(audio.shape[0] / float(audio_rate), pressure.shape[0] / daq_rate, flow.shape[0] / daq_rate)
        starts = iter_window_starts(duration_sec, infer.window_sec, hop_sec)

        for window_index, start_sec in enumerate(starts):
            audio_window = extract_fixed_window(audio, audio_rate, start_sec, infer.window_sec)
            pressure_window_100 = extract_fixed_window(pressure, daq_rate, start_sec, infer.window_sec)
            flow_window_100 = extract_fixed_window(flow, daq_rate, start_sec, infer.window_sec)

            audio_captured = resample_audio_int16(
                audio_window,
                audio_rate,
                simulate_audio_input_rate,
                max(2, int(round(simulate_audio_input_rate * infer.window_sec))),
            )
            audio_board = resample_audio_int16(
                audio_captured,
                simulate_audio_input_rate,
                infer.audio_rate,
                infer.audio_samples,
            )
            pressure_board_source = resample_1d_np(
                pressure_window_100,
                max(2, int(round(simulate_sensor_rate * infer.window_sec))),
            )
            flow_board_source = resample_1d_np(
                flow_window_100,
                max(2, int(round(simulate_sensor_rate * infer.window_sec))),
            )

            normalization_stats = None
            if args.normalization_mode == "rolling":
                end_sec = start_sec + infer.window_sec
                ctx_start_sec = max(0.0, end_sec - normalization_buffer_sec)
                ctx_dur_sec = end_sec - ctx_start_sec

                audio_context_raw = extract_fixed_window(audio, audio_rate, ctx_start_sec, ctx_dur_sec)
                audio_context_captured = resample_audio_int16(
                    audio_context_raw,
                    audio_rate,
                    simulate_audio_input_rate,
                    max(2, int(round(simulate_audio_input_rate * ctx_dur_sec))),
                )
                audio_context_model = resample_audio_int16(
                    audio_context_captured,
                    simulate_audio_input_rate,
                    infer.audio_rate,
                    max(2, int(round(infer.audio_rate * ctx_dur_sec))),
                )
                pressure_context_100 = extract_fixed_window(pressure, daq_rate, ctx_start_sec, ctx_dur_sec)
                flow_context_100 = extract_fixed_window(flow, daq_rate, ctx_start_sec, ctx_dur_sec)
                pressure_context_board = resample_1d_np(
                    pressure_context_100,
                    max(2, int(round(simulate_sensor_rate * ctx_dur_sec))),
                )
                flow_context_board = resample_1d_np(
                    flow_context_100,
                    max(2, int(round(simulate_sensor_rate * ctx_dur_sec))),
                )
                normalization_stats = {
                    "audio": normalization_stats_from_values(
                        audio_context_model.astype(np.float32, copy=False) / 32768.0
                    ),
                    "pressure": normalization_stats_from_values(pressure_context_board),
                    "flow": normalization_stats_from_values(flow_context_board),
                }

            result = infer.infer(
                audio_board,
                pressure_board_source,
                flow_board_source,
                normalization_stats=normalization_stats,
            )
            board_probs = list(result["probabilities"])
            row: dict[str, object] = {
                "session": session_dir.name,
                "window_index": window_index,
                "start_sec": float(start_sec),
                "expected_label": expected_label,
                "board_label": result["predicted_label"],
                "prediction_matches": "",
                "board_prob_max": max(board_probs),
                "board_prob_argmax": result["predicted_label"],
            }
            for name, value in zip(class_names, board_probs):
                row[f"board_prob_{name}"] = value
            rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["session", "window_index", "start_sec", "expected_label", "board_label", "prediction_matches"]
        fieldnames += [f"board_prob_{name}" for name in class_names]
        fieldnames += ["board_prob_max", "board_prob_argmax"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pc_csv.open("r", encoding="utf-8", newline="") as f:
        pc_rows = {(r["session"], r["window_index"], r["start_sec"]): r for r in csv.DictReader(f)}
    board_rows = {(str(r["session"]), str(r["window_index"]), str(r["start_sec"])): r for r in rows}
    keys = sorted(set(pc_rows) & set(board_rows))
    max_abs = []
    mean_abs = []
    label_match = 0
    for key in keys:
        pc_row = pc_rows[key]
        board_row = board_rows[key]
        diffs = [abs(float(pc_row[f"pc_prob_{c}"]) - float(board_row[f"board_prob_{c}"])) for c in class_names]
        max_abs.append(max(diffs))
        mean_abs.append(sum(diffs) / len(diffs))
        label_match += int(pc_row["pc_label"] == board_row["board_label"])

    summary = {
        "backend_requested": args.backend,
        "backend_actual": infer.preprocess_backend,
        "normalization_mode": args.normalization_mode,
        "normalization_buffer_sec": normalization_buffer_sec if args.normalization_mode == "rolling" else None,
        "output_csv": str(output_csv),
        "windows": len(keys),
        "label_match_rate_vs_pc": (label_match / len(keys)) if keys else 0.0,
        "max_abs_prob_diff_stats": stats(max_abs),
        "mean_abs_prob_diff_stats": stats(mean_abs),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
