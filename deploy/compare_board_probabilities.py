from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from edge_deploy_utils import load_deploy_config
from mmdl_baseline.preprocessing.signals import (
    build_mel_transform,
    compute_audio_features,
    load_audio,
    load_daq,
    resolve_audio_frontend_config,
    slice_or_pad_1d,
    zscore_np,
    zscore_torch,
)


EPS = 1e-8


@dataclass
class SessionCache:
    session_id: str
    audio_waveform: Any
    pressure: Any
    flow: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare board probabilities with local ONNX probabilities.")
    parser.add_argument(
        "--board-csv",
        default="deploy/board_probabilities_by_window.csv",
        help="CSV exported from the board side.",
    )
    parser.add_argument(
        "--sessions-root",
        default="board_test_sessions",
        help="Root directory containing smoke/benchmark board test sessions.",
    )
    parser.add_argument(
        "--config",
        default="configs/final_model_unified_evidence.yaml",
        help="Canonical config used for local preprocessing.",
    )
    parser.add_argument(
        "--onnx",
        default="deploy/artifacts/hcaf_pcen_dualxattn.onnx",
        help="Local ONNX model path.",
    )
    parser.add_argument(
        "--output-csv",
        default="deploy/board_probabilities_comparison.csv",
        help="Detailed comparison CSV output path.",
    )
    parser.add_argument(
        "--summary-json",
        default="deploy/board_probabilities_comparison_summary.json",
        help="Summary JSON output path.",
    )
    return parser.parse_args()


def softmax_np(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def resolve_session_dir(sessions_root: Path, session_id: str) -> Path:
    candidates = [
        sessions_root / "smoke" / session_id,
        sessions_root / "benchmark" / session_id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find session directory for {session_id}")


def load_session_cache(
    session_id: str,
    session_dir: Path,
    audio_sr: int,
) -> SessionCache:
    audio_waveform, _ = load_audio(session_dir / "audio.wav", audio_sr)
    daq = load_daq(session_dir / "daq.csv")
    return SessionCache(
        session_id=session_id,
        audio_waveform=zscore_torch(audio_waveform),
        pressure=torch.from_numpy(zscore_np(daq["pressure"])),
        flow=torch.from_numpy(zscore_np(daq["flow"])),
    )


def build_window_inputs(
    cache: SessionCache,
    start_sec: float,
    *,
    audio_sr: int,
    sensor_sr: int,
    audio_length: int,
    sensor_length: int,
    mel_transform: Any,
    frontend: dict[str, Any],
) -> dict[str, np.ndarray]:
    audio_start = int(round(start_sec * audio_sr))
    sensor_start = int(round(start_sec * sensor_sr))

    audio_window = slice_or_pad_1d(cache.audio_waveform, audio_start, audio_length)
    pressure_window = slice_or_pad_1d(cache.pressure, sensor_start, sensor_length)
    flow_window = slice_or_pad_1d(cache.flow, sensor_start, sensor_length)
    audio_feature = compute_audio_features(audio_window, audio_sr, mel_transform, frontend)

    return {
        "audio": audio_feature.unsqueeze(0).numpy().astype(np.float32, copy=False),
        "pressure": np.asarray(pressure_window, dtype=np.float32)[None, None, :],
        "flow": np.asarray(flow_window, dtype=np.float32)[None, None, :],
    }


def compute_relative_logits(probs: np.ndarray) -> np.ndarray:
    log_probs = np.log(np.clip(probs, EPS, 1.0))
    return log_probs - np.mean(log_probs, axis=-1, keepdims=True)


def main() -> None:
    args = parse_args()
    board_csv_path = Path(args.board_csv)
    sessions_root = Path(args.sessions_root)
    output_csv_path = Path(args.output_csv)
    summary_json_path = Path(args.summary_json)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_deploy_config(args.config)
    class_labels = [int(label) for label in config["task"]["class_subset"]]
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}

    audio_sr = int(config["audio_sample_rate"])
    sensor_sr = int(config["sensor_sample_rate"])
    window_sec = float(config["window_sec"])
    audio_length = int(round(window_sec * audio_sr))
    sensor_length = int(round(window_sec * sensor_sr))
    frontend = resolve_audio_frontend_config(config)
    mel_transform = build_mel_transform(audio_sr, frontend)

    ort_session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    session_cache_map: dict[str, SessionCache] = {}
    comparison_rows: list[dict[str, Any]] = []

    with board_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            session_id = row["session"]
            if session_id not in session_cache_map:
                session_dir = resolve_session_dir(sessions_root, session_id)
                session_cache_map[session_id] = load_session_cache(session_id, session_dir, audio_sr)

            start_sec = float(row["start_sec"])
            local_inputs = build_window_inputs(
                session_cache_map[session_id],
                start_sec,
                audio_sr=audio_sr,
                sensor_sr=sensor_sr,
                audio_length=audio_length,
                sensor_length=sensor_length,
                mel_transform=mel_transform,
                frontend=frontend,
            )

            logits = ort_session.run(["logits"], local_inputs)[0]
            local_probs = softmax_np(logits)[0]

            board_probs = np.asarray([float(row[f"board_prob_{label}"]) for label in class_labels], dtype=np.float64)
            local_probs = np.asarray(local_probs, dtype=np.float64)
            prob_diff = local_probs - board_probs

            board_relative_logits = compute_relative_logits(board_probs[None, :])[0]
            local_relative_logits = compute_relative_logits(local_probs[None, :])[0]
            relative_logit_diff = local_relative_logits - board_relative_logits

            out_row: dict[str, Any] = {
                "session": session_id,
                "window_index": int(row["window_index"]),
                "start_sec": start_sec,
                "expected_label": int(row["expected_label"]),
                "board_label": int(row["board_label"]),
                "board_prob_argmax": int(row["board_prob_argmax"]),
                "local_label": class_labels[int(np.argmax(local_probs))],
                "local_prob_argmax": class_labels[int(np.argmax(local_probs))],
                "prediction_matches_board": int(class_labels[int(np.argmax(local_probs))] == int(row["board_prob_argmax"])),
            }

            for label in class_labels:
                idx = label_to_index[label]
                out_row[f"board_prob_{label}"] = float(board_probs[idx])
                out_row[f"local_prob_{label}"] = float(local_probs[idx])
                out_row[f"delta_prob_{label}"] = float(prob_diff[idx])
                out_row[f"board_relative_logit_{label}"] = float(board_relative_logits[idx])
                out_row[f"local_relative_logit_{label}"] = float(local_relative_logits[idx])
                out_row[f"delta_relative_logit_{label}"] = float(relative_logit_diff[idx])

            comparison_rows.append(out_row)

    fieldnames = list(comparison_rows[0].keys())
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary: dict[str, Any] = {
        "board_csv": str(board_csv_path.resolve()),
        "onnx": str(Path(args.onnx).resolve()),
        "config": str(Path(args.config).resolve()),
        "window_sec": window_sec,
        "num_windows": len(comparison_rows),
        "sessions": sorted(session_cache_map.keys()),
        "classes": class_labels,
        "per_class": {},
    }

    for label in class_labels:
        prob_diffs = np.asarray([row[f"delta_prob_{label}"] for row in comparison_rows], dtype=np.float64)
        rel_logit_diffs = np.asarray(
            [row[f"delta_relative_logit_{label}"] for row in comparison_rows],
            dtype=np.float64,
        )
        summary["per_class"][str(label)] = {
            "mean_prob_diff": float(np.mean(prob_diffs)),
            "mean_abs_prob_diff": float(np.mean(np.abs(prob_diffs))),
            "median_abs_prob_diff": float(np.median(np.abs(prob_diffs))),
            "max_abs_prob_diff": float(np.max(np.abs(prob_diffs))),
            "std_prob_diff": float(np.std(prob_diffs)),
            "mean_relative_logit_diff": float(np.mean(rel_logit_diffs)),
            "mean_abs_relative_logit_diff": float(np.mean(np.abs(rel_logit_diffs))),
            "median_abs_relative_logit_diff": float(np.median(np.abs(rel_logit_diffs))),
            "max_abs_relative_logit_diff": float(np.max(np.abs(rel_logit_diffs))),
            "std_relative_logit_diff": float(np.std(rel_logit_diffs)),
        }

    overall_prob_diffs = np.concatenate(
        [
            np.asarray([row[f"delta_prob_{label}"] for row in comparison_rows], dtype=np.float64)
            for label in class_labels
        ]
    )
    overall_rel_logit_diffs = np.concatenate(
        [
            np.asarray([row[f"delta_relative_logit_{label}"] for row in comparison_rows], dtype=np.float64)
            for label in class_labels
        ]
    )
    summary["overall"] = {
        "mean_abs_prob_diff": float(np.mean(np.abs(overall_prob_diffs))),
        "max_abs_prob_diff": float(np.max(np.abs(overall_prob_diffs))),
        "mean_abs_relative_logit_diff": float(np.mean(np.abs(overall_rel_logit_diffs))),
        "max_abs_relative_logit_diff": float(np.max(np.abs(overall_rel_logit_diffs))),
        "match_rate_with_board_argmax": float(
            np.mean([row["prediction_matches_board"] for row in comparison_rows])
        ),
    }

    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
