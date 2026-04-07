from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from deploy.edge_deploy_utils import load_deploy_config, build_window_batch


LABEL_MAP = {
    "no secretion": 0,
    "none": 0,
    "0ml": 0,
    "0 ml": 0,
    "2ml": 2,
    "2 ml": 2,
    "4ml": 4,
    "4 ml": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce PC_probabilities_by_window.csv for smoke sessions.")
    parser.add_argument("--sessions-root", default="board_test_sessions/smoke")
    parser.add_argument("--config", default="configs/final_model_unified_evidence.yaml")
    parser.add_argument("--onnx", default="deploy/artifacts/hcaf_pcen_dualxattn.onnx")
    parser.add_argument("--output-csv", default="board_test_sessions/smoke_diff_artifacts/PC_probabilities_by_window.csv")
    return parser.parse_args()


def softmax_np(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def canonical_label(label_text: str) -> int:
    key = label_text.strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"Unsupported label text: {label_text}")
    return LABEL_MAP[key]


def main() -> None:
    args = parse_args()
    sessions_root = Path(args.sessions_root)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    config = load_deploy_config(args.config)
    class_subset = [int(x) for x in config["task"]["class_subset"]]
    window_sec = float(config["window_sec"])
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    fieldnames = [
        "session",
        "window_index",
        "start_sec",
        "expected_label",
        "pc_label",
        "prediction_matches",
        "pc_prob_0",
        "pc_prob_2",
        "pc_prob_4",
        "pc_prob_max",
        "pc_prob_argmax",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for session_dir in sorted(sessions_root.glob("MMdata_*")):
            meta = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
            expected_label = canonical_label(str(meta["label"]))
            duration_sec = float(meta["duration_sec"])
            num_windows = int(np.floor((duration_sec - window_sec) / window_sec)) + 1

            for window_index in range(num_windows):
                start_sec = window_index * window_sec
                batch = build_window_batch(
                    config=config,
                    audio_path=session_dir / "audio.wav",
                    daq_path=session_dir / "daq.csv",
                    start_sec=start_sec,
                    device="cpu",
                )
                logits = session.run(
                    ["logits"],
                    {name: tensor.detach().cpu().numpy().astype(np.float32) for name, tensor in batch.items()},
                )[0]
                probs = softmax_np(logits)[0]
                predicted_index = int(np.argmax(probs))
                predicted_label = class_subset[predicted_index]

                writer.writerow(
                    {
                        "session": session_dir.name,
                        "window_index": window_index,
                        "start_sec": start_sec,
                        "expected_label": expected_label,
                        "pc_label": predicted_label,
                        "prediction_matches": str(predicted_label == expected_label),
                        "pc_prob_0": float(probs[0]),
                        "pc_prob_2": float(probs[1]),
                        "pc_prob_4": float(probs[2]),
                        "pc_prob_max": float(np.max(probs)),
                        "pc_prob_argmax": predicted_label,
                    }
                )

    print(output_csv)


if __name__ == "__main__":
    main()
