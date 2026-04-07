from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from edge_deploy_utils import (
    build_multimodal_model,
    build_window_batch,
    load_deploy_config,
    load_session_metadata,
    resolve_session_paths,
)

DEFAULT_INPUT_DUMP = Path("deploy/artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline 5-second multimodal inference for one MMdata session.")
    parser.add_argument("session_dir", help="Path to one MMdata_* session directory.")
    parser.add_argument(
        "--config",
        default="configs/final_model_unified_evidence.yaml",
        help="Path to the canonical experiment config.",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "summary-MMmodel/final_model_unified_evidence/runs/"
            "hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold1/best_model.pt"
        ),
        help="Path to a trained multimodal checkpoint.",
    )
    parser.add_argument(
        "--onnx",
        help="Optional ONNX path. If provided with --backend onnx/both, run ONNX Runtime inference too.",
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "onnx", "both"],
        default="pytorch",
        help="Inference backend to run.",
    )
    parser.add_argument("--start-sec", type=float, default=0.0, help="Window start time in seconds.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device. Use cpu on board unless you know a different backend is available.",
    )
    parser.add_argument(
        "--save-inputs",
        help="Optional .npz path for saving the preprocessed model inputs.",
    )
    return parser.parse_args()


def softmax_np(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def run_pytorch(
    config: dict[str, object],
    checkpoint_path: str | Path,
    batch: dict[str, torch.Tensor],
    device: str,
) -> dict[str, object]:
    model = build_multimodal_model(config, checkpoint_path, device=device)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(batch)
        probs = torch.softmax(logits, dim=-1)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logits_np = logits.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    return {
        "latency_ms": elapsed_ms,
        "logits": logits_np[0].tolist(),
        "probabilities": probs_np[0].tolist(),
        "predicted_index": int(np.argmax(probs_np[0])),
    }


def run_onnx(
    onnx_path: str | Path,
    batch: dict[str, torch.Tensor],
) -> dict[str, object]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "audio": batch["audio"].detach().cpu().numpy().astype(np.float32),
        "pressure": batch["pressure"].detach().cpu().numpy().astype(np.float32),
        "flow": batch["flow"].detach().cpu().numpy().astype(np.float32),
    }
    start = time.perf_counter()
    logits = session.run(["logits"], ort_inputs)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    probs = softmax_np(logits)
    return {
        "latency_ms": elapsed_ms,
        "logits": logits[0].tolist(),
        "probabilities": probs[0].tolist(),
        "predicted_index": int(np.argmax(probs[0])),
    }


def main() -> None:
    args = parse_args()
    config = load_deploy_config(args.config)
    paths = resolve_session_paths(args.session_dir)
    metadata = load_session_metadata(args.session_dir)

    batch = build_window_batch(
        config=config,
        audio_path=paths.audio_path,
        daq_path=paths.daq_path,
        start_sec=args.start_sec,
        device=args.device if args.backend in {"pytorch", "both"} else "cpu",
    )

    if args.save_inputs:
        save_path = Path(args.save_inputs)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            audio=batch["audio"].detach().cpu().numpy(),
            pressure=batch["pressure"].detach().cpu().numpy(),
            flow=batch["flow"].detach().cpu().numpy(),
        )

    result: dict[str, object] = {
        "session_dir": str(paths.session_dir.resolve()),
        "session_id": paths.session_dir.name,
        "label_text": metadata.get("label"),
        "start_sec": args.start_sec,
        "window_sec": float(config["window_sec"]),
        "audio_shape": list(batch["audio"].shape),
        "pressure_shape": list(batch["pressure"].shape),
        "flow_shape": list(batch["flow"].shape),
    }

    pytorch_result = None
    onnx_result = None
    if args.backend in {"pytorch", "both"}:
        pytorch_result = run_pytorch(config, args.checkpoint, batch, device=args.device)
        result["pytorch"] = pytorch_result

    if args.backend in {"onnx", "both"}:
        if not args.onnx:
            raise ValueError("--onnx is required when --backend is onnx or both.")
        onnx_batch = {name: tensor.to("cpu") for name, tensor in batch.items()}
        onnx_result = run_onnx(args.onnx, onnx_batch)
        result["onnx"] = onnx_result

    if pytorch_result is not None and onnx_result is not None:
        pt_logits = np.asarray(pytorch_result["logits"], dtype=np.float32)
        onnx_logits = np.asarray(onnx_result["logits"], dtype=np.float32)
        result["parity"] = {
            "max_abs_diff": float(np.max(np.abs(pt_logits - onnx_logits))),
            "mean_abs_diff": float(np.mean(np.abs(pt_logits - onnx_logits))),
            "same_predicted_index": bool(
                int(pytorch_result["predicted_index"]) == int(onnx_result["predicted_index"])
            ),
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
