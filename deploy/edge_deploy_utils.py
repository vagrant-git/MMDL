from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmdl_baseline.models.factory import build_model
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
from mmdl_baseline.utils.config import load_config


@dataclass
class SessionPaths:
    session_dir: Path
    audio_path: Path
    daq_path: Path
    metadata_path: Path


def load_deploy_config(config_path: str | Path) -> dict[str, Any]:
    return load_config(config_path)


def infer_num_classes(config: dict[str, Any]) -> int:
    task_cfg = config.get("task") or {}
    class_subset = task_cfg.get("class_subset")
    if class_subset:
        return len(class_subset)
    labels = config.get("labels") or {}
    return len(labels)


def resolve_session_paths(session_dir: str | Path) -> SessionPaths:
    session_path = Path(session_dir)
    return SessionPaths(
        session_dir=session_path,
        audio_path=session_path / "audio.wav",
        daq_path=session_path / "daq.csv",
        metadata_path=session_path / "metadata.json",
    )


def load_session_metadata(session_dir: str | Path) -> dict[str, Any]:
    paths = resolve_session_paths(session_dir)
    return json.loads(paths.metadata_path.read_text(encoding="utf-8"))


def build_multimodal_model(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    model = build_model("multimodal", config, num_classes=infer_num_classes(config))
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


def build_window_batch(
    config: dict[str, Any],
    audio_path: str | Path,
    daq_path: str | Path,
    start_sec: float = 0.0,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    audio_sr = int(config["audio_sample_rate"])
    sensor_sr = int(config["sensor_sample_rate"])
    window_sec = float(config["window_sec"])
    audio_length = int(round(window_sec * audio_sr))
    sensor_length = int(round(window_sec * sensor_sr))
    audio_start = int(round(start_sec * audio_sr))
    sensor_start = int(round(start_sec * sensor_sr))

    audio_waveform, _ = load_audio(audio_path, audio_sr)
    daq = load_daq(daq_path)
    pressure = torch.from_numpy(zscore_np(daq["pressure"]))
    flow = torch.from_numpy(zscore_np(daq["flow"]))
    audio_waveform = zscore_torch(audio_waveform)

    audio_window = slice_or_pad_1d(audio_waveform, audio_start, audio_length)
    pressure_window = slice_or_pad_1d(pressure, sensor_start, sensor_length)
    flow_window = slice_or_pad_1d(flow, sensor_start, sensor_length)

    frontend = resolve_audio_frontend_config(config)
    mel_transform = build_mel_transform(audio_sr, frontend)
    audio_feature = compute_audio_features(audio_window, audio_sr, mel_transform, frontend)

    return {
        "audio": audio_feature.unsqueeze(0).to(device),
        "pressure": pressure_window.unsqueeze(0).unsqueeze(0).to(device),
        "flow": flow_window.unsqueeze(0).unsqueeze(0).to(device),
    }


def build_dummy_batch(
    config: dict[str, Any],
    batch_size: int = 1,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    audio_sr = int(config["audio_sample_rate"])
    sensor_sr = int(config["sensor_sample_rate"])
    window_sec = float(config["window_sec"])
    audio_length = int(round(window_sec * audio_sr))
    sensor_length = int(round(window_sec * sensor_sr))

    frontend = resolve_audio_frontend_config(config)
    mel_transform = build_mel_transform(audio_sr, frontend)
    zero_waveform = torch.zeros(audio_length, dtype=torch.float32)
    audio_feature = compute_audio_features(zero_waveform, audio_sr, mel_transform, frontend)

    return {
        "audio": audio_feature.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device),
        "pressure": torch.zeros(batch_size, 1, sensor_length, dtype=torch.float32, device=device),
        "flow": torch.zeros(batch_size, 1, sensor_length, dtype=torch.float32, device=device),
    }
