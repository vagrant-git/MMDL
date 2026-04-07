from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import yaml

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("librosa is required for RK3588 ONNX preprocessing.") from exc

try:
    from scipy import signal
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scipy is required for RK3588 ONNX preprocessing.") from exc

SCRIPT_PATH = Path(__file__).resolve()
for candidate in (SCRIPT_PATH.parents[1], SCRIPT_PATH.parents[2]):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    import torch
    from mmdl_baseline.preprocessing.signals import (
        build_mel_transform as torch_build_mel_transform,
        compute_audio_features as torch_compute_audio_features,
        zscore_torch,
    )

    HAS_TORCH_PREPROCESS = True
    TORCH_PREPROCESS_IMPORT_ERROR: str | None = None
except Exception as exc:
    torch = None
    torch_build_mel_transform = None
    torch_compute_audio_features = None
    zscore_torch = None
    HAS_TORCH_PREPROCESS = False
    TORCH_PREPROCESS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_audio_frontend_config(config: dict[str, Any]) -> dict[str, Any]:
    frontend = {
        "feature_type": "logmel",
        "sample_rate": int(config["audio_sample_rate"]),
        "n_mels": int(config["audio_n_mels"]),
        "n_fft": int(config["audio_n_fft"]),
        "hop_length": int(config["audio_hop_length"]),
        "win_length": int(config["audio_win_length"]),
        "f_min": 0.0,
        "f_max": float(int(config["audio_sample_rate"]) / 2),
        "power": 2.0,
        "compression": "log",
        "top_db": 80.0,
        "preemphasis": 0.0,
        "highpass_hz": None,
        "lowpass_hz": None,
        "delta_order": 0,
        "pcen_s": 0.025,
        "pcen_alpha": 0.98,
        "pcen_delta": 2.0,
        "pcen_r": 0.5,
        "pcen_eps": 1e-6,
    }
    frontend.update(config.get("audio_frontend") or {})
    return frontend


def zscore_np(x: np.ndarray) -> np.ndarray:
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-6:
        std = 1.0
    return ((x - mean) / std).astype(np.float32, copy=False)


def normalize_with_stats_np(x: np.ndarray, stats: dict[str, float] | None) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if stats is None:
        return zscore_np(x)
    mean = float(stats.get("mean", 0.0))
    std = float(stats.get("std", 1.0))
    if std < 1e-6:
        std = 1.0
    return ((x - mean) / std).astype(np.float32, copy=False)


def resample_1d_np(x: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 1:
        raise ValueError("target_length must be greater than 1.")
    x = x.astype(np.float32, copy=False)
    if x.shape[0] == target_length:
        return x
    if x.shape[0] < 2:
        raise ValueError(f"Need at least 2 samples to resample, got {x.shape[0]}")
    src_pos = np.linspace(0.0, 1.0, num=x.shape[0], dtype=np.float32)
    dst_pos = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(dst_pos, src_pos, x).astype(np.float32, copy=False)


def apply_audio_filters(waveform: np.ndarray, sample_rate: int, frontend_config: dict[str, Any]) -> np.ndarray:
    waveform = waveform.astype(np.float32, copy=False)
    preemphasis = float(frontend_config.get("preemphasis", 0.0) or 0.0)
    if preemphasis > 0.0 and waveform.size > 1:
        waveform = np.concatenate([waveform[:1], waveform[1:] - preemphasis * waveform[:-1]])

    highpass_hz = frontend_config.get("highpass_hz")
    if highpass_hz is not None:
        sos = signal.butter(2, float(highpass_hz), btype="highpass", fs=sample_rate, output="sos")
        waveform = signal.sosfilt(sos, waveform).astype(np.float32, copy=False)

    lowpass_hz = frontend_config.get("lowpass_hz")
    if lowpass_hz is not None:
        sos = signal.butter(2, float(lowpass_hz), btype="lowpass", fs=sample_rate, output="sos")
        waveform = signal.sosfilt(sos, waveform).astype(np.float32, copy=False)
    return waveform


def compute_audio_feature(waveform: np.ndarray, sample_rate: int, frontend_config: dict[str, Any]) -> np.ndarray:
    filtered = apply_audio_filters(waveform, sample_rate, frontend_config)
    spec = librosa.feature.melspectrogram(
        y=filtered,
        sr=sample_rate,
        n_fft=int(frontend_config["n_fft"]),
        hop_length=int(frontend_config["hop_length"]),
        win_length=int(frontend_config["win_length"]),
        n_mels=int(frontend_config["n_mels"]),
        fmin=float(frontend_config.get("f_min", 0.0)),
        fmax=float(frontend_config.get("f_max", sample_rate / 2)),
        power=float(frontend_config.get("power", 2.0)),
        center=True,
    ).astype(np.float32, copy=False)

    feature_type = str(frontend_config.get("feature_type", "logmel")).lower()
    compression = str(frontend_config.get("compression", "log")).lower()
    if feature_type == "pcen":
        base_feature = librosa.pcen(
            spec,
            sr=sample_rate,
            hop_length=int(frontend_config["hop_length"]),
            gain=float(frontend_config.get("pcen_alpha", 0.98)),
            bias=float(frontend_config.get("pcen_delta", 2.0)),
            power=float(frontend_config.get("pcen_r", 0.5)),
            time_constant=float(frontend_config.get("pcen_s", 0.025)),
            eps=float(frontend_config.get("pcen_eps", 1e-6)),
        ).astype(np.float32, copy=False)
    elif compression == "db":
        base_feature = librosa.power_to_db(spec, top_db=float(frontend_config.get("top_db", 80.0))).astype(
            np.float32, copy=False
        )
    else:
        base_feature = np.log(spec + 1e-5).astype(np.float32, copy=False)

    base_feature = zscore_np(base_feature)
    return base_feature[np.newaxis, :, :].astype(np.float32, copy=False)


def softmax_np(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class RuntimeOnnxInfer:
    def __init__(self, onnx_path: str | Path, config_path: str | Path) -> None:
        self.onnx_path = str(onnx_path)
        self.config = load_config(config_path)
        self.frontend = resolve_audio_frontend_config(self.config)
        self.audio_rate = int(self.config["audio_sample_rate"])
        self.sensor_rate = int(self.config["sensor_sample_rate"])
        self.window_sec = float(self.config["window_sec"])
        self.audio_samples = int(round(self.audio_rate * self.window_sec))
        self.sensor_samples = int(round(self.sensor_rate * self.window_sec))
        self.class_names = [str(x) for x in (self.config.get("task") or {}).get("class_names", [])]
        self.session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        self.use_precise_torch_preprocess = HAS_TORCH_PREPROCESS
        self.mel_transform = (
            torch_build_mel_transform(self.audio_rate, self.frontend) if self.use_precise_torch_preprocess else None
        )
        self.preprocess_backend = "torch" if self.use_precise_torch_preprocess else "librosa"
        self.preprocess_backend_reason = None if self.use_precise_torch_preprocess else TORCH_PREPROCESS_IMPORT_ERROR

    def _prepare_audio_feature_precise(
        self,
        audio: np.ndarray,
        audio_stats: dict[str, float] | None = None,
    ) -> np.ndarray:
        assert torch is not None
        assert zscore_torch is not None
        assert torch_compute_audio_features is not None
        assert self.mel_transform is not None
        waveform = torch.from_numpy(audio.astype(np.float32, copy=False))
        if audio_stats is None:
            waveform = zscore_torch(waveform)
        else:
            mean = float(audio_stats.get("mean", 0.0))
            std = float(audio_stats.get("std", 1.0))
            if std < 1e-6:
                std = 1.0
            waveform = (waveform - mean) / std
        audio_feature = torch_compute_audio_features(waveform, self.audio_rate, self.mel_transform, self.frontend)
        return audio_feature.detach().cpu().numpy().astype(np.float32, copy=False)

    def prepare_inputs(
        self,
        audio_int16: np.ndarray,
        pressure: np.ndarray,
        flow: np.ndarray,
        normalization_stats: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, np.ndarray]:
        audio = audio_int16.astype(np.float32, copy=False) / 32768.0
        if audio.shape[0] != self.audio_samples:
            raise ValueError(f"Expected {self.audio_samples} audio samples, got {audio.shape[0]}")
        if pressure.shape[0] != flow.shape[0]:
            raise ValueError(f"Sensor lengths do not match: pressure={pressure.shape[0]}, flow={flow.shape[0]}")
        original_sensor_length = pressure.shape[0]
        if original_sensor_length != self.sensor_samples:
            pressure = resample_1d_np(pressure, self.sensor_samples)
            flow = resample_1d_np(flow, self.sensor_samples)

        audio_stats = None if normalization_stats is None else normalization_stats.get("audio")
        pressure_stats = None if normalization_stats is None else normalization_stats.get("pressure")
        flow_stats = None if normalization_stats is None else normalization_stats.get("flow")

        pressure = normalize_with_stats_np(pressure.astype(np.float32, copy=False), pressure_stats)
        flow = normalize_with_stats_np(flow.astype(np.float32, copy=False), flow_stats)
        if self.use_precise_torch_preprocess:
            audio_feature = self._prepare_audio_feature_precise(audio, audio_stats=audio_stats)
        else:
            audio = normalize_with_stats_np(audio, audio_stats)
            audio_feature = compute_audio_feature(audio, self.audio_rate, self.frontend)

        return {
            "audio": audio_feature[np.newaxis, :, :, :].astype(np.float32, copy=False),
            "pressure": pressure[np.newaxis, np.newaxis, :].astype(np.float32, copy=False),
            "flow": flow[np.newaxis, np.newaxis, :].astype(np.float32, copy=False),
        }

    def infer(
        self,
        audio_int16: np.ndarray,
        pressure: np.ndarray,
        flow: np.ndarray,
        normalization_stats: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        ort_inputs = self.prepare_inputs(audio_int16, pressure, flow, normalization_stats=normalization_stats)
        logits = self.session.run(["logits"], ort_inputs)[0].astype(np.float32, copy=False)
        probs = softmax_np(logits)
        predicted_index = int(np.argmax(probs[0]))
        predicted_label = self.class_names[predicted_index] if predicted_index < len(self.class_names) else str(predicted_index)
        return {
            "predicted_index": predicted_index,
            "predicted_label": predicted_label,
            "logits": logits[0].tolist(),
            "probabilities": probs[0].tolist(),
        }
