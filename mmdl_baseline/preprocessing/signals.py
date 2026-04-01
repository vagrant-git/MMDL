from __future__ import annotations

import csv
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F


def load_audio(audio_path: str | Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    with wave.open(str(audio_path), "rb") as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        raw_bytes = wav_file.readframes(num_frames)
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM wav is supported, got sample width={sample_width}")
    audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        audio_np = audio_np.reshape(-1, num_channels).mean(axis=1)
    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        sample_rate = target_sr
    return waveform.squeeze(0), sample_rate


def load_daq(daq_path: str | Path) -> Dict[str, np.ndarray]:
    time_vals: List[float] = []
    pressure_vals: List[float] = []
    flow_vals: List[float] = []
    with Path(daq_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_vals.append(float(row["Time (s)"]))
            pressure_vals.append(float(row["Pressure (cmH2O)"]))
            flow_vals.append(float(row["Flowrate (L/min)"]))
    time_array = np.asarray(time_vals, dtype=np.float32)
    time_array = time_array - time_array[0]
    return {
        "time": time_array,
        "pressure": np.asarray(pressure_vals, dtype=np.float32),
        "flow": np.asarray(flow_vals, dtype=np.float32),
    }


def zscore_np(x: np.ndarray) -> np.ndarray:
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-6:
        std = 1.0
    return (x - mean) / std


def zscore_torch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(x)
    std = torch.std(x)
    if torch.isnan(std) or std < 1e-6:
        std = torch.tensor(1.0, dtype=x.dtype, device=x.device)
    return (x - mean) / std


def resolve_audio_frontend_config(config: Dict[str, object]) -> Dict[str, object]:
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


def audio_feature_channels(frontend_config: Dict[str, object]) -> int:
    return 1 + int(frontend_config.get("delta_order", 0))


def build_mel_transform(sample_rate: int, frontend_config: Dict[str, object]) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=int(frontend_config["n_fft"]),
        win_length=int(frontend_config["win_length"]),
        hop_length=int(frontend_config["hop_length"]),
        n_mels=int(frontend_config["n_mels"]),
        f_min=float(frontend_config.get("f_min", 0.0)),
        f_max=float(frontend_config.get("f_max", sample_rate / 2)),
        power=float(frontend_config.get("power", 2.0)),
    )


def apply_audio_filters(
    waveform: torch.Tensor,
    sample_rate: int,
    frontend_config: Dict[str, object],
) -> torch.Tensor:
    waveform = waveform.unsqueeze(0)
    preemphasis = float(frontend_config.get("preemphasis", 0.0) or 0.0)
    if preemphasis > 0.0:
        waveform = torch.cat(
            [waveform[:, :1], waveform[:, 1:] - preemphasis * waveform[:, :-1]],
            dim=1,
        )
    highpass_hz = frontend_config.get("highpass_hz")
    if highpass_hz is not None:
        waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, float(highpass_hz))
    lowpass_hz = frontend_config.get("lowpass_hz")
    if lowpass_hz is not None:
        waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, float(lowpass_hz))
    return waveform.squeeze(0)


def _pcen(spec: torch.Tensor, frontend_config: Dict[str, object]) -> torch.Tensor:
    alpha = float(frontend_config.get("pcen_alpha", 0.98))
    delta = float(frontend_config.get("pcen_delta", 2.0))
    root = float(frontend_config.get("pcen_r", 0.5))
    smoothing = float(frontend_config.get("pcen_s", 0.025))
    eps = float(frontend_config.get("pcen_eps", 1e-6))
    smoothed = [spec[:, :1]]
    for frame_idx in range(1, spec.shape[1]):
        prev = smoothed[-1]
        current = (1.0 - smoothing) * prev + smoothing * spec[:, frame_idx : frame_idx + 1]
        smoothed.append(current)
    smoother = torch.cat(smoothed, dim=1)
    return (spec / (eps + smoother).pow(alpha) + delta).pow(root) - delta**root


def _crop_mel_band(
    feature: torch.Tensor,
    reference_spec: torch.Tensor,
    frontend_config: Dict[str, object],
) -> torch.Tensor:
    start_bin = int(frontend_config.get("mel_bin_start", 0) or 0)
    end_bin = int(frontend_config.get("mel_bin_end", feature.shape[0]) or feature.shape[0])
    end_bin = max(start_bin + 1, min(end_bin, feature.shape[0]))

    if frontend_config.get("adaptive_top_crop"):
        ratio = float(frontend_config.get("adaptive_top_crop_ratio", 0.05))
        min_keep_bins = int(frontend_config.get("adaptive_top_crop_min_bins", max(16, feature.shape[0] // 2)))
        energy = reference_spec.mean(dim=1)
        max_energy = float(torch.max(energy).item()) if energy.numel() > 0 else 0.0
        threshold = max_energy * ratio
        active = torch.where(energy >= threshold)[0]
        if active.numel() > 0:
            adaptive_end = max(int(active[-1].item()) + 1, start_bin + min_keep_bins)
            end_bin = min(max(adaptive_end, start_bin + 1), feature.shape[0])

    cropped = feature[start_bin:end_bin]
    target_bins = int(frontend_config.get("output_n_mels", frontend_config["n_mels"]))
    if cropped.shape[0] != target_bins:
        cropped = F.interpolate(
            cropped.unsqueeze(0).unsqueeze(0),
            size=(target_bins, cropped.shape[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    return cropped


def compute_audio_features(
    waveform: torch.Tensor,
    sample_rate: int,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    frontend_config: Dict[str, object],
) -> torch.Tensor:
    filtered = apply_audio_filters(waveform, sample_rate, frontend_config)
    spec = mel_transform(filtered.unsqueeze(0)).squeeze(0)

    feature_type = str(frontend_config.get("feature_type", "logmel")).lower()
    compression = str(frontend_config.get("compression", "log")).lower()
    if feature_type == "pcen":
        base_feature = _pcen(spec, frontend_config)
    elif compression == "db":
        base_feature = torchaudio.functional.amplitude_to_DB(
            spec,
            multiplier=10.0,
            amin=1e-10,
            db_multiplier=0.0,
            top_db=float(frontend_config.get("top_db", 80.0)),
        )
    else:
        base_feature = torch.log(spec + 1e-5)

    base_feature = _crop_mel_band(base_feature, spec, frontend_config)
    base_feature = (base_feature - base_feature.mean()) / (base_feature.std() + 1e-6)
    channels = [base_feature]
    delta_order = int(frontend_config.get("delta_order", 0))
    if delta_order >= 1:
        delta1 = torchaudio.functional.compute_deltas(base_feature.unsqueeze(0)).squeeze(0)
        channels.append(delta1)
    if delta_order >= 2:
        delta2 = torchaudio.functional.compute_deltas(channels[1].unsqueeze(0)).squeeze(0)
        channels.append(delta2)
    return torch.stack(channels, dim=0)


def slice_or_pad_1d(x: torch.Tensor, start: int, length: int) -> torch.Tensor:
    segment = x[start : start + length]
    if segment.numel() == length:
        return segment
    pad_amount = length - segment.numel()
    return F.pad(segment, (0, pad_amount))


def resample_1d_to_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros(target_length, dtype=x.dtype, device=x.device)
    if x.numel() == target_length:
        return x
    if x.numel() == 1:
        return x.repeat(target_length)
    resized = F.interpolate(
        x.unsqueeze(0).unsqueeze(0),
        size=target_length,
        mode="linear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0)


def moving_average_np(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(x, kernel, mode="same")


def detect_flow_cycle_start_times(
    time_array: np.ndarray,
    flow_array: np.ndarray,
    threshold_lpm: float = 2.0,
    smooth_width: int = 21,
    min_phase_sec: float = 0.6,
    min_cycle_sec: float = 2.5,
    max_cycle_sec: float = 5.5,
) -> List[float]:
    smoothed = moving_average_np(flow_array, smooth_width)
    state = np.zeros_like(smoothed, dtype=np.int8)
    state[smoothed >= threshold_lpm] = 1
    state[smoothed <= -threshold_lpm] = -1

    phase_starts: List[Tuple[float, int]] = []
    current_state = int(state[0])
    current_start = float(time_array[0])
    for idx in range(1, len(state)):
        next_state = int(state[idx])
        if next_state == current_state:
            continue
        phase_end = float(time_array[idx])
        if current_state != 0 and phase_end - current_start >= min_phase_sec:
            phase_starts.append((current_start, current_state))
        current_state = next_state
        current_start = float(time_array[idx])
    if current_state != 0 and float(time_array[-1]) - current_start >= min_phase_sec:
        phase_starts.append((current_start, current_state))

    cycle_starts: List[float] = []
    for idx in range(len(phase_starts) - 2):
        start_t, start_state = phase_starts[idx]
        mid_t, mid_state = phase_starts[idx + 1]
        next_t, next_state = phase_starts[idx + 2]
        if start_state == 1 and mid_state == -1 and next_state == 1:
            cycle_sec = next_t - start_t
            if min_cycle_sec <= cycle_sec <= max_cycle_sec:
                cycle_starts.append(start_t)
    return cycle_starts
