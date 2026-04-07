from __future__ import annotations

import math
from typing import Any

import torch
import torchaudio


def zscore_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.to(dtype=torch.float32)
    mean = x.mean()
    std = x.std(unbiased=False)
    if float(std) < 1e-6:
        std = torch.tensor(1.0, dtype=x.dtype, device=x.device)
    return (x - mean) / std


def build_mel_transform(sample_rate: int, frontend_config: dict[str, Any]) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=int(frontend_config["n_fft"]),
        win_length=int(frontend_config["win_length"]),
        hop_length=int(frontend_config["hop_length"]),
        f_min=float(frontend_config.get("f_min", 0.0)),
        f_max=float(frontend_config.get("f_max", sample_rate / 2)),
        n_mels=int(frontend_config["n_mels"]),
        power=float(frontend_config.get("power", 2.0)),
        center=True,
        pad_mode="reflect",
        normalized=False,
        norm=None,
        mel_scale="htk",
    )


def _apply_audio_filters_torch(
    waveform: torch.Tensor,
    sample_rate: int,
    frontend_config: dict[str, Any],
) -> torch.Tensor:
    waveform = waveform.to(dtype=torch.float32)

    preemphasis = float(frontend_config.get("preemphasis", 0.0) or 0.0)
    if preemphasis > 0.0 and waveform.numel() > 1:
        waveform = torchaudio.functional.preemphasis(waveform, coeff=preemphasis)

    highpass_hz = frontend_config.get("highpass_hz")
    if highpass_hz is not None:
        waveform = torchaudio.functional.highpass_biquad(
            waveform,
            sample_rate=sample_rate,
            cutoff_freq=float(highpass_hz),
        )

    lowpass_hz = frontend_config.get("lowpass_hz")
    if lowpass_hz is not None:
        waveform = torchaudio.functional.lowpass_biquad(
            waveform,
            sample_rate=sample_rate,
            cutoff_freq=float(lowpass_hz),
        )

    return waveform


def _pcen_torch(
    spec: torch.Tensor,
    sample_rate: int,
    frontend_config: dict[str, Any],
) -> torch.Tensor:
    gain = float(frontend_config.get("pcen_alpha", 0.98))
    bias = float(frontend_config.get("pcen_delta", 2.0))
    power = float(frontend_config.get("pcen_r", 0.5))
    time_constant = float(frontend_config.get("pcen_s", 0.025))
    eps = float(frontend_config.get("pcen_eps", 1e-6))
    hop_length = int(frontend_config["hop_length"])

    t_frames = time_constant * sample_rate / float(hop_length)
    b = (math.sqrt(1.0 + 4.0 * t_frames * t_frames) - 1.0) / (2.0 * t_frames * t_frames)

    spec = spec.to(dtype=torch.float32)
    smooth = torch.empty_like(spec)
    state = torch.full(spec.shape[:-1], 1.0 - b, dtype=spec.dtype, device=spec.device)
    for frame_index in range(spec.shape[-1]):
        state = b * spec[..., frame_index] + (1.0 - b) * state
        smooth[..., frame_index] = state

    gain_term = torch.exp(-gain * (math.log(eps) + torch.log1p(smooth / eps)))
    if power == 0.0:
        return torch.log1p(spec * gain_term)
    if bias == 0.0:
        return torch.exp(power * (torch.log(spec.clamp_min(eps)) + torch.log(gain_term.clamp_min(eps))))
    return (bias**power) * torch.expm1(power * torch.log1p(spec * gain_term / bias))


def compute_audio_features(
    waveform: torch.Tensor,
    sample_rate: int,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    frontend_config: dict[str, Any],
) -> torch.Tensor:
    waveform = _apply_audio_filters_torch(waveform, sample_rate, frontend_config)
    spec = mel_transform(waveform)

    feature_type = str(frontend_config.get("feature_type", "logmel")).lower()
    compression = str(frontend_config.get("compression", "log")).lower()

    if feature_type == "pcen":
        feature = _pcen_torch(spec, sample_rate, frontend_config)
    elif compression == "db":
        feature = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=float(frontend_config.get("top_db", 80.0)))(
            spec
        )
    else:
        feature = torch.log(spec + 1e-5)

    feature = zscore_torch(feature)
    if feature.ndim == 2:
        feature = feature.unsqueeze(0)
    return feature.to(dtype=torch.float32)
