from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

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

from .discovery import SessionRecord


@dataclass
class WindowIndex:
    session_id: str
    label: int
    start_sec: float
    end_sec: float


class MultiModalWindowDataset(Dataset):
    def __init__(
        self,
        sessions: List[SessionRecord],
        config: Dict[str, object],
        modality: str,
    ) -> None:
        self.sessions = sessions
        self.config = config
        self.modality = modality
        self.audio_sr = int(config["audio_sample_rate"])
        self.sensor_sr = int(config["sensor_sample_rate"])
        self.window_sec = float(config["window_sec"])
        self.hop_sec = float(config["window_hop_sec"])
        self.pad_short = bool(config.get("pad_short_recording", False))
        self.window_indexes: List[WindowIndex] = []
        self.session_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.audio_frontend = resolve_audio_frontend_config(config)
        self.mel_transform = build_mel_transform(self.audio_sr, self.audio_frontend)
        self._build_window_index()

    def _needs_audio(self) -> bool:
        return self.modality in {"audio_only", "multimodal", "multimodal_minus_audio", "multimodal_minus_pressure", "multimodal_minus_flow"}

    def _needs_sensors(self) -> bool:
        return self.modality in {"pressure_flow", "multimodal", "multimodal_minus_audio", "multimodal_minus_pressure", "multimodal_minus_flow"}

    def _build_window_index(self) -> None:
        for session in self.sessions:
            total_sec = float(session.duration_sec)
            if total_sec < self.window_sec and not self.pad_short:
                continue
            if total_sec < self.window_sec and self.pad_short:
                self.window_indexes.append(
                    WindowIndex(
                        session_id=session.session_id,
                        label=session.label,
                        start_sec=0.0,
                        end_sec=self.window_sec,
                    )
                )
                continue
            start = 0.0
            while start + self.window_sec <= total_sec + 1e-6:
                self.window_indexes.append(
                    WindowIndex(
                        session_id=session.session_id,
                        label=session.label,
                        start_sec=float(start),
                        end_sec=float(start + self.window_sec),
                    )
                )
                start += self.hop_sec

    def _load_session(self, session: SessionRecord) -> Dict[str, torch.Tensor]:
        cached = self.session_cache.get(session.session_id)
        if cached is not None:
            return cached
        audio, audio_sr = load_audio(session.audio_path, self.audio_sr)
        daq = load_daq(session.daq_path)
        pressure = torch.from_numpy(zscore_np(daq["pressure"]))
        flow = torch.from_numpy(zscore_np(daq["flow"]))
        audio = zscore_torch(audio)
        duration_sec = min(audio.shape[0] / audio_sr, pressure.shape[0] / self.sensor_sr, flow.shape[0] / self.sensor_sr)
        common_audio_len = int(duration_sec * audio_sr)
        common_sensor_len = int(duration_sec * self.sensor_sr)
        cached = {
            "audio": audio[:common_audio_len],
            "pressure": pressure[:common_sensor_len],
            "flow": flow[:common_sensor_len],
        }
        self.session_cache[session.session_id] = cached
        return cached

    def __len__(self) -> int:
        return len(self.window_indexes)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.window_indexes[index]
        session = next(session for session in self.sessions if session.session_id == item.session_id)
        session_data = self._load_session(session)
        audio_length = int(self.window_sec * self.audio_sr)
        sensor_length = int(self.window_sec * self.sensor_sr)
        audio_start = int(item.start_sec * self.audio_sr)
        sensor_start = int(item.start_sec * self.sensor_sr)

        output: Dict[str, torch.Tensor] = {
            "label": torch.tensor(item.label, dtype=torch.long),
        }
        if self._needs_audio():
            audio_window = slice_or_pad_1d(session_data["audio"], audio_start, audio_length)
            output["audio"] = compute_audio_features(
                audio_window,
                self.audio_sr,
                self.mel_transform,
                self.audio_frontend,
            )
        if self._needs_sensors():
            pressure_window = slice_or_pad_1d(session_data["pressure"], sensor_start, sensor_length)
            flow_window = slice_or_pad_1d(session_data["flow"], sensor_start, sensor_length)
            output["pressure"] = pressure_window.unsqueeze(0)
            output["flow"] = flow_window.unsqueeze(0)
        output["session_id"] = item.session_id  # type: ignore[assignment]
        output["start_sec"] = torch.tensor(item.start_sec, dtype=torch.float32)
        return output
