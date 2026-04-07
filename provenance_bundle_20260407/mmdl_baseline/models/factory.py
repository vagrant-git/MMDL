from __future__ import annotations

from typing import Dict

from torch import nn

from .audio_cnn import AudioCNN
from .multimodal import AudioPressureFlowCNN, HCAFNet
from .sensor_cnn import PressureFlowCNN


def build_model(modality: str, config: Dict[str, object], num_classes: int) -> nn.Module:
    if modality == "audio_only":
        model_cfg = config["models"][modality]
        return AudioCNN(num_classes=num_classes, **model_cfg)
    if modality == "pressure_flow":
        model_cfg = config["models"][modality]
        return PressureFlowCNN(num_classes=num_classes, **model_cfg)
    if modality in {"multimodal", "multimodal_minus_audio", "multimodal_minus_pressure", "multimodal_minus_flow"}:
        model_cfg = dict(config["models"]["multimodal"])
        if modality == "multimodal_minus_audio":
            model_cfg = {**model_cfg, "enabled_modalities": ["pressure", "flow"]}
        elif modality == "multimodal_minus_pressure":
            model_cfg = {**model_cfg, "enabled_modalities": ["audio", "flow"]}
        elif modality == "multimodal_minus_flow":
            model_cfg = {**model_cfg, "enabled_modalities": ["audio", "pressure"]}
        model_type = str(model_cfg.pop("model_type", "legacy")).lower()
        if model_type == "hcaf_net":
            return HCAFNet(num_classes=num_classes, **model_cfg)
        return AudioPressureFlowCNN(num_classes=num_classes, **model_cfg)
    raise ValueError(f"Unsupported modality: {modality}")
