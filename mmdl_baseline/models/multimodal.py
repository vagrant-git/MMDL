from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

from .sensor_cnn import SensorEncoder1D, SensorTemporalEncoder, build_fusion_module


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.act(out)


def _make_resnet2d_stage(
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int,
    dropout: float,
) -> nn.Sequential:
    layers = [ResidualBlock2D(in_channels, out_channels, stride=stride, dropout=dropout)]
    for _ in range(1, num_blocks):
        layers.append(ResidualBlock2D(out_channels, out_channels, stride=1, dropout=dropout))
    return nn.Sequential(*layers)


def _audio_feature_dim(base_channels: int, encoder_type: str) -> int:
    return base_channels * 8 if encoder_type.lower() in {"resnet18", "resnet34"} else base_channels * 4


def _sensor_feature_dim(base_channels: int, encoder_type: str) -> int:
    return base_channels * 8 if encoder_type.lower() in {"resnet18", "resnet34", "hybrid_conformer", "res_tcn"} else base_channels * 4


def _build_torchvision_audio_resnet(encoder_type: str, in_channels: int, pretrained: bool) -> tuple[nn.Sequential, int]:
    encoder_type = encoder_type.lower()
    if encoder_type == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        hidden = 512
    elif encoder_type == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet34(weights=weights)
        hidden = 512
    else:
        raise ValueError(f"Unsupported torchvision audio encoder_type: {encoder_type}")

    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        if pretrained:
            if in_channels == 1:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                repeat_factor = (in_channels + old_conv.weight.shape[1] - 1) // old_conv.weight.shape[1]
                expanded = old_conv.weight.repeat(1, repeat_factor, 1, 1)[:, :in_channels]
                new_conv.weight.copy_(expanded / repeat_factor)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    model.conv1 = new_conv
    encoder = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )
    return encoder, hidden


class AudioEncoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 16,
        in_channels: int = 1,
        encoder_type: str = "basic",
        dropout: float = 0.0,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        encoder_type = encoder_type.lower()
        hidden = _audio_feature_dim(base_channels, encoder_type)
        if encoder_type == "basic":
            channels = [in_channels, base_channels, base_channels * 2, base_channels * 4]
            layers = []
            for in_ch, out_ch in zip(channels[:-1], channels[1:]):
                layers.extend(
                    [
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2),
                    ]
                )
            self.encoder = nn.Sequential(*layers)
        elif encoder_type in {"resnet18", "resnet34"}:
            self.encoder, hidden = _build_torchvision_audio_resnet(
                encoder_type=encoder_type,
                in_channels=in_channels,
                pretrained=pretrained,
            )
        else:
            raise ValueError(f"Unsupported audio encoder_type: {encoder_type}")
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encoder(x))


class AudioTokenEncoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 16,
        embedding_dim: int = 64,
        token_frames: int = 16,
        in_channels: int = 1,
        encoder_type: str = "basic",
        dropout: float = 0.0,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        encoder_type = encoder_type.lower()
        hidden = _audio_feature_dim(base_channels, encoder_type)
        if encoder_type == "basic":
            channels = [in_channels, base_channels, base_channels * 2, hidden]
            layers = []
            for in_ch, out_ch in zip(channels[:-1], channels[1:]):
                layers.extend(
                    [
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.GELU(),
                        nn.MaxPool2d(kernel_size=2),
                    ]
                )
            self.encoder = nn.Sequential(*layers)
        elif encoder_type in {"resnet18", "resnet34"}:
            self.encoder, hidden = _build_torchvision_audio_resnet(
                encoder_type=encoder_type,
                in_channels=in_channels,
                pretrained=pretrained,
            )
        else:
            raise ValueError(f"Unsupported audio token encoder_type: {encoder_type}")
        self.temporal_pool = nn.AdaptiveAvgPool2d((1, token_frames))
        self.token_proj = nn.Conv1d(hidden, embedding_dim, kernel_size=1)
        self.summary = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden, embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        pooled = self.temporal_pool(feat).squeeze(2)
        tokens = self.token_proj(pooled).transpose(1, 2)
        summary = self.summary(feat)
        return tokens, summary


class FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3, expansion: int = 4) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForwardBlock(dim, dropout=dropout)
        self.last_attn_weights: torch.Tensor | None = None

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attended, attn_weights = self.attn(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attn_weights = attn_weights.detach()
        query = query + self.dropout(attended)
        return self.ffn(query)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForwardBlock(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(self.norm(x), self.norm(x), self.norm(x))
        x = x + self.dropout(attended)
        return self.ffn(x)


class MaskedTokenGate(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        left_mask: torch.Tensor,
        right_mask: torch.Tensor,
    ) -> torch.Tensor:
        both = left_mask * right_mask
        left_only = left_mask * (1.0 - right_mask)
        right_only = right_mask * (1.0 - left_mask)
        fused = torch.zeros_like(left)
        if torch.any(both > 0):
            gate = self.gate(torch.cat([left, right], dim=-1))
            fused = fused + both * (gate * left + (1.0 - gate) * right)
        fused = fused + left_only * left + right_only * right
        availability = torch.clamp(left_mask + right_mask, max=1.0)
        return self.out(fused) * availability


class ReliabilityGate(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.weight = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2),
        )
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.last_weights: torch.Tensor | None = None

    def forward(
        self,
        audio_repr: torch.Tensor,
        sensor_repr: torch.Tensor,
        audio_mask: torch.Tensor,
        sensor_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_logits = self.weight(torch.cat([audio_repr, sensor_repr], dim=-1))
        availability = torch.cat([audio_mask, sensor_mask], dim=-1)
        masked_logits = raw_logits.masked_fill(availability <= 0, -1e4)
        weights = torch.softmax(masked_logits, dim=-1)
        self.last_weights = weights.detach()
        fused = weights[:, :1] * audio_repr + weights[:, 1:] * sensor_repr
        fused = self.out(fused) * torch.clamp(audio_mask + sensor_mask, max=1.0)
        return fused, weights


class ConfidenceAwareReliabilityGate(nn.Module):
    def __init__(self, dim: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        confidence_dim = 6
        self.weight = nn.Sequential(
            nn.LayerNorm(dim * 2 + confidence_dim),
            nn.Linear(dim * 2 + confidence_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2),
        )
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.num_classes = num_classes
        self.last_weights: torch.Tensor | None = None
        self.last_audio_conf: torch.Tensor | None = None
        self.last_sensor_conf: torch.Tensor | None = None

    def _confidence_features(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, k=min(2, self.num_classes), dim=-1).values
        top1 = topk[:, :1]
        if self.num_classes > 1:
            margin = top1 - topk[:, 1:2]
        else:
            margin = top1
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        entropy = entropy / math.log(max(self.num_classes, 2))
        confidence = 1.0 - entropy
        return torch.cat([top1, margin, confidence], dim=-1)

    def forward(
        self,
        audio_repr: torch.Tensor,
        sensor_repr: torch.Tensor,
        audio_logits: torch.Tensor,
        sensor_logits: torch.Tensor,
        audio_mask: torch.Tensor,
        sensor_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_conf = self._confidence_features(audio_logits)
        sensor_conf = self._confidence_features(sensor_logits)
        raw_logits = self.weight(torch.cat([audio_repr, sensor_repr, audio_conf, sensor_conf], dim=-1))
        availability = torch.cat([audio_mask, sensor_mask], dim=-1)
        masked_logits = raw_logits.masked_fill(availability <= 0, -1e4)
        weights = torch.softmax(masked_logits, dim=-1)
        self.last_weights = weights.detach()
        self.last_audio_conf = audio_conf.detach()
        self.last_sensor_conf = sensor_conf.detach()
        fused = weights[:, :1] * audio_repr + weights[:, 1:] * sensor_repr
        fused = self.out(fused) * torch.clamp(audio_mask + sensor_mask, max=1.0)
        return fused, weights


class HCAFNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        audio_base_channels: int = 16,
        sensor_base_channels: int = 16,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.3,
        embedding_dim: int = 128,
        tcn_layers: int = 2,
        num_heads: int = 4,
        modality_dropout: float = 0.0,
        enabled_modalities: Iterable[str] = ("audio", "pressure", "flow"),
        audio_token_frames: int = 12,
        sensor_token_length: int = 16,
        self_attention_layers: int = 1,
        audio_in_channels: int = 1,
        legacy_shared_sensor_norm: bool = False,
        confidence_aware_gate: bool = False,
        expert_residual_scale: float = 0.0,
        use_pq_cross_attention: bool = True,
        use_summary_in_repr: bool = True,
        use_summary_token_in_attention: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        extra_cfg = dict(_)
        self.modality_dropout = modality_dropout
        self.enabled_modalities = set(enabled_modalities)
        self.legacy_shared_sensor_norm = legacy_shared_sensor_norm
        self.confidence_aware_gate = confidence_aware_gate
        self.expert_residual_scale = expert_residual_scale
        self.use_pq_cross_attention = use_pq_cross_attention
        self.use_summary_in_repr = use_summary_in_repr
        self.use_summary_token_in_attention = use_summary_token_in_attention
        self.num_classes = num_classes
        self.capture_debug = False
        self.last_debug: dict[str, torch.Tensor] = {}

        audio_encoder_type = str(extra_cfg.get("audio_encoder_type", "basic"))
        audio_pretrained = bool(extra_cfg.get("audio_pretrained", False))
        sensor_encoder_type = str(extra_cfg.get("sensor_encoder_type", "tcn"))
        self.audio_sensor_interaction = str(extra_cfg.get("audio_sensor_interaction", "cross_attention")).lower()
        sensor_transformer_layers = int(extra_cfg.get("sensor_transformer_layers", 2))
        sensor_transformer_heads = int(extra_cfg.get("sensor_transformer_heads", num_heads))
        sensor_transformer_dropout = extra_cfg.get("sensor_transformer_dropout")
        self.audio_encoder = AudioTokenEncoder(
            base_channels=audio_base_channels,
            embedding_dim=embedding_dim,
            token_frames=audio_token_frames,
            in_channels=audio_in_channels,
            encoder_type=audio_encoder_type,
            dropout=dropout,
            pretrained=audio_pretrained,
        )
        self.pressure_encoder = SensorTemporalEncoder(
            base_channels=sensor_base_channels,
            embedding_dim=embedding_dim,
            tcn_layers=tcn_layers,
            dropout=dropout,
            token_length=sensor_token_length,
            encoder_type=sensor_encoder_type,
            transformer_layers=sensor_transformer_layers,
            transformer_heads=sensor_transformer_heads,
            transformer_dropout=None if sensor_transformer_dropout is None else float(sensor_transformer_dropout),
        )
        self.flow_encoder = SensorTemporalEncoder(
            base_channels=sensor_base_channels,
            embedding_dim=embedding_dim,
            tcn_layers=tcn_layers,
            dropout=dropout,
            token_length=sensor_token_length,
            encoder_type=sensor_encoder_type,
            transformer_layers=sensor_transformer_layers,
            transformer_heads=sensor_transformer_heads,
            transformer_dropout=None if sensor_transformer_dropout is None else float(sensor_transformer_dropout),
        )

        self.pressure_to_flow = CrossAttentionBlock(embedding_dim, num_heads=num_heads, dropout=dropout)
        self.flow_to_pressure = CrossAttentionBlock(embedding_dim, num_heads=num_heads, dropout=dropout)
        self.sensor_token_fusion = MaskedTokenGate(embedding_dim, dropout=dropout)
        self.sensor_repr_fusion = MaskedTokenGate(embedding_dim, dropout=dropout)

        self.audio_to_sensor = CrossAttentionBlock(embedding_dim, num_heads=num_heads, dropout=dropout)
        self.sensor_to_audio = CrossAttentionBlock(embedding_dim, num_heads=num_heads, dropout=dropout)
        self.self_attention = nn.ModuleList(
            [SelfAttentionBlock(embedding_dim, num_heads=num_heads, dropout=dropout) for _ in range(self_attention_layers)]
        )

        self.audio_repr_norm = nn.LayerNorm(embedding_dim)
        self.pressure_repr_norm = nn.LayerNorm(embedding_dim)
        self.flow_repr_norm = nn.LayerNorm(embedding_dim)
        self.sensor_repr_norm = nn.LayerNorm(embedding_dim)
        if self.audio_sensor_interaction == "direct_concat":
            self.reliability_gate = None
            self.audio_expert = None
            self.sensor_expert = None
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim * 2, fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_dim, num_classes),
            )
        else:
            self.use_expert_heads = confidence_aware_gate or expert_residual_scale > 0
            if confidence_aware_gate:
                self.reliability_gate = ConfidenceAwareReliabilityGate(embedding_dim, num_classes=num_classes, dropout=dropout)
            else:
                self.reliability_gate = ReliabilityGate(embedding_dim, dropout=dropout)
            self.audio_expert = nn.Linear(embedding_dim, num_classes) if self.use_expert_heads else None
            self.sensor_expert = nn.Linear(embedding_dim, num_classes) if self.use_expert_heads else None
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_dim, num_classes),
            )

    def _sample_modality_masks(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        masks = {}
        active_modalities = [name for name in ("audio", "pressure", "flow") if name in self.enabled_modalities]
        for name in ("audio", "pressure", "flow"):
            value = 1.0 if name in self.enabled_modalities else 0.0
            masks[name] = torch.full((batch_size, 1), value, device=device)
        if not self.training or self.modality_dropout <= 0.0 or len(active_modalities) <= 1:
            return masks

        keep_prob = 1.0 - self.modality_dropout
        for name in active_modalities:
            keep = (torch.rand(batch_size, 1, device=device) < keep_prob).float()
            masks[name] = masks[name] * keep

        total_active = sum(masks[name] for name in active_modalities)
        empty_rows = (total_active == 0).squeeze(1)
        if torch.any(empty_rows):
            restore_name = active_modalities[0]
            masks[restore_name][empty_rows] = 1.0
        return masks

    @staticmethod
    def _expand_mask(mask: torch.Tensor, ndim: int) -> torch.Tensor:
        if ndim == 2:
            return mask
        return mask.unsqueeze(1)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = tokens.mean(dim=1)
        return pooled * mask

    def _repr_from_tokens(
        self,
        tokens: torch.Tensor,
        summary: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_summary_token_in_attention:
            repr_base = tokens[:, 0]
            if self.use_summary_in_repr:
                repr_base = repr_base + summary
            return repr_base * mask

        repr_base = self._masked_mean(tokens, mask)
        if self.use_summary_in_repr:
            repr_base = repr_base + summary
        return repr_base * mask

    def _apply_mask(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return tensor * self._expand_mask(mask, tensor.dim())

    def _sensor_branch_norm(self, branch: str) -> nn.LayerNorm:
        if self.legacy_shared_sensor_norm:
            return self.audio_repr_norm
        if branch == "pressure":
            return self.pressure_repr_norm
        return self.flow_repr_norm

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        audio_tokens, audio_summary = self.audio_encoder(batch["audio"])
        pressure_tokens, pressure_summary = self.pressure_encoder(batch["pressure"])
        flow_tokens, flow_summary = self.flow_encoder(batch["flow"])

        batch_size = audio_tokens.shape[0]
        modality_masks = self._sample_modality_masks(batch_size, audio_tokens.device)
        audio_mask = modality_masks["audio"]
        pressure_mask = modality_masks["pressure"]
        flow_mask = modality_masks["flow"]
        sensor_mask = torch.clamp(pressure_mask + flow_mask, max=1.0)

        audio_tokens = self._apply_mask(audio_tokens, audio_mask)
        audio_summary = self._apply_mask(audio_summary, audio_mask)
        pressure_tokens = self._apply_mask(pressure_tokens, pressure_mask)
        pressure_summary = self._apply_mask(pressure_summary, pressure_mask)
        flow_tokens = self._apply_mask(flow_tokens, flow_mask)
        flow_summary = self._apply_mask(flow_summary, flow_mask)

        if self.use_summary_token_in_attention:
            audio_tokens = torch.cat([audio_summary.unsqueeze(1), audio_tokens], dim=1)
            pressure_tokens = torch.cat([pressure_summary.unsqueeze(1), pressure_tokens], dim=1)
            flow_tokens = torch.cat([flow_summary.unsqueeze(1), flow_tokens], dim=1)

        if self.use_pq_cross_attention:
            pressure_cross = self.pressure_to_flow(pressure_tokens, flow_tokens)
            flow_cross = self.flow_to_pressure(flow_tokens, pressure_tokens)
            pressure_tokens = pressure_mask.unsqueeze(1) * (
                flow_mask.unsqueeze(1) * pressure_cross + (1.0 - flow_mask.unsqueeze(1)) * pressure_tokens
            )
            flow_tokens = flow_mask.unsqueeze(1) * (
                pressure_mask.unsqueeze(1) * flow_cross + (1.0 - pressure_mask.unsqueeze(1)) * flow_tokens
            )

        pressure_repr = self._sensor_branch_norm("pressure")(self._repr_from_tokens(pressure_tokens, pressure_summary, pressure_mask))
        flow_repr = self._sensor_branch_norm("flow")(self._repr_from_tokens(flow_tokens, flow_summary, flow_mask))
        sensor_tokens = self.sensor_token_fusion(
            pressure_tokens,
            flow_tokens,
            pressure_mask.unsqueeze(1),
            flow_mask.unsqueeze(1),
        )
        sensor_repr = self.sensor_repr_fusion(pressure_repr, flow_repr, pressure_mask, flow_mask)

        if self.audio_sensor_interaction == "direct_concat":
            audio_repr = self.audio_repr_norm(self._repr_from_tokens(audio_tokens, audio_summary, audio_mask))
            sensor_repr = self.sensor_repr_norm(self._repr_from_tokens(sensor_tokens, sensor_repr, sensor_mask))
            logits = self.classifier(torch.cat([audio_repr, sensor_repr], dim=-1))
            weights = None
            audio_logits = None
            sensor_logits = None
        else:
            audio_cross = self.audio_to_sensor(audio_tokens, sensor_tokens)
            sensor_cross = self.sensor_to_audio(sensor_tokens, audio_tokens)
            audio_tokens = audio_mask.unsqueeze(1) * (
                sensor_mask.unsqueeze(1) * audio_cross + (1.0 - sensor_mask.unsqueeze(1)) * audio_tokens
            )
            sensor_tokens = sensor_mask.unsqueeze(1) * (
                audio_mask.unsqueeze(1) * sensor_cross + (1.0 - audio_mask.unsqueeze(1)) * sensor_tokens
            )

            joint_tokens = torch.cat([audio_tokens, sensor_tokens], dim=1)
            for block in self.self_attention:
                joint_tokens = block(joint_tokens)
            audio_length = audio_tokens.shape[1]
            audio_tokens = joint_tokens[:, :audio_length]
            sensor_tokens = joint_tokens[:, audio_length:]

            audio_repr = self.audio_repr_norm(self._repr_from_tokens(audio_tokens, audio_summary, audio_mask))
            sensor_repr = self.sensor_repr_norm(self._repr_from_tokens(sensor_tokens, sensor_repr, sensor_mask))
            audio_logits = self.audio_expert(audio_repr) if self.audio_expert is not None else None
            sensor_logits = self.sensor_expert(sensor_repr) if self.sensor_expert is not None else None
            if self.confidence_aware_gate:
                assert audio_logits is not None and sensor_logits is not None
                fused_repr, weights = self.reliability_gate(
                    audio_repr,
                    sensor_repr,
                    audio_logits,
                    sensor_logits,
                    audio_mask,
                    sensor_mask,
                )
            else:
                fused_repr, weights = self.reliability_gate(audio_repr, sensor_repr, audio_mask, sensor_mask)
            logits = self.classifier(fused_repr)
            if self.expert_residual_scale > 0:
                assert audio_logits is not None and sensor_logits is not None
                expert_logits = weights[:, :1] * audio_logits + weights[:, 1:] * sensor_logits
                logits = logits + self.expert_residual_scale * expert_logits
        if self.capture_debug:
            debug_tensors = {
                "audio_mask": audio_mask,
                "pressure_mask": pressure_mask,
                "flow_mask": flow_mask,
                "sensor_mask": sensor_mask,
                "audio_logits": audio_logits,
                "sensor_logits": sensor_logits,
                "audio_to_sensor_attn": self.audio_to_sensor.last_attn_weights,
                "sensor_to_audio_attn": self.sensor_to_audio.last_attn_weights,
                "pressure_to_flow_attn": self.pressure_to_flow.last_attn_weights,
                "flow_to_pressure_attn": self.flow_to_pressure.last_attn_weights,
            }
            if weights is not None:
                debug_tensors["audio_gate_weight"] = weights[:, :1]
                debug_tensors["sensor_gate_weight"] = weights[:, 1:]
            if self.confidence_aware_gate and self.reliability_gate is not None:
                debug_tensors["audio_conf_features"] = self.reliability_gate.last_audio_conf
                debug_tensors["sensor_conf_features"] = self.reliability_gate.last_sensor_conf
            self.last_debug = {name: value.detach() for name, value in debug_tensors.items() if value is not None}
        return logits


class AudioPressureFlowCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        audio_base_channels: int = 16,
        sensor_base_channels: int = 16,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.3,
        architecture: str = "basic",
        embedding_dim: int = 64,
        tcn_layers: int = 3,
        num_heads: int = 4,
        modality_dropout: float = 0.0,
        enabled_modalities: Iterable[str] = ("audio", "pressure", "flow"),
        sensor_fusion_method: str = "gated",
        final_fusion_method: str = "gated",
        audio_in_channels: int = 1,
        **_: object,
    ) -> None:
        super().__init__()
        extra_cfg = dict(_)
        self.architecture = architecture
        self.modality_dropout = modality_dropout
        self.enabled_modalities = set(enabled_modalities)
        audio_encoder_type = str(extra_cfg.get("audio_encoder_type", "basic"))
        audio_pretrained = bool(extra_cfg.get("audio_pretrained", False))
        default_sensor_encoder = "tcn" if architecture == "advanced" else "basic"
        sensor_encoder_type = str(extra_cfg.get("sensor_encoder_type", default_sensor_encoder))
        sensor_transformer_layers = int(extra_cfg.get("sensor_transformer_layers", 2))
        sensor_transformer_heads = int(extra_cfg.get("sensor_transformer_heads", num_heads))
        sensor_transformer_dropout = extra_cfg.get("sensor_transformer_dropout")

        if architecture == "advanced":
            self.audio_encoder = AudioTokenEncoder(
                base_channels=audio_base_channels,
                embedding_dim=embedding_dim,
                in_channels=audio_in_channels,
                encoder_type=audio_encoder_type,
                dropout=dropout,
                pretrained=audio_pretrained,
            )
            self.pressure_encoder = SensorTemporalEncoder(
                base_channels=sensor_base_channels,
                embedding_dim=embedding_dim,
                tcn_layers=tcn_layers,
                dropout=dropout,
                encoder_type=sensor_encoder_type,
                transformer_layers=sensor_transformer_layers,
                transformer_heads=sensor_transformer_heads,
                transformer_dropout=None if sensor_transformer_dropout is None else float(sensor_transformer_dropout),
            )
            self.flow_encoder = SensorTemporalEncoder(
                base_channels=sensor_base_channels,
                embedding_dim=embedding_dim,
                tcn_layers=tcn_layers,
                dropout=dropout,
                encoder_type=sensor_encoder_type,
                transformer_layers=sensor_transformer_layers,
                transformer_heads=sensor_transformer_heads,
                transformer_dropout=None if sensor_transformer_dropout is None else float(sensor_transformer_dropout),
            )
            self.p_to_f = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
            self.f_to_p = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
            self.sensor_fusion = build_fusion_module(sensor_fusion_method, embedding_dim, dropout=dropout)
            self.a_to_s = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
            self.s_to_a = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
            self.final_fusion = build_fusion_module(final_fusion_method, embedding_dim, dropout=dropout)
            feature_dim = embedding_dim
        else:
            self.audio_encoder = AudioEncoder(
                base_channels=audio_base_channels,
                in_channels=audio_in_channels,
                encoder_type=audio_encoder_type,
                dropout=dropout,
                pretrained=audio_pretrained,
            )
            self.pressure_encoder = SensorEncoder1D(
                base_channels=sensor_base_channels,
                encoder_type=sensor_encoder_type,
                dropout=dropout,
            )
            self.flow_encoder = SensorEncoder1D(
                base_channels=sensor_base_channels,
                encoder_type=sensor_encoder_type,
                dropout=dropout,
            )
            feature_dim = _audio_feature_dim(audio_base_channels, audio_encoder_type) + _sensor_feature_dim(
                sensor_base_channels,
                sensor_encoder_type,
            ) * 2

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def _apply_modality_masks(self, feature_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        available = {name: tensor for name, tensor in feature_map.items()}
        for name in list(available):
            if name not in self.enabled_modalities:
                available[name] = torch.zeros_like(available[name])
        if self.training and self.modality_dropout > 0:
            active = [name for name in available if name in self.enabled_modalities]
            if len(active) > 1:
                for name in active:
                    if torch.rand(1, device=available[name].device).item() < self.modality_dropout:
                        available[name] = torch.zeros_like(available[name])
        return available

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.architecture == "advanced":
            audio_tokens, audio_summary = self.audio_encoder(batch["audio"])
            pressure_tokens, pressure_summary = self.pressure_encoder(batch["pressure"])
            flow_tokens, flow_summary = self.flow_encoder(batch["flow"])

            pressure_attended, _ = self.p_to_f(pressure_tokens, flow_tokens, flow_tokens)
            flow_attended, _ = self.f_to_p(flow_tokens, pressure_tokens, pressure_tokens)
            pressure_feat = pressure_attended.mean(dim=1) + pressure_summary
            flow_feat = flow_attended.mean(dim=1) + flow_summary
            masked_sensor = self._apply_modality_masks({"pressure": pressure_feat, "flow": flow_feat, "audio": audio_summary})
            sensor_feat = self.sensor_fusion(masked_sensor["pressure"], masked_sensor["flow"])

            sensor_tokens = (pressure_attended + flow_attended) / 2.0
            audio_tokens_att, _ = self.a_to_s(audio_tokens, sensor_tokens, sensor_tokens)
            sensor_tokens_att, _ = self.s_to_a(sensor_tokens, audio_tokens, audio_tokens)
            audio_feat = audio_tokens_att.mean(dim=1) + masked_sensor["audio"]
            sensor_feat = sensor_tokens_att.mean(dim=1) + sensor_feat
            fused = self.final_fusion(audio_feat, sensor_feat)
        else:
            audio_feat = self.audio_encoder(batch["audio"])
            pressure_feat = self.pressure_encoder(batch["pressure"])
            flow_feat = self.flow_encoder(batch["flow"])
            masked = self._apply_modality_masks({"audio": audio_feat, "pressure": pressure_feat, "flow": flow_feat})
            fused = torch.cat([masked["audio"], masked["pressure"], masked["flow"]], dim=1)
        return self.classifier(fused)
