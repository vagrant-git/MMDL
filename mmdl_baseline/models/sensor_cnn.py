from __future__ import annotations

import math

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
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


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.net(x))


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.norm1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.dropout1(out)
        out = self.conv2(self.act2(self.norm2(out)))
        out = self.dropout2(out)
        return x + out


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class SensorConformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, expansion: int = 4, conv_kernel_size: int = 7) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(dim)
        self.conv_pw_in = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.conv_glu = nn.GLU(dim=1)
        self.conv_dw = nn.Conv1d(
            dim,
            dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.conv_bn = nn.BatchNorm1d(dim)
        self.conv_act = nn.SiLU()
        self.conv_pw_out = nn.Conv1d(dim, dim, kernel_size=1)
        self.conv_dropout = nn.Dropout(dropout)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        attn_input = self.attn_norm(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input)
        x = x + self.attn_dropout(attn_out)
        conv_input = self.conv_norm(x).transpose(1, 2)
        conv_out = self.conv_pw_in(conv_input)
        conv_out = self.conv_glu(conv_out)
        conv_out = self.conv_dw(conv_out)
        conv_out = self.conv_bn(conv_out)
        conv_out = self.conv_act(conv_out)
        conv_out = self.conv_pw_out(conv_out)
        conv_out = self.conv_dropout(conv_out).transpose(1, 2)
        x = x + conv_out
        x = x + 0.5 * self.ffn2(x)
        return self.out_norm(x)


class SensorEncoder1D(nn.Module):
    def __init__(self, base_channels: int = 16, encoder_type: str = "basic", dropout: float = 0.0) -> None:
        super().__init__()
        encoder_type = encoder_type.lower()
        if encoder_type == "basic":
            self.net = nn.Sequential(
                nn.Conv1d(1, base_channels, kernel_size=7, stride=1, padding=3),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(base_channels * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(base_channels * 4),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
        elif encoder_type in {"resnet18", "resnet34"}:
            blocks = [2, 2, 2, 2] if encoder_type == "resnet18" else [3, 4, 6, 3]
            self.net = nn.Sequential(
                nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(base_channels),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                _make_resnet1d_stage(base_channels, base_channels, blocks[0], stride=1, dropout=dropout),
                _make_resnet1d_stage(base_channels, base_channels * 2, blocks[1], stride=2, dropout=dropout),
                _make_resnet1d_stage(base_channels * 2, base_channels * 4, blocks[2], stride=2, dropout=dropout),
                _make_resnet1d_stage(base_channels * 4, base_channels * 8, blocks[3], stride=2, dropout=dropout),
                nn.AdaptiveAvgPool1d(1),
            )
        else:
            raise ValueError(f"Unsupported sensor encoder_type: {encoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _make_resnet1d_stage(
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int,
    dropout: float,
) -> nn.Sequential:
    layers = [ResidualBlock1D(in_channels, out_channels, stride=stride, dropout=dropout)]
    for _ in range(1, num_blocks):
        layers.append(ResidualBlock1D(out_channels, out_channels, stride=1, dropout=dropout))
    return nn.Sequential(*layers)


class SensorTemporalEncoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 16,
        embedding_dim: int = 64,
        tcn_layers: int = 3,
        dropout: float = 0.3,
        token_length: int = 16,
        encoder_type: str = "tcn",
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_dropout: float | None = None,
    ) -> None:
        super().__init__()
        encoder_type = encoder_type.lower()
        hidden = base_channels * 4
        attn_dropout = dropout if transformer_dropout is None else transformer_dropout
        if encoder_type in {"cnn", "tcn", "gru", "cnn_gru"}:
            hidden = base_channels * 4
            self.stem = nn.Sequential(
                nn.Conv1d(1, base_channels, kernel_size=9, padding=4),
                nn.BatchNorm1d(base_channels),
                nn.GELU(),
                nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2, stride=2),
                nn.BatchNorm1d(base_channels * 2),
                nn.GELU(),
                nn.Conv1d(base_channels * 2, hidden, kernel_size=5, padding=2, stride=2),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
            )
            if encoder_type == "cnn":
                self.temporal_backbone = nn.Identity()
            elif encoder_type == "tcn":
                self.temporal_backbone = nn.Sequential(
                    *[TCNBlock(hidden, dilation=2**i, dropout=dropout) for i in range(tcn_layers)]
                )
            elif encoder_type == "gru":
                self.temporal_backbone = nn.GRU(
                    input_size=hidden,
                    hidden_size=hidden // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            else:
                self.temporal_backbone = nn.Sequential(
                    *[TCNBlock(hidden, dilation=2**i, dropout=dropout) for i in range(max(1, tcn_layers))]
                )
                self.recurrent_backbone = nn.GRU(
                    input_size=hidden,
                    hidden_size=hidden // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
        elif encoder_type in {"resnet18", "resnet34"}:
            blocks = [2, 2, 2, 2] if encoder_type == "resnet18" else [3, 4, 6, 3]
            hidden = base_channels * 8
            self.stem = nn.Sequential(
                nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(base_channels),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            )
            self.temporal_backbone = nn.Sequential(
                _make_resnet1d_stage(base_channels, base_channels, blocks[0], stride=1, dropout=dropout),
                _make_resnet1d_stage(base_channels, base_channels * 2, blocks[1], stride=2, dropout=dropout),
                _make_resnet1d_stage(base_channels * 2, base_channels * 4, blocks[2], stride=2, dropout=dropout),
                _make_resnet1d_stage(base_channels * 4, hidden, blocks[3], stride=2, dropout=dropout),
            )
        elif encoder_type == "hybrid_conformer":
            hidden = base_channels * 8
            self.stem = nn.Sequential(
                nn.Conv1d(1, base_channels, kernel_size=9, padding=4, bias=False),
                nn.BatchNorm1d(base_channels),
                nn.GELU(),
                nn.Conv1d(base_channels, base_channels * 2, kernel_size=7, padding=3, stride=2, bias=False),
                nn.BatchNorm1d(base_channels * 2),
                nn.GELU(),
                nn.Conv1d(base_channels * 2, hidden, kernel_size=5, padding=2, stride=2, bias=False),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
            )
            dilations = [1, 2, 4, 8, 16, 32][: max(2, tcn_layers + 2)]
            self.temporal_backbone = nn.Sequential(
                *[ResidualTCNBlock(hidden, dilation=dilation, dropout=dropout) for dilation in dilations]
            )
            self.sequence_positional_encoding = SinusoidalPositionalEncoding(hidden)
            self.sequence_model = nn.Sequential(
                *[
                    SensorConformerBlock(
                        dim=hidden,
                        num_heads=transformer_heads,
                        dropout=attn_dropout,
                    )
                    for _ in range(max(1, transformer_layers))
                ]
            )
        elif encoder_type == "res_tcn":
            hidden = base_channels * 8
            self.stem = nn.Sequential(
                nn.Conv1d(1, base_channels, kernel_size=9, padding=4, bias=False),
                nn.BatchNorm1d(base_channels),
                nn.GELU(),
                nn.Conv1d(base_channels, base_channels * 2, kernel_size=7, padding=3, stride=2, bias=False),
                nn.BatchNorm1d(base_channels * 2),
                nn.GELU(),
                nn.Conv1d(base_channels * 2, hidden, kernel_size=5, padding=2, stride=2, bias=False),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
            )
            dilations = [1, 2, 4, 8, 16, 32][: max(3, tcn_layers + 3)]
            layers: list[nn.Module] = []
            for dilation in dilations:
                layers.append(ResidualTCNBlock(hidden, dilation=dilation, dropout=dropout))
                layers.append(SqueezeExcite1D(hidden))
            self.temporal_backbone = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported sensor temporal encoder_type: {encoder_type}")

        self.token_pool = nn.AdaptiveAvgPool1d(token_length)
        self.token_proj = nn.Conv1d(hidden, embedding_dim, kernel_size=1)
        self.summary_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, embedding_dim),
            nn.GELU(),
        )
        self.encoder_type = encoder_type

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stem_features = self.stem(x)
        if self.encoder_type == "gru":
            gru_features, _ = self.temporal_backbone(stem_features.transpose(1, 2))
            features = gru_features.transpose(1, 2)
        elif self.encoder_type == "cnn_gru":
            conv_features = self.temporal_backbone(stem_features)
            gru_features, _ = self.recurrent_backbone(conv_features.transpose(1, 2))
            features = gru_features.transpose(1, 2)
        elif self.encoder_type == "hybrid_conformer":
            conv_features = self.temporal_backbone(stem_features)
            seq_features = self.sequence_positional_encoding(conv_features.transpose(1, 2))
            features = self.sequence_model(seq_features).transpose(1, 2)
        elif self.encoder_type == "res_tcn":
            features = self.temporal_backbone(stem_features)
        else:
            features = self.temporal_backbone(stem_features)
        tokens = self.token_proj(self.token_pool(features)).transpose(1, 2)
        summary = self.summary_proj(features)
        return tokens, summary


class GatedFusion(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([left, right], dim=-1))
        fused = gate * left + (1.0 - gate) * right
        return self.out(fused)


class ConcatMLPFusion(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.skip = nn.Linear(dim * 2, dim)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.GELU())

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([left, right], dim=-1)
        return self.out(self.proj(merged) + self.skip(merged))


class SoftmaxGateFusion(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.weight = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2),
        )
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        logits = self.weight(torch.cat([left, right], dim=-1))
        weights = torch.softmax(logits, dim=-1)
        fused = weights[:, :1] * left + weights[:, 1:] * right
        return self.out(fused)


class ResidualAddFusion(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.out(left + right)


def build_fusion_module(method: str, dim: int, dropout: float = 0.3) -> nn.Module:
    if method == "gated":
        return GatedFusion(dim, dropout=dropout)
    if method == "concat_mlp":
        return ConcatMLPFusion(dim, dropout=dropout)
    if method == "softmax_gate":
        return SoftmaxGateFusion(dim, dropout=dropout)
    if method == "residual_add":
        return ResidualAddFusion(dim, dropout=dropout)
    raise ValueError(f"Unsupported fusion method: {method}")


def _sensor_feature_dim(base_channels: int, encoder_type: str) -> int:
    return base_channels * 8 if encoder_type.lower() in {"resnet18", "resnet34", "hybrid_conformer", "res_tcn"} else base_channels * 4


class PressureFlowCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_channels: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        architecture: str = "basic",
        embedding_dim: int = 64,
        tcn_layers: int = 3,
        num_heads: int = 4,
        fusion_method: str = "gated",
        sensor_encoder_type: str = "tcn",
    ) -> None:
        super().__init__()
        self.architecture = architecture
        if architecture == "advanced":
            self.pressure_encoder = SensorTemporalEncoder(
                base_channels=base_channels,
                embedding_dim=embedding_dim,
                tcn_layers=tcn_layers,
                dropout=dropout,
                encoder_type=sensor_encoder_type,
            )
            self.flow_encoder = SensorTemporalEncoder(
                base_channels=base_channels,
                embedding_dim=embedding_dim,
                tcn_layers=tcn_layers,
                dropout=dropout,
                encoder_type=sensor_encoder_type,
            )
            self.p_to_f = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
            self.f_to_p = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
            self.fusion = build_fusion_module(fusion_method, embedding_dim, dropout=dropout)
            feature_dim = embedding_dim
        else:
            self.pressure_encoder = SensorEncoder1D(
                base_channels=base_channels,
                encoder_type=sensor_encoder_type,
                dropout=dropout,
            )
            self.flow_encoder = SensorEncoder1D(
                base_channels=base_channels,
                encoder_type=sensor_encoder_type,
                dropout=dropout,
            )
            feature_dim = _sensor_feature_dim(base_channels, sensor_encoder_type) * 2
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.architecture == "advanced":
            pressure_tokens, pressure_summary = self.pressure_encoder(batch["pressure"])
            flow_tokens, flow_summary = self.flow_encoder(batch["flow"])
            pressure_attended, _ = self.p_to_f(pressure_tokens, flow_tokens, flow_tokens)
            flow_attended, _ = self.f_to_p(flow_tokens, pressure_tokens, pressure_tokens)
            pressure_feat = pressure_attended.mean(dim=1) + pressure_summary
            flow_feat = flow_attended.mean(dim=1) + flow_summary
            fused = self.fusion(pressure_feat, flow_feat)
        else:
            pressure_feat = self.pressure_encoder(batch["pressure"])
            flow_feat = self.flow_encoder(batch["flow"])
            fused = torch.cat([pressure_feat, flow_feat], dim=1)
        return self.classifier(fused)
