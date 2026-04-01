from __future__ import annotations

import torch
from torch import nn


class AudioCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_channels: int = 16,
        dropout: float = 0.3,
        in_channels: int = 1,
        **_: object,
    ) -> None:
        super().__init__()
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
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.encoder(batch["audio"])
        return self.head(features)
