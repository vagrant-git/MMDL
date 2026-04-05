# 硕士论文模型结构图草稿

本文当前主模型为 `hcaf_audio_r18img_pq_xattn_5s`。下图使用 `Mermaid` 描述整体数据流，适合在 Markdown 中快速预览和持续修改。

## 1. Overall Architecture

```mermaid
flowchart LR
    A[Session-Level Input] --> A1[Audio Waveform 5 s]
    A --> A2[Pressure Signal 5 s]
    A --> A3[Flow Signal 5 s]

    A1 --> B1[HP80 + Mel Spectrogram + PCEN + Normalization]
    B1 --> C1[Audio Encoder<br/>ResNet18 Backbone]
    C1 --> D1[Audio Tokens + Audio Summary]

    A2 --> B2[Pressure Encoder<br/>1D CNN Stem + TCN]
    B2 --> C2[Pressure Tokens + Pressure Summary]

    A3 --> B3[Flow Encoder<br/>1D CNN Stem + TCN]
    B3 --> C3[Flow Tokens + Flow Summary]

    C2 --> E[Pressure-Flow Bidirectional Cross-Attention]
    C3 --> E
    E --> F[Sensor Fusion Gate]
    F --> G[Sensor Tokens + Sensor Representation]

    D1 --> H[Audio-Sensor Bidirectional Cross-Attention]
    G --> H
    H --> I[Joint Self-Attention]
    I --> J[Confidence-Aware Gate + Expert Residual]
    J --> K[Window-Level Logits]
    K --> L[0 / 2 / 4 Classification]
```

## 2. Compact Chinese Version

```mermaid
flowchart LR
    A[输入: 音频 压力 流量] --> B[统一 5 s 切窗]
    B --> C1[音频前端<br/>HP80 Mel PCEN]
    B --> C2[压力编码器<br/>1D CNN + TCN]
    B --> C3[流量编码器<br/>1D CNN + TCN]

    C1 --> D1[ResNet18 音频特征]
    C2 --> D2[Pressure token]
    C3 --> D3[Flow token]

    D2 --> E[PQ 双向 Cross-Attention]
    D3 --> E
    E --> F[传感器门控融合]

    D1 --> G[Audio-Sensor Cross-Attention]
    F --> G
    G --> H[联合 Self-Attention]
    H --> I[置信度感知融合 + Expert Residual]
    I --> J[三分类输出]
```

## 3. Notes

- 如果你的 Markdown 预览器支持 `Mermaid`，可以直接渲染。
- 如果后面要投稿或放进论文终稿，建议再转成 `TikZ`、`draw.io` 或导出为矢量图。
- 如果你希望，我下一步可以继续把这张图改成“带张量尺寸”的版本，例如标出 `[B,1,96,501]`、`[B,N,D]` 等。
