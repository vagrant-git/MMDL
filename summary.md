# 当前主模型技术说明

## 1. 模型身份

- current_model: `hcaf_audio_r18img_pq_xattn_5s`
- task: `0 / 2 / 4` 三分类
- modalities: `audio + pressure + flow`
- audio encoder: `ResNet18 (ImageNet init)`
- sensor encoder: `TCN`
- fusion: `PQ + audio cross-attention`
- primary metric: `window macro-F1`
- main_config: `configs/hcaf_audioresnet_xattn_vs_concat.yaml`
- main_result_dir: `summary-MMmodel/hcaf_audioresnet_xattn_vs_concat`

当前默认主模型不再使用旧的 `basic audio encoder` 口径，而是切换到：

- 固定非对齐 `5 s` 窗
- `audio = ResNet18(ImageNet init)`
- `PQ + audio cross-attention`

原因是这条主线在窗口级指标上同时强于：

- `PQ-only`
- `audio-only`
- `direct concat PQ+audio`

## 2. 评估协议

### 2.1 为什么当前以 window-level 为主

当前记录级 `session` 有两个问题：

1. 时长很长
2. 长度不等

因此当前主线不再用 `session-level` 作为第一选择标准，而改用：

- `window macro-F1`

`session-level` 仍然保留，但主要作为辅助观察，而不是决定当前最终主模型的唯一依据。

### 2.2 数据与切分

- 数据根目录: `data/`
- 标签: `0 / 2 / 4`
- 切分单位: `session`
- grouped CV: `1 repeat x 3 folds`
- 主窗口长度: `5 s`
- hop length: `5 s`
- split manifest:
  - `summary-MMmodel/pq_vs_multimodal_check/split_manifest.json`

### 2.3 模态与采样率

- audio:
  - source: `audio.wav`
  - sample rate: `16000 Hz`
- pressure:
  - source: `daq.csv -> Pressure (cmH2O)`
  - sample rate: `100 Hz`
- flow:
  - source: `daq.csv -> Flowrate (L/min)`
  - sample rate: `100 Hz`

所有模态在切窗前先裁到共同可覆盖的最短时长。

## 3. 当前主结果

统一主对照如下：

| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `pressure_flow_5s` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` |
| `hcaf_audio_r18img_audio_only_5s` | `0.8709 ± 0.0722` | `0.9407 ± 0.0838` |
| `hcaf_audio_r18img_pq_directconcat_5s` | `0.7800 ± 0.1610` | `0.7852 ± 0.1923` |
| `hcaf_audio_r18img_pq_xattn_5s` | `0.9145 ± 0.0745` | `0.9407 ± 0.0838` |

窗口级差值：

- `xattn - audio_only = +0.0436`
- `xattn - PQ_only = +0.1646`
- `xattn - direct_concat = +0.1345`

因此当前可保留的主结论是：

> 在 `window-level` 指标上，`PQ + audio cross-attention` 已经强于 `PQ-only`、`audio-only` 与 `direct concat PQ+audio`。

![主结果对比](summary-MMmodel/hcaf_audioresnet_xattn_vs_concat/model_comparison.png)

## 4. 模型结构

### 4.1 整体数据流

```mermaid
flowchart LR
    A[session] --> B[load audio.wav]
    A --> C[load daq.csv]
    B --> D[mono + resample 16 kHz]
    C --> E[pressure / flow z-score]
    D --> F[HP80 + 96-mel + PCEN]
    F --> G[AudioTokenEncoder ResNet18]
    E --> H[Pressure SensorTemporalEncoder TCN]
    E --> I[Flow SensorTemporalEncoder TCN]
    H --> J[pressure-flow bidirectional cross-attention]
    I --> J
    J --> K[sensor token / repr fusion]
    G --> L[audio-sensor cross-attention]
    K --> L
    L --> M[self-attention]
    M --> N[confidence-aware gate + expert residual]
    N --> O[window logits]
```

### 4.2.1 PQ波形编码器结构

PQ 编码器对应 `SensorTemporalEncoder(encoder_type="tcn")`，pressure 与 flow 使用同构结构。

输入：

```text
[B, 1, 500]
```

因为：

- 传感器采样率 `100 Hz`
- 窗长 `5 s`
- 每窗 `500` 点

结构由两部分组成：

1. `1D CNN stem`
2. `TCN backbone`

`1D CNN stem`：

```text
Conv1d(1 -> 16, kernel=9, stride=1, padding=4)
-> BatchNorm1d
-> GELU
-> Conv1d(16 -> 32, kernel=5, stride=2, padding=2)
-> BatchNorm1d
-> GELU
-> Conv1d(32 -> 64, kernel=5, stride=2, padding=2)
-> BatchNorm1d
-> GELU
```

`TCN backbone`：

- `tcn_layers = 2`
- dilation 分别为 `1` 和 `2`

每条 PQ 分支最终输出：

- `16` 个 tokens
  - `[B, 16, 128]`
- `1` 个 summary vector
  - `[B, 128]`

### 4.2.2 呼吸音编码器结构

音频分支使用的是 `AudioTokenEncoder(encoder_type="resnet18", pretrained=True)`，即：

- `torchvision ResNet18`
- `ImageNet` 初始化
- 首层 `conv1` 改成单通道输入
- 若输入通道为 `1`，初始化权重由 RGB 通道均值转换得到

音频前端：

- `PCEN`
- `96 mel bins`
- `high-pass 80 Hz`

输入形状：

```text
[B, 1, 96, T]
```

ResNet18 主干部分为：

```text
conv1 -> bn1 -> relu -> maxpool
-> layer1 -> layer2 -> layer3 -> layer4
```

主干输出通道为 `512`，之后分成两条支路：

1. token branch
   - `AdaptiveAvgPool2d((1, 12))`
   - `Conv1d(512 -> 128, kernel=1)`
   - 输出：

```text
[B, 12, 128]
```

2. summary branch
   - `AdaptiveAvgPool2d((1, 1))`
   - `Linear(512 -> 128)`
   - `GELU`
   - 输出：

```text
[B, 128]
```

### 4.2.3 融合结构设计

当前主模型不是直接拼接三路特征，而是按层次融合：

1. `pressure -> flow` 与 `flow -> pressure` 双向 cross-attention
2. `sensor_token_fusion` / `sensor_repr_fusion`
3. `audio -> sensor` 与 `sensor -> audio` 双向 cross-attention
4. joint tokens 过 `self-attention`
5. 分类前再过 `confidence-aware gate + expert residual`

这一步是当前主模型与 `direct concat` 最大的区别。

`direct concat` 对照模型：

- 保留同样的 audio / PQ encoder
- 保留 pressure-flow 内部建模
- 去掉 audio-sensor cross-attention
- 改为直接拼接 `audio_repr` 和 `sensor_repr` 后分类

结果证明：

- 单纯共享同样的 encoder，不足以得到同样的效果
- 关键差异确实来自 cross-attention 交互本身

## 5. 为什么选 Cross-Attention 而不是 Direct Concat

正式三折结果：

- `direct concat`: `0.7800 ± 0.1610`
- `cross-attention`: `0.9145 ± 0.0745`

差值：

- `+0.1345`

这说明对当前任务来说：

- audio 与 PQ 之间不是简单互补关系
- 更合理的做法是先让两路 token 交互，再做分类

## 6. 为什么当前保留固定非对齐 5s

虽然我们后面补做了：

- `8 s`
- `10 s`
- `12 s`
- 周期对齐 `4 s / 8 s`
- 单周期 / 双周期归一化

但到目前为止，固定非对齐窗下最稳的正式主结果仍然是 `5 s`。

在 `repeat1_fold1` 的固定窗 smoke 里：

- `8 s xattn`: `0.8005`
- `10 s xattn`: `0.7920`
- `12 s xattn`: `0.8390`
- `5 s xattn`: `0.7965`（fold1），正式三折均值 `0.9145`

这说明：

- “不一定非得 5s”是合理怀疑
- 但当前已经完成的固定窗实验里，还没有看到更长固定窗稳定超过现有 `5 s` 主线

## 7. 尝试过但未保留的方向

当前主线已经试过但不作为默认文档口径保留的方向包括：

- 更复杂 PQ encoder
- 周期对齐固定窗
- 单周期 / 双周期归一化表示
- 提高 `modality_dropout`
- `direct concat PQ+audio`

其中：

- 周期级表示在生理上有道理
- 但在当前 `ResNet18` 路线上，还没有稳定带来更强的多模态优势

## 8. 当前限制

- `window-level` 已经满足主目标，但 `session-level` 上 `xattn` 与 `audio-only` 打平
- 长窗固定非对齐只做了 fold1 smoke，尚未全部补成完整三折
- 周期级表示已经实现，但当前还没有形成新的最佳主线
- `sensor` 分支仍可能是当前主要瓶颈

## 9. 当前默认引用表述

可直接用于当前版本说明：

> 在固定非对齐 `5 s` 窗、`audio encoder = ResNet18(ImageNet init)` 的设置下，`PQ + audio cross-attention` 模型 `hcaf_audio_r18img_pq_xattn_5s` 的 window-level macro-F1 达到 `0.9145 ± 0.0745`，高于 `audio-only` 的 `0.8709 ± 0.0722`、`pressure+flow-only` 的 `0.7499 ± 0.2513`，也高于 `direct concat PQ+audio` 的 `0.7800 ± 0.1610`。因此在当前窗口级判别目标下，`PQ + audio cross-attention` 是默认主模型。
