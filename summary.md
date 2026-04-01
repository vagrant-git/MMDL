# Final Model Technical Summary

## 1. Model Identity

- final_model: `hcaf_confgate_residual_pcen96hp80_5s`
- task: `0 / 2 / 4` three-class classification
- modality: `audio + pressure + flow`
- final_selection_date: `2026-04-01`
- final_selection_rule: 在与 `PQ-only` 严格同 split 的 grouped CV 下，选取 session-level macro-F1 最高且后续补充搜索未能继续超过的模型

最终采用的不是“所有实验里最复杂的网络”，而是证据链最完整、最终指标最高、并且后续补充搜索没有再刷新的那一个：

| model | source | window macro-F1 | session macro-F1 |
| --- | --- | ---: | ---: |
| `pressure_flow_5s` | `summary-MMmodel/pq_vs_multimodal_check` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` |
| `hcaf_confgate_residual_5s` | `summary-MMmodel/pq_vs_multimodal_check` | `0.7760 ± 0.0972` | `0.8815 ± 0.0838` |
| `hcaf_confgate_residual_pcen96hp80_5s` | `summary-MMmodel/hcaf_confgate_improve_search` | `0.9207 ± 0.0261` | `0.9407 ± 0.0838` |

## 2. Data And Evaluation Protocol

- unit_of_split: `session`
- windowing: `5 s` non-overlapping windows
- window_hop_sec: `5.0`
- split: `1 repeat x 3 folds` grouped CV
- validation: 从 train sessions 内再划出 `25%` 做 early stopping
- seed: `20260330`
- excluded_session: `MMdata_265.10s_0322_224132_no_secretion`
- data_leakage_rule: 先按 session 划分，再切窗

### 2.1 Input Modalities

- audio
  - source: `audio.wav`
  - target sample rate: `16000 Hz`
  - multi-channel handling: 均值混合到单通道
- pressure
  - source: `daq.csv -> Pressure (cmH2O)`
  - target sample rate: `100 Hz`
- flow
  - source: `daq.csv -> Flowrate (L/min)`
  - target sample rate: `100 Hz`

### 2.2 Alignment And Window Construction

数据集会先把三条模态裁到共同有效时长：

```text
duration_sec = min(audio_duration, pressure_duration, flow_duration)
```

然后基于这段公共时长切 `5 s` 窗。这样同一窗口内的音频、pressure、flow 始终时间对齐，不会出现某一模态越界或多出尾部片段的情况。

## 3. Data Flow

```text
session
  -> load audio.wav / daq.csv
  -> audio resample + mono + z-score
  -> pressure z-score
  -> flow z-score
  -> truncate to common duration
  -> split into 5 s windows

for each window:
  audio window
    -> high-pass 80 Hz
    -> Mel spectrogram (96 bins)
    -> PCEN
    -> feature standardization
    -> AudioTokenEncoder
  pressure window
    -> 1 x 500 waveform
    -> SensorTemporalEncoder
  flow window
    -> 1 x 500 waveform
    -> SensorTemporalEncoder

three encoded branches
  -> pressure-flow cross-attention
  -> sensor token/repr gated fusion
  -> audio-sensor cross-attention
  -> lightweight self-attention
  -> audio expert + sensor expert
  -> confidence-aware reliability gate
  -> main classifier
  -> expert-logit residual add-back
  -> logits / probabilities
  -> session aggregation by majority voting
```

## 4. Audio Frontend

最终模型真正刷新的关键不在更大的 encoder，而在音频前端：

- feature_type: `pcen`
- n_mels: `96`
- f_min: `80.0`
- f_max: `6000.0`
- highpass_hz: `80.0`
- n_fft: `1024`
- win_length: `400`
- hop_length: `160`

### 4.1 Why `PCEN96 + HP80`

- `PCEN96 + HP80` 把 session macro-F1 从 `0.8815` 提升到 `0.9407`
- 仅做 `LP300` 会回落到 `0.8815`
- `BP80-300` 进一步回落到 `0.8259`

这说明：

- `80 Hz` 以下的超低频成分更像漂移、基线扰动或接触噪声，去掉有帮助
- `300 Hz` 以上并不是纯噪声，仍包含对分类有用的信息

## 5. Architecture

模型主体是 `HCAFNet`，核心实现位于 `mmdl_baseline/models/multimodal.py`。

### 5.1 Branch Encoders

- audio branch
  - `AudioTokenEncoder`
  - encoder type: `basic`
  - 主干共有 `3` 个卷积块
  - 输出 `audio_token_frames=12` 个 token
  - 同时输出一个 summary vector
- pressure branch
  - `SensorTemporalEncoder`
  - encoder type: `tcn`
  - 由 `3` 层 1D CNN stem + `2` 层 TCN block 组成
  - 输出 `sensor_token_length=16` 个 token
  - 同时输出一个 summary vector
- flow branch
  - 与 pressure branch 同构

### 5.1.1 Audio Encoder Detail

当前最终模型的 audio encoder 不是 ResNet，而是 `AudioTokenEncoder(encoder_type="basic")`。输入是标准化后的音频时频图，shape 可以写成：

```text
[B, 1, 96, T]
```

其中：

- `B` 是 batch size
- `1` 是单通道时频图
- `96` 是 Mel bins
- `T` 是时间帧数，`5 s` 窗下由 `hop_length=160` 决定

其 backbone 共有 `3` 个 2D 卷积块，每个块的结构都相同：

```text
Conv2d(kernel=3x3, padding=1)
-> BatchNorm2d
-> GELU
-> MaxPool2d(kernel=2)
```

具体通道变化为：

1. Block 1
   - `1 -> 16`
2. Block 2
   - `16 -> 32`
3. Block 3
   - `32 -> 64`

所以 audio encoder 的主干可以概括成：

```text
[B, 1, 96, T]
-> Conv-BN-GELU-Pool
-> Conv-BN-GELU-Pool
-> Conv-BN-GELU-Pool
-> [B, 64, F', T']
```

在得到 backbone feature map 后，模型再分成两条支路：

1. token branch
   - `AdaptiveAvgPool2d((1, 12))`
   - 把频率维压成 `1`，把时间维压成固定 `12` 帧
   - 得到 `[B, 64, 1, 12]`
   - `squeeze` 后变成 `[B, 64, 12]`
   - 再过一个 `1x1 Conv1d(64 -> 128)` 做 token projection
   - 最终得到 audio tokens:

```text
[B, 12, 128]
```

2. summary branch
   - `AdaptiveAvgPool2d((1, 1))`
   - `Flatten`
   - `Linear(64 -> 128)`
   - `GELU`
   - 最终得到 audio summary:

```text
[B, 128]
```

因此，audio 分支最终输出的是：

- `12` 个 audio tokens
- `1` 个 `128` 维 audio summary vector

### 5.1.2 PQ Encoder Detail

pressure 和 flow 使用同构的 `SensorTemporalEncoder(encoder_type="tcn")`。每条传感器输入都是：

```text
[B, 1, 500]
```

因为：

- 传感器采样率是 `100 Hz`
- 窗长是 `5 s`
- 所以每窗长度为 `500` 个采样点

每个 PQ encoder 可以拆成两部分：

1. CNN stem
2. temporal backbone (`2` 层 TCN)

#### CNN stem

stem 一共有 `3` 层 1D 卷积：

1. Conv1d layer 1
   - `1 -> 16`
   - `kernel_size=9`
   - `stride=1`
   - `padding=4`
   - 后接 `BatchNorm1d + GELU`
2. Conv1d layer 2
   - `16 -> 32`
   - `kernel_size=5`
   - `stride=2`
   - `padding=2`
   - 后接 `BatchNorm1d + GELU`
3. Conv1d layer 3
   - `32 -> 64`
   - `kernel_size=5`
   - `stride=2`
   - `padding=2`
   - 后接 `BatchNorm1d + GELU`

所以 stem 的作用是：

- 先把单通道传感器波形映射到 `64` 个通道
- 再通过两次 stride=2 做时间降采样

得到的中间特征可写成：

```text
[B, 64, L']
```

#### TCN backbone

当前最终模型里 `tcn_layers=2`，所以 temporal backbone 有 `2` 个 TCN block：

1. TCN block 1
   - dilation=`1`
2. TCN block 2
   - dilation=`2`

每个 TCN block 内部都是：

```text
Conv1d(64 -> 64, kernel=3, dilation=d, padding=d)
-> BatchNorm1d
-> GELU
-> Dropout
-> Conv1d(64 -> 64, kernel=3, dilation=d, padding=d)
-> BatchNorm1d
-> GELU
-> residual add
-> Dropout
```

也就是说，单个传感器 encoder 的时序主干共有：

- `3` 层 stem Conv1d
- `2` 个 TCN block
- 每个 TCN block 含 `2` 层 dilated Conv1d

若按卷积层数来数，相当于每个 PQ encoder 总共有：

- stem `3` 层卷积
- TCN `4` 层卷积
- 合计 `7` 层一维卷积

在时序 backbone 之后，同样分成两条支路：

1. token branch
   - `AdaptiveAvgPool1d(16)`
   - 把时间维压成固定 `16` 个位置
   - `Conv1d(64 -> 128, kernel=1)`
   - 转置后得到：

```text
[B, 16, 128]
```

2. summary branch
   - `AdaptiveAvgPool1d(1)`
   - `Flatten`
   - `Linear(64 -> 128)`
   - `GELU`
   - 得到：

```text
[B, 128]
```

所以 pressure encoder 和 flow encoder 各自都会输出：

- `16` 个 sensor tokens
- `1` 个 `128` 维 summary vector

### 5.2 Fusion Order

先做 `pressure <-> flow`，再做 `audio <-> sensor`，不是三路直接拼接。

1. `pressure_to_flow` 与 `flow_to_pressure` cross-attention
2. `MaskedTokenGate` 融合 pressure/flow token 与 repr
3. `audio_to_sensor` 与 `sensor_to_audio` cross-attention
4. 拼接 joint tokens
5. `self_attention_layers=1` 做轻量 self-attention

### 5.3 Confidence-Aware Gate

最终门控不只看表征本身，还看两路 expert 的置信度特征：

- top-1 probability
- top-1 / top-2 margin
- normalized `1 - entropy`

门控输出两路权重：

```text
weights = [audio_weight, sensor_weight]
fused_repr = audio_weight * audio_repr + sensor_weight * sensor_repr
```

### 5.4 Expert Residual

主分类头先基于 `fused_repr` 给出 logits，然后再把两路 expert logits 按门控权重加权，并以 `0.3` 的比例回加：

```text
final_logits =
    classifier(fused_repr)
    + 0.3 * (audio_weight * audio_logits + sensor_weight * sensor_logits)
```

这是最终模型与普通 HCAF 的关键差异之一。

## 6. Structural Adjustments That Mattered

### 6.1 Sensor Normalization Fix

早期版本里 pressure / flow 分支会复用 audio 的 `LayerNorm`。修复后：

- `hcaf_legacy_sharednorm_5s`: window macro-F1=`0.7919`
- `hcaf_normfix_5s`: window macro-F1=`0.8728`

这一步先解决“表示空间本身不稳”的问题。

### 6.2 Confidence Gate Alone Is Not Enough

- `hcaf_confgate_5s`: session macro-F1=`0.6857`

单独上 confidence-aware gate 会显著退化，说明小数据下 gate 会过早放大单模态偏差。

### 6.3 Confidence Gate + Expert Residual Works

- `hcaf_confgate_residual_5s`: session macro-F1=`0.8815`

加入 expert residual 后，模型不再完全依赖单一门控决策，融合更稳定，也首次稳定超过 PQ-only。

### 6.4 Audio Frontend Upgrade Gives The Final Jump

- `hcaf_confgate_residual_base_5s`: session macro-F1=`0.8815`
- `hcaf_confgate_residual_pcen96hp80_5s`: session macro-F1=`0.9407`

最终刷新来自前端，而不是换大 backbone。

## 7. Training Hyperparameters

- optimizer: `Adam`
- learning_rate: `1e-3`
- weight_decay: `1e-4`
- epochs: `8`
- early_stop_patience: `3`
- batch_size: `16`
- weighted_sampler: `True`
- loss: `cross_entropy`
- grad_clip_norm: `3.0`
- dropout: `0.3`
- modality_dropout: `0.1`
- embedding_dim: `128`
- fusion_hidden_dim: `128`
- num_heads: `4`
- tcn_layers: `2`
- audio_token_frames: `12`
- sensor_token_length: `16`
- self_attention_layers: `1`

## 8. What Was Tried But Not Kept

### 8.1 Filter Variants

- `PCEN96 + LP300`: session macro-F1=`0.8815`
- `PCEN96 + BP80-300`: session macro-F1=`0.8259`

### 8.2 Architecture / Training Budget Variants

这些都没有超过当前 best：

- batch size `8`
- batch size `32`
- shorter attention tokens
- longer attention tokens + deeper self-attention
- audio/sensor `ResNet18`
- audio/sensor `ResNet34`

### 8.3 ResNet18 Transfer Learning

- `ResNet18 scratch`: session macro-F1=`0.8222`
- `ResNet18 + ImageNet init`: session macro-F1=`0.9407`

迁移学习有效，但只是追平当前 best，不是新的最佳方案。

### 8.4 2026-04-01 Additional Checks

为了继续验证还能否刷结果，额外跑了两轮聚焦实验：

- `modality_dropout=0.0`
  - fold1: window macro-F1=`0.8344`, session macro-F1=`0.8222`
  - 结论: 无法严格超过当前 best，提前停止
- `focal loss (gamma=1.5)`
  - fold1: window macro-F1=`0.8213`, session macro-F1=`0.8222`
  - 结论: 同样无法严格超过当前 best，提前停止

## 9. Interpretation Signals

来自 `summary-MMmodel/hcaf_confgate_interpretability/report.md` 的关键证据：

- audio gate 与 audio expert top-1 confidence 相关系数: `0.695`
- sensor gate 与 sensor expert top-1 confidence 相关系数: `0.373`
- 最主要窗口级混淆: `0 -> 2`
- 边界窗口错误率: `0.0878`
- 中段窗口错误率: `0.0563`

类别层面的 gate 倾向也很明显：

- class `0`: 平均 audio gate=`0.0446`
- class `2`: 平均 audio gate=`0.1090`
- class `4`: 平均 audio gate=`0.6218`

这意味着模型并不是机械地平均三模态，而是在不同类别上学出了不同的模态依赖模式。

## 10. Risks And Remaining Limits

- 数据量仍小，`3 folds` 下 session 数有限，单折波动仍然存在
- 主要错误仍集中在 `0 -> 2` 边界样本
- `5 s` 固定窗存在边界效应，缺少逐事件起止标注时无法进一步对齐
- 缺失模态鲁棒性分析使用的是另一份 split manifest，因此更适合作为鲁棒性证据，而不是主表替代
- 当前模型已经明显优于 PQ-only，但优势更像“稳定提升”而不是大幅碾压

## 11. Recommended Citation Form

可直接用于论文或答辩说明：

> 在固定 `5 s` 窗口、相同 session-level grouped CV 划分和相同训练预算下，最终多模态模型 `hcaf_confgate_residual_pcen96hp80_5s` 的 session-level macro-F1 达到 `0.9407 ± 0.0838`，高于 `pressure_flow-only` 的 `0.8519 ± 0.2095`。结果表明，多模态增益并不会由普通融合自动产生，而是依赖于 sensor normalization 修复、`confidence-aware gate + expert residual` 的稳定融合，以及 `PCEN96 + high-pass 80 Hz` 音频前端的共同作用。
