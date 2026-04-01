# 第四章结果整理：以 `hcaf_confgate_residual_pcen96hp80_5s` 为主模型

## 1. 本章采用的结果口径

- 本章当前最终口径统一围绕最优多模态模型 `hcaf_confgate_residual_pcen96hp80_5s` 展开。
- 其中 `hcaf_confgate_residual_5s` 作为上一版 best，用于说明融合结构本身已经优于普通多模态；`hcaf_confgate_residual_pcen96hp80_5s` 则是在相同 HCAF 主干上进一步替换音频前端后得到的当前最终模型。
- 所有实验均采用 `0 / 2 / 4` 三分类任务，且遵循先按 `session` 分组、再切 `5 s` 窗口的 grouped CV 原则，避免窗口泄漏。
- 主结果、机制消融、缺失模态鲁棒性分别来自三组独立配置；引用时应按各自配置内的对比关系解读，不跨不同轮次混用数值。

## 2. 现有模型与多模态数据处理流程

本节用于说明当前项目里三类模态数据是如何被读取、预处理、切窗、编码并送入模型的。以下描述以代码实现为准，主要对应:

- `mmdl_baseline/preprocessing/signals.py`
- `mmdl_baseline/dataset/windowed_dataset.py`
- `mmdl_baseline/models/audio_cnn.py`
- `mmdl_baseline/models/sensor_cnn.py`
- `mmdl_baseline/models/multimodal.py`

### 2.1 原始数据组织与对齐方式

- 每个 `session` 对应一个独立采集目录，包含音频文件 `audio.wav`、传感器文件 `daq.csv` 和 `metadata.json`。
- 音频从 `wav` 中读取；若是多声道，先做均值混合为单声道。
- 传感器从 `daq.csv` 读取，使用两列连续信号:
  - `Pressure (cmH2O)`
  - `Flowrate (L/min)`
- `Time (s)` 会被平移到以 `0` 为起点，但当前模型实际使用的是 pressure 与 flow 两条数值序列本身。
- 为保证多模态对齐，数据集构建时会以三条序列共同可覆盖的最短时长作为有效时长:
  - `duration_sec = min(audio_duration, pressure_duration, flow_duration)`
- 之后所有窗都在这段共同时长上切分，因此不会出现音频和传感器窗口长度不一致的问题。

### 2.2 切窗策略

- 默认主实验使用固定不重叠滑窗:
  - `window_sec = 5.0`
  - `window_hop_sec = 5.0`
- 也就是说，一个 `session` 会被切成多个连续但不重叠的窗口，每个窗口都继承该 `session` 的标签。
- 若某个 `session` 的总长度短于设定窗长，且 `pad_short_recording = false`，该 `session` 在该实验下不会生成窗口。
- 当前第四章的绝大部分实验都使用 `pad_short_recording = false`，因此模型训练与测试都建立在“真实完整窗口”而非补零伪窗上。
- 所有数据划分先在 `session` 级别完成，再对 train / val / test 各自切窗，从而避免同一 `session` 的不同窗口泄漏到不同数据集。

### 2.3 音频模态如何处理

音频处理流程如下:

1. 读取 `audio.wav`
2. 若采样率不是目标值，则重采样到 `16000 Hz`
3. 若是双声道或多声道，则先做均值混合成单声道
4. 对整条波形做 z-score 标准化
5. 按当前设定窗长截取音频片段
6. 将音频片段转换为时频表示，再送入 2D CNN 编码器

上一版 HCAF 基础模型默认使用的音频前端是 `log-Mel 64`:

- `sample_rate = 16000`
- `n_fft = 1024`
- `win_length = 400`
- `hop_length = 160`
- `n_mels = 64`
- 压缩方式默认是 `log(spec + 1e-5)`

补充说明:

- 代码里还支持更复杂的前端选项，如 `PCEN`、预加重、带通滤波、Mel 频带裁剪与 delta 特征。
- 第四章最终采用的最优模型 `hcaf_confgate_residual_pcen96hp80_5s` 使用的是 `PCEN96 + high-pass 80 Hz`，而 `log-Mel 64` 主要对应上一版基线 HCAF。
- 生成 Mel 频谱后，会再做一次窗口内标准化:
  - `(feature - mean) / (std + 1e-6)`
- 因此音频分支实际输入给网络的是标准化后的 `1 x n_mels x time_frames` 张量。

### 2.4 Pressure 与 Flow 模态如何处理

pressure 和 flow 的处理路径是并行且对称的。

基本流程如下:

1. 从 `daq.csv` 分别读取 pressure 与 flow 两列
2. 各自对整条 recording 做 z-score 标准化
3. 按窗长截取固定长度片段
4. 每个窗以 `1 x T` 的一维波形形式输入网络

其中:

- pressure 保留原始物理量顺序，但经过 z-score，因此输入强调的是相对波动形态而不是绝对幅值
- flow 也采用同样的 z-score 处理
- 传感器目标采样率在配置中固定为 `100 Hz`
- 因此 `5 s` 窗下，每个 pressure / flow 窗长度都是 `500` 个采样点；`10 s` 窗对应 `1000` 个点，以此类推

### 2.5 单模态与双模态基线是怎么建模的

`audio_only` 基线:

- 输入是单通道音频时频图
- 使用 3 层轻量 2D CNN:
  - `Conv2d + BatchNorm2d + ReLU + MaxPool2d`
- 之后做 `AdaptiveAvgPool2d((1,1))`
- 最后接 dropout 与线性分类层输出类别 logits

`pressure_flow` 基线:

- pressure 与 flow 分别走 `SensorTemporalEncoder`
- `SensorTemporalEncoder` 由两部分组成:
  - 前端 1D CNN stem，用于局部时域模式抽取和时间降采样
  - 多层 dilation TCN block，用于扩大感受野并编码时序依赖
- 每个传感器分支输出:
  - 一组 token 序列
  - 一个 summary 向量
- 然后做双向 cross-attention:
  - pressure query flow
  - flow query pressure
- 得到两个交互后的传感器表征后，再通过 fusion 模块合并
- 当前主实验中的 PQ-only 基线使用 `gated` fusion
- 最后通过 MLP classifier 输出类别 logits

这条基线可以理解为“只看传感器的轻量双分支 cross-attention 模型”。

### 2.6 当前最优多模态模型 `hcaf_confgate_residual_pcen96hp80_5s` 的结构

当前第四章最终模型建立在 `HCAFNet` 上，并叠加了以下两个关键开关:

- `confidence_aware_gate = true`
- `expert_residual_scale = 0.3`

其整体流程可以分为 5 步。

第 1 步，三条模态分别编码为 token:

- audio:
  - 使用 `AudioTokenEncoder`
  - 先经过 3 层 2D CNN 编码
  - 再通过 `AdaptiveAvgPool2d((1, token_frames))` 压缩到固定数量的时间 token
  - 默认 `audio_token_frames = 12`
  - 同时额外输出一个 audio summary 向量
- pressure:
  - 使用 `SensorTemporalEncoder`
  - 默认输出 `sensor_token_length = 16` 个 token
  - 同时输出 pressure summary 向量
- flow:
  - 与 pressure 完全同构

第 2 步，先做 pressure-flow 内部融合:

- pressure token 以 flow token 为上下文做 cross-attention
- flow token 以 pressure token 为上下文做 cross-attention
- 然后分别得到交互后的 pressure / flow token
- 再用 `MaskedTokenGate` 同时完成两类融合:
  - token 级融合，得到 sensor tokens
  - representation 级融合，得到 sensor repr

这一步的作用是先在传感器内部形成一个较稳定的“PQ 联合表征”，再让音频与之交互。

第 3 步，做 audio-sensor 跨模态融合:

- audio token 以 sensor token 为上下文做 cross-attention
- sensor token 以 audio token 为上下文做 cross-attention
- 之后把 audio tokens 与 sensor tokens 拼接
- 再经过 `self_attention_layers = 1` 层轻量 self-attention

这一步相当于在 PQ 已经先融合的基础上，再让音频与“联合传感器表征”发生跨模态信息交换。

第 4 步，形成模态级表示并做 reliability gating:

- audio tokens 池化后与 audio summary 相加，得到 `audio_repr`
- sensor tokens 池化后与前面融合得到的 `sensor_repr` 相加，得到最终 `sensor_repr`
- 然后分别经过两个 expert 头:
  - `audio_expert`
  - `sensor_expert`
- 这两个 expert 会各自产生一份类别 logits

在当前最终模型中，最终门控不是只看表征本身，而是使用 `ConfidenceAwareReliabilityGate`。它会同时利用:

- `audio_repr`
- `sensor_repr`
- `audio_expert logits`
- `sensor_expert logits`

从 expert logits 中提取的 confidence 特征包括:

- top-1 概率
- top-1 与 top-2 的 margin
- 归一化后的 `1 - entropy`

然后基于这些信息预测 audio 与 sensor 两路的融合权重，再得到最终 fused representation。

第 5 步，加入 expert residual 并输出最终结果:

- 先对 fused representation 做主分类器 `classifier`
- 再把 audio expert 与 sensor expert 的 logits，按照 gate 权重加权求和
- 最后以 `expert_residual_scale = 0.3` 的比例加回主分类 logits

因此当前最终模型的最终输出可以理解为:

- 主干融合分类器的输出
- 加上一个“按门控权重加权的 expert-logit residual”

这正是 `hcaf_confgate_residual_5s` 相比普通 HCAF 的关键区别；而 `hcaf_confgate_residual_pcen96hp80_5s` 则是在完全相同的融合主干上，将音频前端进一步替换为 `PCEN96 + HP80` 后得到的当前最优版本。

### 2.7 缺失模态与模态 dropout 是怎么实现的

当前 HCAF 框架已经原生支持缺失模态，不需要另外改模型结构。

实现方式有两层:

1. `enabled_modalities`
   - 用于显式指定某次实验哪些模态可用
   - 例如:
     - `["pressure", "flow"]` 表示缺失音频
     - `["audio"]` 表示只保留音频
2. `modality_dropout`
   - 训练时随机将某些模态 mask 掉
   - 默认主模型使用 `modality_dropout = 0.1`

在 HCAFNet 内部:

- 每个 batch 会先采样 `audio_mask / pressure_mask / flow_mask`
- 被 mask 的模态，其 token 和 summary 会直接乘零
- 后续 cross-attention、token fusion、reliability gate 都会继续接收这些 availability mask
- 如果某个样本在训练时恰好所有模态都被丢掉，代码会强制恢复至少一个模态，避免空输入

因此，当前项目中的“缺失模态鲁棒性”并不是后处理技巧，而是模型结构层面已经考虑了模态可用性。

### 2.8 为什么当前模型设计适合本任务

从实现角度看，当前模型的设计逻辑是:

- 音频更适合转成时频图后用 2D CNN 编码局部谱纹理
- pressure / flow 更适合保留为一维序列，用 1D CNN + TCN 抽取时域波形和节律结构
- PQ 两条传感器之间耦合紧密，因此先在传感器内部做 cross-attention
- 音频与传感器的信息质量不总是稳定，因此最终融合不能简单拼接，而要引入 reliability gate
- 门控如果只依赖表征，容易不稳，因此进一步加入 confidence-aware gate 与 expert residual

也就是说，当前代码中的多模态建模并不是简单把三路特征拼起来，而是按“模态内先建模、模态间再交互、分类前再按置信度加权”的层次组织起来的。

## 3. 可复现实验清单

### 3.1 主结果对比：PQ-only vs 上一版 best 多模态

- 配置文件: `configs/pq_vs_multimodal_check.yaml`
- 输出目录: `summary-MMmodel/pq_vs_multimodal_check`
- split manifest: `summary-MMmodel/pq_vs_multimodal_check/split_manifest.json`
- seed: `20260330`
- grouped CV: `1 repeat x 3 folds`
- 训练预算: `epochs=8`, `early_stop_patience=3`
- 复现实验命令:

```bash
conda run -n dl python summary_mmmodel_experiments.py --config configs/pq_vs_multimodal_check.yaml
```

### 3.2 融合机制消融：normalization / confidence-aware gate / expert residual

- 配置文件: `configs/hcaf_fusion_gate_followup.yaml`
- 输出目录: `summary-MMmodel/hcaf_fusion_gate_followup`
- split manifest: `summary-MMmodel/hcaf_fusion_gate_followup/split_manifest.json`
- seed: `20260330`
- grouped CV: `1 repeat x 3 folds`
- 训练预算: `epochs=8`, `early_stop_patience=3`
- 复现实验命令:

```bash
conda run -n dl python summary_mmmodel_experiments.py --config configs/hcaf_fusion_gate_followup.yaml
```

### 3.3 缺失模态鲁棒性补充实验

- 配置文件: `configs/hcaf_missing_modalities.yaml`
- 输出目录: `summary-MMmodel/hcaf_missing_modalities`
- split manifest: `summary-MMmodel/hcaf_missing_modalities/split_manifest.json`
- seed: `20260330`
- grouped CV: `1 repeat x 3 folds`
- 训练预算: `epochs=8`, `early_stop_patience=3`
- 运行环境: `dl` conda 环境, `PyTorch 2.9.1+cu130`, `NVIDIA GeForce RTX 3070`
- 复现实验命令:

```bash
conda run -n dl python summary_mmmodel_experiments.py --config configs/hcaf_missing_modalities.yaml
```

说明:

- `hcaf_missing_modalities` 与 `pq_vs_multimodal_check` 使用相同的 grouped CV 划分原则、相同 seed 和相同训练预算，但生成的 `split_manifest` 不完全相同。
- 因此，缺失模态实验用于分析“当前最佳 HCAF 在不同可用模态条件下的相对退化趋势”，不直接替代主结果表中的严格 head-to-head 结论。

## 4. 主结果表：第四章正文优先引用版本

主表建议优先引用 `configs/pq_vs_multimodal_check.yaml` 这组结果，因为它专门回答“上一版 best 多模态是否已经超过 PQ-only”，且三种模型共用同一轮对比设置。

| Model | Window-level macro-F1 | Session-level macro-F1 | Best session aggregation | 解释 |
| --- | --- | --- | --- | --- |
| `pressure_flow_5s` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` | `majority_voting` | PQ-only 双模态传感器基线 |
| `hcaf_normfix_5s` | `0.8011 ± 0.1424` | `0.8259 ± 0.1406` | `majority_voting` | 修复 shared norm 后的 HCAF |
| `hcaf_confgate_residual_5s` | `0.7760 ± 0.0972` | `0.8815 ± 0.0838` | `majority_voting` | 上一版 best 多模态模型 |

可直接写入论文的主结论:

- 上一版 best 多模态模型 `hcaf_confgate_residual_5s` 在 session-level macro-F1 上略优于 PQ-only 基线，`0.8815 > 0.8519`。
- 该提升幅度有限，但方差明显更小，`0.0838 < 0.2095`，说明音频信息带来的主要收益是“略优且更稳定”，而不是显著碾压。
- `hcaf_normfix_5s` 虽然在 window-level 指标上优于 PQ-only，但其 session-level macro-F1 仍低于 `pressure_flow_5s`，说明普通多模态结构并不会自动转化为最终 recording/session 粒度增益。

建议正文表述:

> 在固定 `5 s` 窗口、相同训练预算和同一组 grouped CV 设置下，上一版 best 多模态模型 `hcaf_confgate_residual_5s` 的 session-level macro-F1 为 `0.8815 ± 0.0838`，略高于 PQ-only 基线 `pressure_flow_5s` 的 `0.8519 ± 0.2095`。该结果表明，在融合机制已经改进后，音频信息开始带来可复现的增益，但增益主要体现为稳定性改善，而非显著幅度的性能跃升。

### 4.1 在相同 split 上的性能提升尝试

为了判断当前多模态模型是否还能继续提升，额外做了一轮严格同条件的回灌实验:

- 配置文件: `configs/hcaf_confgate_improve_search.yaml`
- 输出目录: `summary-MMmodel/hcaf_confgate_improve_search`
- split manifest: `summary-MMmodel/hcaf_confgate_improve_search/split_manifest.json`
- 复用来源: `summary-MMmodel/pq_vs_multimodal_check/split_manifest.json`
- 验证结果: 两份 `split_manifest` 已逐字一致
- seed: `20260330`
- 训练预算: `epochs=8`, `early_stop_patience=3`
- 目标: 在不改 HCAF 主干结构的前提下，仅替换音频前端，观察是否能继续提升 `hcaf_confgate_residual_5s`

对比结果如下:

| Variant | Window-level macro-F1 | Session-level macro-F1 | 相对 base 的 session 变化 |
| --- | --- | --- | --- |
| `hcaf_confgate_residual_base_5s` | `0.8671 ± 0.0547` | `0.8815 ± 0.0838` | `0.0000` |
| `hcaf_confgate_residual_preemphasis16k_5s` | `0.8541 ± 0.0405` | `0.8815 ± 0.0838` | `0.0000` |
| `hcaf_confgate_residual_preemphasis12k_5s` | `0.7705 ± 0.1494` | `0.7926 ± 0.1826` | `-0.0889` |
| `hcaf_confgate_residual_pcen96hp80_5s` | `0.9207 ± 0.0261` | `0.9407 ± 0.0838` | `+0.0593` |

这一轮的结论非常明确:

- 单纯把 `preemphasis 16k` 回灌到当前最佳 HCAF，并没有继续提升，session-level macro-F1 与 base 持平。
- `preemphasis 12k` 明显退化，说明对当前融合模型而言，音频降采样并不划算。
- 真正有效的改进来自 `PCEN96 + high-pass 80 Hz`。
- 在与原始 best 完全相同的 split 上，`hcaf_confgate_residual_pcen96hp80_5s` 将 session-level macro-F1 从 `0.8815` 提升到了 `0.9407`，window-level macro-F1 也从 `0.8671` 提升到 `0.9207`。

因此，若将第四章最终模型口径更新为当前最佳版本，最有力的新结论是:

> 在保持 `confidence-aware gate + expert residual` 融合结构不变的前提下，将音频前端替换为 `PCEN96 + high-pass 80 Hz` 可以进一步提升最终多模态模型的性能；在与上一版 best 完全相同的 grouped CV 划分上，session-level macro-F1 从 `0.8815` 提升到 `0.9407`。因此，`hcaf_confgate_residual_pcen96hp80_5s` 应作为第四章的当前最终模型。

### 4.2 低通 / 带通补充验证：为什么最终保留 `HP80`

针对“低频是否主要是噪声、是否应进一步做低通”的问题，额外补做了同 split、同主干、只改滤波策略的最小对照实验。

- 配置文件: `configs/hcaf_confgate_filter_lowpass300.yaml`
- 输出目录: `summary-MMmodel/hcaf_confgate_filter_lowpass300`
- split manifest: `summary-MMmodel/hcaf_confgate_filter_lowpass300/split_manifest.json`
- 复用来源: `summary-MMmodel/pq_vs_multimodal_check/split_manifest.json`
- seed: `20260330`
- 训练预算: `epochs=8`, `early_stop_patience=3`
- 对照口径:
  - `PCEN96 + HP80` 结果来自 `configs/hcaf_confgate_improve_search.yaml`
  - `PCEN96 + LP300` 与 `PCEN96 + BP80-300` 结果来自本节新实验

对比结果如下:

| Variant | Window-level macro-F1 | Session-level macro-F1 | 相对 `HP80` 的 session 变化 |
| --- | --- | --- | --- |
| `hcaf_confgate_residual_pcen96hp80_5s` | `0.9207 ± 0.0261` | `0.9407 ± 0.0838` | `0.0000` |
| `hcaf_confgate_residual_pcen96lp300_5s` | `0.8570 ± 0.0769` | `0.8815 ± 0.0838` | `-0.0593` |
| `hcaf_confgate_residual_pcen96hp80lp300_5s` | `0.7709 ± 0.1303` | `0.8259 ± 0.1406` | `-0.1148` |

这一组结果说明:

- 仅做 `LP300` 后，session-level macro-F1 从 `0.9407` 回落到 `0.8815`，几乎退回到上一版 best 水平。
- 做更窄的 `BP80-300` 后退化更明显，说明把 `300 Hz` 以上频段全部裁掉会损失对当前任务有用的音频判别信息。
- 因此，当前最合理的解释不是“高频主要都是噪声”，而是:
  - `80 Hz` 以下确实存在较强低频漂移 / 接触扰动 / 呼吸基线成分，适合用 `HP80` 抑制；
  - 但 `300 Hz` 以上并非纯噪声，其中仍包含对吞咽事件有价值的瞬态或谐波线索。

对“噪声如何判断”的口径，本文采用的是“频谱先验 + 任务验证”两步法，而不是仅凭能量大小主观判断:

1. 频谱先验  
   对全数据按 `5 s` 窗做粗统计时，`0-80 Hz` 频带能量占比约为 `87.3%`，`80-300 Hz` 约为 `6.4%`，`300-1000 Hz` 约为 `3.6%`。这说明数据中确实存在很强的超低频成分，但“能量大”并不自动等于“信息最有用”。
2. 任务验证  
   在相同 grouped CV 划分、相同 HCAF 主干下，只改滤波方式进行 head-to-head 对照。若滤掉某一频段后性能提升，才可认为该频段更可能以噪声或冗余成分为主；若性能下降，则说明该频段仍包含有效判别信息。

因此，本轮补充实验最终支持继续保留 `PCEN96 + HP80` 作为当前最优音频前端，而不建议将 `LP300` 或 `BP80-300` 作为新的默认方案。

### 4.3 结构与训练超参补充搜索：batch size / attention 长度 / ResNet encoder

在确认 `PCEN96 + HP80` 是当前最优音频前端后，又额外做了一轮更偏“结构侧”的补充搜索，用来回答下面三个问题:

1. 调整 `batch_size` 是否还能继续提升当前 best？
2. 改变 cross/self-attention 的 token 长度是否更适合当前数据规模？
3. 将 audio encoder 替换为 `ResNet18 / ResNet34`，并让 PQ 分支也改为相近复杂度的 1D ResNet，是否能进一步提高最终性能？

实验设置如下:

- 配置文件: `configs/hcaf_arch_search.yaml`
- 输出目录: `summary-MMmodel/hcaf_arch_search`
- split manifest 复用: `summary-MMmodel/pq_vs_multimodal_check/split_manifest.json`
- 固定前端: `PCEN96 + high-pass 80 Hz`
- 统一训练预算: `epochs=8`, `early_stop_patience=3`
- 统一主干: `confidence-aware gate + expert residual`
- 主要比较口径: session-level macro-F1

已完整跑完的结果如下:

| Variant | Window-level macro-F1 | Session-level macro-F1 | 相对当前 best 的 session 变化 |
| --- | --- | --- | --- |
| `hcaf_pcen_base_5s` | `0.9259 ± 0.0079` | `0.9407 ± 0.0838` | `0.0000` |
| `hcaf_pcen_bs8_5s` | `0.9022 ± 0.0154` | `0.8815 ± 0.0838` | `-0.0593` |
| `hcaf_pcen_bs32_5s` | `0.8343 ± 0.0892` | `0.8296 ± 0.1362` | `-0.1111` |
| `hcaf_pcen_attn_short_5s` | `0.8504 ± 0.0701` | `0.8259 ± 0.1406` | `-0.1148` |
| `hcaf_pcen_attn_long_5s` | `0.8477 ± 0.0807` | `0.8296 ± 0.1362` | `-0.1111` |
| `hcaf_pcen_resnet18_5s` | `0.8933 ± 0.0271` | `0.8815 ± 0.0838` | `-0.0593` |

关于 `ResNet34`，本轮没有继续把三折全部跑完，而是在完成 `repeat1_fold1` 后就停止了后续训练。原因是:

- `hcaf_pcen_resnet34_5s` 在 `repeat1_fold1` 的 session-level macro-F1 已经只有 `0.8222`
- 在当前 `1 x 3 folds` 的平均口径下，剩余两折即使都达到满分 `1.0`，最终平均值也只能追平 `0.9407`，不可能严格超过当前 best
- 因此，为避免无效算力开销，后续 `ResNet34` 训练被提前终止

这一轮搜索的结论也比较明确:

- `batch_size` 从 `16` 改为 `8` 或 `32` 都没有带来提升，其中 `32` 退化最明显。
- 无论把 attention token 长度缩短，还是把 `audio_token_frames / sensor_token_length` 拉长并叠加 `2` 层 self-attention，结果都低于当前基线。
- 将 audio 与 PQ encoder 同时替换为 `ResNet18` 后，window-level 表现仍然不错，但 session-level macro-F1 回落到 `0.8815`，说明更重的 encoder 并没有转化为更稳定的 recording-level 判别收益。
- 因此，在当前数据规模与训练预算下，`hcaf_confgate_residual_pcen96hp80_5s` 仍应保留为第四章最终模型，不建议进一步切换到更大的 ResNet 编码器或更长 / 更短的 token 注意力长度。

### 4.4 ResNet18 是否应使用 ImageNet 迁移学习

进一步地，又对 `ResNet18` 做了一轮更干净的迁移学习对照，只比较:

- `ResNet18 scratch`
- `ResNet18 + ImageNet init`

实验设置如下:

- 配置文件: `configs/hcaf_resnet18_imagenet_only.yaml`
- 输出目录: `summary-MMmodel/hcaf_resnet18_imagenet_only`
- split manifest 复用: `summary-MMmodel/pq_vs_multimodal_check/split_manifest.json`
- 固定前端: `PCEN96 + high-pass 80 Hz`
- 固定 PQ 分支: `1D ResNet18`
- 唯一变化: audio `ResNet18` 是随机初始化还是 `ImageNet` 初始化
- 训练策略: 全量 fine-tune，不冻结 backbone

结果如下:

| Variant | Window-level macro-F1 | Session-level macro-F1 | 相对 scratch 的 session 变化 |
| --- | --- | --- | --- |
| `hcaf_resnet18_scratch_5s` | `0.8348 ± 0.0433` | `0.8222 ± 0.0000` | `0.0000` |
| `hcaf_resnet18_imagenet_5s` | `0.9111 ± 0.0839` | `0.9407 ± 0.0838` | `+0.1185` |

这一组结果说明:

- 对 `ResNet18` 音频编码器本身而言，`ImageNet` 迁移学习是有效的。
- 与从头训练相比，`ImageNet` 初始化将 session-level macro-F1 从 `0.8222` 提升到了 `0.9407`，提升幅度达到 `+0.1185`。
- 因此，如果坚持采用 `ResNet18` 作为 audio encoder，那么“用预训练”明显优于“从头训练”。
- 但需要强调的是，迁移学习后的 `ResNet18` 最终只是追平了当前 best，并没有继续超过 `hcaf_confgate_residual_pcen96hp80_5s`。
- 所以第四章最终模型仍不需要改成 `ResNet18`；更准确的表述是:
  - `ResNet18` 若不用迁移学习，会明显退化
  - `ResNet18` 若使用 `ImageNet` 迁移学习，可以恢复到与当前 best 相当的水平
  - 但它没有证明自己优于当前默认的轻量音频分支

## 5. 机制验证与最小必要消融

本节目标不是继续堆模型，而是回答两个问题:

1. 为什么普通多模态未必优于 PQ-only？
2. 为什么加入 `confidence-aware gate` 和 `expert residual` 后，才出现可引用的多模态增益？

以下结果来自 `configs/hcaf_fusion_gate_followup.yaml`。

| Variant | Window-level macro-F1 | Session-level macro-F1 | 机制解释 |
| --- | --- | --- | --- |
| `hcaf_legacy_sharednorm_5s` | `0.7919 ± 0.1489` | `0.8815 ± 0.0838` | 早期 HCAF，pressure/flow 分支复用 audio `LayerNorm` |
| `hcaf_normfix_5s` | `0.8728 ± 0.0529` | `0.8815 ± 0.0838` | 先修复表示空间的 normalization 问题 |
| `hcaf_confgate_5s` | `0.6791 ± 0.1195` | `0.6857 ± 0.1931` | 仅引入 confidence-aware gate，性能明显不稳 |
| `hcaf_confgate_residual_5s` | `0.8847 ± 0.0733` | `0.8815 ± 0.0838` | 在 confidence gate 基础上加入 expert residual |

本节结论:

- 首先需要修复表示层问题。`hcaf_normfix_5s` 相比 `legacy_sharednorm`，window-level macro-F1 从 `0.7919` 提升到 `0.8728`，且标准差明显下降，说明旧版共享归一化确实削弱了压力/流速分支的表达稳定性。
- 仅加入 `confidence-aware gate` 并不能带来收益，反而会显著退化到 `0.6857 ± 0.1931`。这说明“更聪明的门控”本身并不等于更好的融合；在小样本条件下，confidence 信号可能过早放大某一模态，导致鲁棒性下降。
- 在 confidence gate 之后再加入 `expert residual`，window-level 指标恢复并略超 `normfix`，达到 `0.8847 ± 0.0733`。这说明 residual expert logits 在一定程度上缓解了单纯 gate 的不稳定问题，使门控决策不再完全依赖单一路径。
- 将本节与主结果表合并解读，可以得到一条较完整的证据链:
  - 普通多模态并不必然优于 PQ-only，`hcaf_normfix_5s` 在主结果表中仍低于 `pressure_flow_5s`。
  - 仅增加 confidence gate 还会进一步退化。
  - 只有当 `confidence-aware gate` 与 `expert residual` 联合使用时，多模态系统才在 session-level 上出现“略优且更稳定”的增益。

可直接写入论文的解释性表述:

> 结果表明，多模态收益并非来自“简单增加一个音频分支”，而是来自更稳健的融合机制。普通 HCAF 变体在 session-level 上未能稳定超过 PQ-only；仅引入 confidence-aware gate 甚至会带来明显退化。相比之下，`confidence-aware gate + expert residual` 的组合在保留融合灵活性的同时抑制了门控不稳定性，从而使多模态模型获得了小幅但可复现的性能增益。

## 6. 缺失模态鲁棒性

本节使用现有 HCAF 框架内置的 `enabled_modalities` 与 `modality_dropout` 机制，不额外引入新模型，仅对当前最佳 HCAF 变体做固定可用模态条件实验。结果来自 `configs/hcaf_missing_modalities.yaml`。

| Condition | Window-level macro-F1 | Session-level macro-F1 | 相对 full 的 session 变化 |
| --- | --- | --- | --- |
| Full multimodal (`audio + pressure + flow`) | `0.8065 ± 0.1489` | `0.8296 ± 0.1362` | `0.0000` |
| Missing audio (`pressure + flow`) | `0.7028 ± 0.2052` | `0.8042 ± 0.2769` | `-0.0254` |
| Missing pressure (`audio + flow`) | `0.8662 ± 0.0910` | `0.8815 ± 0.0838` | `+0.0519` |
| Missing flow (`audio + pressure`) | `0.7266 ± 0.1130` | `0.7026 ± 0.0868` | `-0.1270` |
| Missing PQ (`audio only`) | `0.6255 ± 0.1125` | `0.7386 ± 0.1319` | `-0.0910` |

本节解读时应注意:

- 该表的绝对值不用于替代主结果表，因为它对应的是另一份 `split_manifest`。
- 其价值在于观察“当某类模态在床旁采集中缺失时，当前最佳 HCAF 会如何退化”。

鲁棒性结论:

- 缺失音频后，模型仍能保持 `0.8042` 的 session-level macro-F1，说明当麦克风信号不可用时，PQ 仍可支撑基本判别能力。
- 缺失 PQ、只保留音频时，session-level macro-F1 下降到 `0.7386`，说明单靠音频可以提供一定区分度，但难以替代传感器模态。
- 在双模态条件中，`audio + flow` 的结果优于 full multimodal，而 `audio + pressure` 明显下降，提示当前数据划分下 flow 的信息密度高于 pressure，pressure 还可能引入一定噪声或冗余。
- 缺失 flow 的退化幅度最大（`-0.1270`），说明在当前系统中，flow 是比 pressure 更关键的传感器模态。

床旁应用意义:

- 模型并不依赖“三模态同时完美可用”这一过强前提，模态缺失时仍可退化运行。
- 当音频采集受环境噪声、遮挡或设备限制影响时，PQ-only 仍是可接受的后备方案。
- 当传感器链路部分失效时，优先保证 flow 信号的稳定采集更有价值。
- 当 PQ 全部缺失、仅剩音频时，系统仍有一定判别能力，但不应作为与完整系统等价的替代方案。

## 7. 上一版 best 模型的窗长影响分析

为避免继续引用更早期 HCAF 的旧窗长结果，专门针对上一版 best 模型 `hcaf_confgate_residual` 补做了窗长实验。

- 配置文件: `configs/hcaf_confgate_window_lengths.yaml`
- 输出目录: `summary-MMmodel/hcaf_confgate_window_lengths`
- split manifest: `summary-MMmodel/hcaf_confgate_window_lengths/split_manifest.json`
- seed: `20260330`
- grouped CV: `1 repeat x 3 folds`
- 训练预算: `epochs=8`, `early_stop_patience=3`
- 复现实验命令:

```bash
conda run -n dl python summary_mmmodel_experiments.py --config configs/hcaf_confgate_window_lengths.yaml
```

结果如下:

| Window length | Window-level macro-F1 | Session-level macro-F1 | Best session aggregation | Session-level std |
| --- | --- | --- | --- | --- |
| `5 s` | `0.6745 ± 0.0918` | `0.7545 ± 0.0958` | `majority_voting` | `0.0958` |
| `10 s` | `0.6613 ± 0.1246` | `0.7619 ± 0.1695` | `logit_averaging` | `0.1695` |
| `20 s` | `0.7577 ± 0.1943` | `0.7926 ± 0.1826` | `majority_voting` | `0.1826` |

补充窗长实验 `6 s / 8 s / 15 s` 使用配置 `configs/hcaf_confgate_window_lengths_6_8_15.yaml`，输出目录为 `summary-MMmodel/hcaf_confgate_window_lengths_6_8_15`。将两组结果合并后，可得到上一版 best 模型的总窗长对比表:

| Window length | Window-level macro-F1 | Session-level macro-F1 | Best session aggregation | Session-level std | 备注 |
| --- | --- | --- | --- | --- | --- |
| `5 s` | `0.6745 ± 0.0918` | `0.7545 ± 0.0958` | `majority_voting` | `0.0958` | 当前最稳定 |
| `6 s` | `0.6419 ± 0.1336` | `0.7026 ± 0.0868` | `majority_voting` | `0.0868` | 稳定但均值偏低 |
| `8 s` | `0.7099 ± 0.2337` | `0.7407 ± 0.2516` | `majority_voting` | `0.2516` | 波动最大 |
| `10 s` | `0.6613 ± 0.1246` | `0.7619 ± 0.1695` | `logit_averaging` | `0.1695` | 需要 logit 聚合 |
| `15 s` | `0.6651 ± 0.1116` | `0.6868 ± 0.0958` | `majority_voting` | `0.0958` | 长窗但均值不占优 |
| `20 s` | `0.7577 ± 0.1943` | `0.7926 ± 0.1826` | `majority_voting` | `0.1826` | 平均值最高但不稳 |

若按 session-level macro-F1 的均值排序:

1. `20 s`: `0.7926 ± 0.1826`
2. `10 s`: `0.7619 ± 0.1695`
3. `5 s`: `0.7545 ± 0.0958`
4. `8 s`: `0.7407 ± 0.2516`
5. `6 s`: `0.7026 ± 0.0868`
6. `15 s`: `0.6868 ± 0.0958`

若同时考虑稳定性与可解释性，当前更适合正文强调的是:

- `5 s`: 均值不是最高，但稳定性最好，且与主结果表的默认设定一致
- `20 s`: 平均值最高，可作为“更长上下文有潜力”的补充观察
- `10 s`: 位于中间，但其最佳结果依赖 `logit_averaging`，说明该窗长下单窗不确定性更高

对应的数据量变化:

| Window length | Mean train windows / fold | Mean test windows / fold | Mean test windows / session |
| --- | --- | --- | --- |
| `5 s` | `1426.3` | `938.0` | `156.3` |
| `10 s` | `711.3` | `468.0` | `78.0` |
| `20 s` | `353.3` | `232.3` | `38.7` |

窗长影响的主要观察:

- 在这组当前模型的专门实验中，`20 s` 的平均 session-level macro-F1 最高，但方差也最大。
- `5 s` 的平均值不是最高，却是三种窗长里最稳定的方案。
- `10 s` 处于中间状态，平均表现略高于 `5 s`，但跨折波动明显增大，且只有在 `logit_averaging` 下才能达到最优。

原因分析:

- 第一，窗长变大后，训练样本数近似按比例下降。`5 s -> 10 s -> 20 s` 时，平均训练窗口数从 `1426.3 -> 711.3 -> 353.3`，模型可见样本大幅减少，因此更容易出现 fold 间波动放大。
- 第二，长窗口为单个样本提供了更完整的吞咽上下文，可能帮助当前的 cross-attention 与 confidence-aware fusion 捕获更稳定的跨模态依赖，因此 `20 s` 的平均值反而高于 `5 s`。
- 第三，长窗口也会混入更多与目标负荷无关的前后背景，导致“有时很有帮助，有时会稀释判别线索”，这与 `20 s` 明显更大的标准差相一致。
- 第四，短窗口虽然上下文较少，但提供了更多 session 内投票单元。`5 s` 条件下每个 test session 平均有 `156.3` 个窗口，而 `20 s` 仅有 `38.7` 个，因此 `5 s` 更容易依靠 session aggregation 获得稳健结果。
- 第五，`10 s` 只有 `logit_averaging` 优于其他聚合方式，说明这一中间窗长下单窗预测的不确定性更高，简单投票不足以稳定整段 recording 的判断，必须保留连续 logit 信息。

本节建议写法:

> 对当前最优 `hcaf_confgate_residual` 而言，窗长增加并未单调恶化性能。`20 s` 在平均 session-level macro-F1 上最高，但跨折方差也最大；`5 s` 的平均性能略低，却表现出更好的稳定性。这表明长窗口能够为融合模块提供更完整的跨模态上下文，但同时也因样本数减少和背景信息混入而放大了 fold 间波动。因此，在当前数据规模下，若追求最稳妥的论文结论，应优先强调 `5 s` 的稳定性；若讨论模型上限，则可以指出 `20 s` 具有更高的平均潜力，但仍需更大样本验证其可靠性。

## 8. 第四章结论建议写法

可作为第四章结论段的精简版本:

> 综合主结果、机制消融、缺失模态实验、窗长分析、音频前端对照以及后续的 batch size / attention 长度 / encoder 搜索可以看出，当前多模态系统的优势并不表现为对 PQ-only 的显著碾压，而是体现在更稳定的 session-level 判别性能。普通多模态结构并不会自动优于 PQ-only，甚至在融合设计不当时会出现退化；只有在修复分支表示后，再结合 `confidence-aware gate` 与 `expert residual`，多模态模型才获得了小幅但可复现的增益。进一步地，在保持 HCAF 主干不变的前提下，将音频前端替换为 `PCEN96 + HP80` 后，当前最终模型 `hcaf_confgate_residual_pcen96hp80_5s` 将 session-level macro-F1 提升到 `0.9407`；而后续对 `batch_size`、attention token 长度和 `ResNet18/34` 编码器的补充搜索均未能继续超过这一结果。补充的迁移学习对照还表明，若采用 `ResNet18` 作为音频编码器，则 `ImageNet` 初始化显著优于从头训练，但最终也只是追平当前 best，而没有继续刷新结果。与此同时，缺失模态实验表明该模型具备一定退化运行能力，其中 flow 对最终性能的贡献高于 pressure；窗长实验则提示长窗口可能带来更高平均性能，但 `5 s` 方案在当前样本规模下更稳定。

## 9. 局限性与替代性说明

- 当前所有结论均建立在 `0 / 2 / 4` 三分类任务之上，不直接外推到 `1 / 3 ml`。
- 样本量仍然偏小，session-level 每折测试 session 数有限，因此不宜将 `0.8815 vs 0.8519` 解读为统计意义上的显著领先。
- 缺失模态补充实验没有再实现额外的“训练时缺失 / 测试时缺失”复杂策略，而是优先复用了 HCAF 现有的 `enabled_modalities` 与 `modality_dropout` 机制。这是刻意采用的最小实现方案，目的是补齐第四章关键证据链，而不是引入新的方法学变量。
- `hcaf_missing_modalities` 的 `split_manifest` 与 `pq_vs_multimodal_check` 不完全一致，因此缺失模态结果应作为鲁棒性分析，而不是主表的严格替代。
- `hcaf_confgate_window_lengths` 的 `split_manifest` 也与 `pq_vs_multimodal_check` 不完全一致，因此窗长实验应优先在其内部做相对比较，不直接与主结果表做绝对值横向绑定。

## 10. 本轮新增内容摘要

- 新增缺失模态鲁棒性配置: `configs/hcaf_missing_modalities.yaml`
- 新增缺失模态结果目录: `summary-MMmodel/hcaf_missing_modalities`
- 新增窗长分析配置: `configs/hcaf_confgate_window_lengths.yaml`
- 新增窗长分析结果目录: `summary-MMmodel/hcaf_confgate_window_lengths`
- 新增同 split 提升实验配置: `configs/hcaf_confgate_improve_search.yaml`
- 新增同 split 提升实验结果目录: `summary-MMmodel/hcaf_confgate_improve_search`
- 新增结构与训练超参搜索配置: `configs/hcaf_arch_search.yaml`
- 新增结构与训练超参搜索结果目录: `summary-MMmodel/hcaf_arch_search`
- 新增 ResNet18 迁移学习对照配置: `configs/hcaf_resnet18_imagenet_only.yaml`
- 新增 ResNet18 迁移学习对照结果目录: `summary-MMmodel/hcaf_resnet18_imagenet_only`
- 新增并核对的核心证据链:
  - 主结果表: `pressure_flow_5s` / `hcaf_normfix_5s` / `hcaf_confgate_residual_5s`
  - 同 split 提升实验: `base` / `preemphasis 16k` / `preemphasis 12k` / `PCEN96 HP80`
  - 同 split 滤波补充实验: `PCEN96 LP300` / `PCEN96 BP80-300`
  - 结构与训练超参搜索: `batch 8` / `batch 32` / `short attention` / `long attention` / `ResNet18`
  - ResNet18 迁移学习对照: `scratch` / `ImageNet init`
  - 机制消融表: `legacy shared norm` / `norm fix` / `confidence-aware gate` / `confidence-aware gate + expert residual`
  - 缺失模态鲁棒性表: full / missing audio / missing pressure / missing flow / missing PQ
  - 上一版 best 模型窗长表: `5 s` / `10 s` / `20 s`
  - 报告口径已统一为:
  - 上一版 best 多模态相较 PQ-only 为“略优且更稳定”
  - 不能表述为“显著碾压”
  - 多模态增益依赖改进融合，而不是自然出现
  - 当前最终模型更新为 `hcaf_confgate_residual_pcen96hp80_5s`
  - 在严格同 split 的进一步实验中，`PCEN96 + HP80` 已将最终多模态模型提升到 `0.9407`
  - 后续对 `batch_size`、attention 长度与 `ResNet` encoder 的搜索没有继续超过该结果
  - 若采用 `ResNet18` 音频编码器，则 `ImageNet` 迁移学习显著优于从头训练，但最终仅追平当前 best

## 11. 采后同步质量控制

为在数据集构建阶段增加采后质量控制，对每条 session 计算音频保存时长与 DAQ 有效采样时长之比:

> `rho = T_audio / T_daq`

其中:

- `T_audio` 为保存的 `audio.wav` 总时长，由 `num_frames / sample_rate` 直接计算。
- `T_daq` 为 DAQ 数据对应的有效采样时长，按 `num_samples / sample_rate_hz` 计算；若元数据缺失，则回退到 `daq.csv` 行数与时间戳估计。

当 `rho` 明显偏离 `1` 时，可认为本次采集中可能存在链路提前终止、音频回调阻塞或软时钟漂移等异常，因此该指标可作为多模态样本同步质量的一级筛查条件。

对当前 `data/` 下全部 `29` 条 session 的全量统计结果如下:

- `rho` 均值为 `1.000146`，中位数为 `1.000045`
- `rho` 取值范围为 `[0.999957, 1.000820]`
- 平均偏差 `mean(|rho-1|)` 为 `0.000150`
- 最大偏差 `max(|rho-1|)` 为 `0.000820`
- 以 `|rho-1| > 0.01` 作为预警阈值时，触发预警的 session 数为 `0`

这说明当前数据集中，音频链路与 DAQ 链路在录制总时长层面整体保持了较好一致性，尚未发现明显失配样本。偏差最大的几条记录也都控制在千分之一以内，因此从 recording 级总时长角度看，当前数据可认为满足后续多模态建模的基本同步要求。

偏差最大的 session 例如:

- `MMdata_272.75s_0327_174501_2ml`: `rho = 1.000820`
- `MMdata_442.75s_0327_203239_4ml`: `rho = 1.000648`
- `MMdata_474.50s_0323_001822_4ml_yumi`: `rho = 1.000538`

对应的全量结果文件已保存为:

- `summary-MMmodel/capture_consistency.csv`
- `summary-MMmodel/capture_consistency_summary.json`
- `summary-MMmodel/capture_consistency_report.md`

若正文需要更正式的写法，可直接使用如下表述:

> 为保证多模态样本的同步可靠性，本文在数据集构建阶段引入采后质量控制机制，计算音频保存总时长与 DAQ 有效采样总时长之比 `rho = T_audio / T_daq`。当 `rho` 明显偏离 `1` 时，说明该样本可能存在提前终止、回调阻塞或时钟漂移等异常。对本研究全部 `29` 条记录的统计表明，`rho` 的均值为 `1.000146`，取值范围为 `[0.999957, 1.000820]`，且没有样本超过 `|rho-1| > 0.01` 的预警阈值，说明当前数据集在 recording 级总时长上一致性较好，可为后续多模态模型训练提供较可靠的数据基础。 

## 12. 基于当前最佳模型的可解释性与错误分析补充

为回答“当前最佳模型是否还能提供更强的机制证据”这一问题，额外针对 `hcaf_confgate_residual_pcen96hp80_5s` 做了一轮不重训的回放分析。具体做法是: 直接加载 `summary-MMmodel/hcaf_confgate_improve_search/runs/hcaf_confgate_residual_pcen96hp80_5s/repeat1_fold{1,2,3}/best_model.pt`，在各 fold 测试集上重新前向，提取:

- audio-sensor 双向 cross-attention 权重
- confidence-aware gate 的 audio / sensor 权重
- audio expert 与 sensor expert 的 confidence 特征
- 每个窗口的预测结果与 session 内相对时间位置

分析输出目录为:

- `summary-MMmodel/hcaf_confgate_interpretability`
- 其中核心文件包括:
  - `report.md`
  - `summary.json`
  - `window_debug.csv`
  - `session_debug.csv`
  - `attention_examples.png`
  - `gate_vs_expert_confidence.png`
  - `error_rate_by_position.png`
  - `window_confusion_matrix.png`
  - `session_confusion_matrix_majority.png`

本轮共回放:

- `3` 个 fold
- `2814` 个测试窗口
- `18` 个测试 session

### 12.1 动态门控行为分析

这一部分的核心目的是判断 confidence-aware gate 是否真的按模态可靠性重新分配权重，而不是仅仅作为一个额外的参数层存在。

首先，从全量窗口统计看，gate 权重与 expert confidence 确实存在明显正相关:

- audio gate 与 audio expert top-1 probability 的相关系数为 `0.695`
- sensor gate 与 sensor expert top-1 probability 的相关系数为 `0.373`

这说明当前 gate 的行为与设计初衷基本一致，即当某一路 expert 更有把握时，该路在最终融合中的权重也会随之提高，且这一关系在 audio 分支上尤其明显。

其次，不同类别对应的平均 audio gate 差异非常大:

| True class | Mean audio gate | Std |
| --- | --- | --- |
| `0` | `0.0446` | `0.0798` |
| `2` | `0.1090` | `0.1633` |
| `4` | `0.6218` | `0.3883` |

这表明当前最优模型在类别 `0` 与 `2` 上整体更依赖 PQ 传感器，而在类别 `4` 上明显更愿意提升音频分支权重。换言之，gate 不是对所有样本固定偏向某一模态，而是在不同类别、不同窗口之间动态切换。

再看典型 session 级案例:

- `MMdata_1200.00s_0327_172321_2ml`
  - true = `2`
  - session pred = `2`
  - mean audio gate = `0.0062`
  - mean sensor gate = `0.9938`
  - window error rate = `0.0000`
- `MMdata_1071.50s_0327_201445_4ml`
  - true = `4`
  - session pred = `4`
  - mean audio gate = `0.9135`
  - mean sensor gate = `0.0865`
  - window error rate = `0.0000`
- `MMdata_318.75s_0327_175326_no`
  - true = `0`
  - session pred = `2`
  - mean audio gate = `0.3053`
  - mean sensor gate = `0.6947`
  - window error rate = `0.9524`

这三条记录对应的 gate 时间曲线已保存为:

- `summary-MMmodel/hcaf_confgate_interpretability/gate_session_MMdata_1200.00s_0327_172321_2ml.png`
- `summary-MMmodel/hcaf_confgate_interpretability/gate_session_MMdata_1071.50s_0327_201445_4ml.png`
- `summary-MMmodel/hcaf_confgate_interpretability/gate_session_MMdata_318.75s_0327_175326_no.png`

这些案例说明，当前 gate 确实会在不同 session 上表现出明显不同的模态偏好，因此“动态融合”这一设计不仅存在于结构描述层面，也能够在实际推理行为中被观测到。

需要注意的是:

- 目前没有逐窗噪声标签，因此不能直接把“audio gate 降低”严格解释为“模型识别出了环境噪声”
- 但可以较有把握地说，当前 gate 会随 expert confidence 改变而调整模态权重，因此具备可量化的可靠性重加权行为

![Audio gate by class](/home/oi/MMDL/summary-MMmodel/hcaf_confgate_interpretability/audio_gate_by_class.png)

![Gate vs expert confidence](/home/oi/MMDL/summary-MMmodel/hcaf_confgate_interpretability/gate_vs_expert_confidence.png)

![Typical gate trajectory: 2 ml](/home/oi/MMDL/summary-MMmodel/hcaf_confgate_interpretability/gate_session_MMdata_1200.00s_0327_172321_2ml.png)

![Typical gate trajectory: 4 ml](/home/oi/MMDL/summary-MMmodel/hcaf_confgate_interpretability/gate_session_MMdata_1071.50s_0327_201445_4ml.png)

![Typical gate trajectory: misclassified no](/home/oi/MMDL/summary-MMmodel/hcaf_confgate_interpretability/gate_session_MMdata_318.75s_0327_175326_no.png)

### 12.2 Cross-attention 可视化

针对 `0 / 2 / 4` 三类，额外从测试集挑选了每类一个“预测正确且最终置信度最高”的代表性窗口，导出:

- audio -> sensor cross-attention heatmap
- sensor -> audio cross-attention heatmap

对应图已保存为:

- `summary-MMmodel/hcaf_confgate_interpretability/attention_examples.png`

这一图可作为第四章的案例分析图使用，用于说明当前最优模型在高置信正确样本上，确实会在音频 token 与联合 sensor token 之间形成非均匀的跨模态关注分布，而不是简单平均融合。

更谨慎的表述应为:

> 当前 HCAF 最优模型的可解释性不仅来自最终 gate 权重，还可通过 cross-attention 热图观察到音频 token 与联合传感器 token 之间的非均匀信息交换。代表性样本表明，模型在高置信预测时会集中关注有限的跨模态 token 区域，而非对整窗信息做均匀平均。

![Cross-attention examples](/home/oi/MMDL/summary-MMmodel/hcaf_confgate_interpretability/attention_examples.png)

### 12.3 错误分析

窗口级混淆矩阵统计如下:

| True \\ Pred | `0` | `2` | `4` |
| --- | --- | --- | --- |
| `0` | `717` | `77` | `0` |
| `2` | `67` | `978` | `0` |
| `4` | `8` | `42` | `925` |

session 级（majority voting）混淆矩阵如下:

| True \\ Pred | `0` | `2` | `4` |
| --- | --- | --- | --- |
| `0` | `5` | `1` | `0` |
| `2` | `0` | `6` | `0` |
| `4` | `0` | `0` | `6` |

这一组结果显示:

- 当前最佳模型最主要的错误模式是 `0 -> 2`
- `2 -> 4` 在当前回放中并不是主要混淆来源
- `4` 类的大部分错误更常表现为 `4 -> 2`，而不是直接掉到 `0`

这意味着模型更容易在相邻中间类别附近发生混淆，而不是在 `0` 与 `4` 之间出现大幅跨级误判。若按任务语义解读，这与“中间等级更容易吸收边界不确定样本”的直觉是一致的。

进一步看窗口在 session 内的位置分布，边界窗口的错误率高于中段窗口:

- 边界窗口（前 `20%` + 后 `20%`）错误率: `0.0878`
- 中间 `60%` 窗口错误率: `0.0563`

因此，`5 s` 固定切窗确实存在一定的边界效应，说明 recording 开头与结尾附近的窗口更容易被误判。这为后续改进提供了一个明确方向:

- 可考虑在分析阶段补充重叠滑窗或更柔和的 session aggregation
- 也可在未来引入事件级标注，进一步判断“吞咽事件被窗边界截断”是否是主要误差来源

但也要明确，目前还不能直接证明“错误主要发生在吞咽事件正中间被截断时”，因为现有数据没有逐事件时间标注。当前结论仅能表述为:

- 错误更偏向 session 相对边界位置
- 边界效应是一个合理且已被数据支持的误差来源假设

![Window confusion matrix](/home/oi/MMDL/summary-MMmodel/hcaf_confgate_interpretability/window_confusion_matrix.png)

![Session confusion matrix](/home/oi/MMDL/summary-MMmodel/hcaf_confgate_interpretability/session_confusion_matrix_majority.png)

![Error rate by position](/home/oi/MMDL/summary-MMmodel/hcaf_confgate_interpretability/error_rate_by_position.png)

### 12.4 本节可写入论文的简洁表述

> 针对当前最佳模型 `hcaf_confgate_residual_pcen96hp80_5s` 的回放分析表明，confidence-aware gate 的权重分配与 expert confidence 具有正相关关系，其中 audio gate 与 audio expert top-1 probability 的相关系数达到 `0.695`，说明门控行为并非随机扰动，而是在随模态可靠性动态调整。不同类别的平均 gate 权重也表现出明显差异: 类别 `0 / 2` 更依赖 PQ 传感器，而类别 `4` 中音频权重明显升高。错误分析进一步表明，当前模型最主要的混淆模式为 `0 -> 2`，而边界窗口的错误率高于 session 中段窗口，提示固定 `5 s` 切窗仍存在一定边界效应。综上，当前最优模型不仅在性能上优于上一版 HCAF，也已经能够提供较为一致的融合行为证据与明确的误差结构。 

## 13. 第四章图版整理

为避免正文和答辩材料继续引用分散在不同实验目录中的局部图片，额外统一生成了一套第四章汇总图，输出目录为:

- `summary-MMmodel/figures`

生成脚本为:

- `generate_chapter4_figures.py`

运行命令为:

```bash
conda run -n dl python generate_chapter4_figures.py
```

### 13.1 已生成的汇总图

- `summary-MMmodel/figures/primary_performance_progression.png`
  - 用途: 作为第四章最核心主图之一
  - 内容: `PQ-only -> 上一版 HCAF best -> 当前最终模型`
  - 可直接支撑“当前最终模型相较上一版 HCAF 提升 `+0.0593`，相较 PQ-only 提升 `+0.0889`”
- `summary-MMmodel/figures/frontend_same_split_comparison.png`
  - 用途: 展示同 split 下音频前端替换是否真的带来提升
  - 内容: `base / preemphasis16k / preemphasis12k / PCEN96+HP80`
  - 适合放在“当前最终模型如何得到”这一小节
- `summary-MMmodel/figures/filter_strategy_comparison.png`
  - 用途: 说明为什么最终保留 `HP80`
  - 内容: `HP80 / LP300 / BP80-300`
  - 适合放在“低通 / 带通补充验证”小节
- `summary-MMmodel/figures/fusion_mechanism_ablation.png`
  - 用途: 展示 normalization、confidence-aware gate、expert residual 的机制证据链
  - 内容: `legacy shared norm / norm fix / conf-gate / conf-gate+residual`
  - 适合放在机制消融部分
- `summary-MMmodel/figures/missing_modality_robustness.png`
  - 用途: 展示缺失模态时的退化趋势
  - 内容: `full / missing audio / missing pressure / missing flow / audio only`
  - 适合放在鲁棒性分析部分
- `summary-MMmodel/figures/window_length_tradeoff.png`
  - 用途: 展示窗长变化对 window-level 与 session-level 指标的影响
  - 内容: `5 / 6 / 8 / 10 / 15 / 20 s`
  - 适合放在窗长分析部分
- `summary-MMmodel/figures/interpretability_error_summary.png`
  - 用途: 以更紧凑的方式总结 gate 行为与边界效应
  - 内容: 左图为各类别 audio gate，右图为边界 vs 中段错误率
  - 适合放在可解释性或补充材料
- `summary-MMmodel/figures/hcaf_current_best_architecture.png`
  - 用途: 作为模型结构示意图
  - 内容: 从 `PCEN96 + HP80` 音频前端、PQ 内部 cross-attention、audio-sensor cross-attention、confidence-aware gate 到 expert residual 的全链路示意
  - 适合放在模型方法部分或答辩汇报页

![Primary performance progression](/home/oi/MMDL/summary-MMmodel/figures/primary_performance_progression.png)

![Same-split frontend comparison](/home/oi/MMDL/summary-MMmodel/figures/frontend_same_split_comparison.png)

![Filter strategy comparison](/home/oi/MMDL/summary-MMmodel/figures/filter_strategy_comparison.png)

![Fusion mechanism ablation](/home/oi/MMDL/summary-MMmodel/figures/fusion_mechanism_ablation.png)

![Missing-modality robustness](/home/oi/MMDL/summary-MMmodel/figures/missing_modality_robustness.png)

![Window-length tradeoff](/home/oi/MMDL/summary-MMmodel/figures/window_length_tradeoff.png)

![Interpretability and error summary](/home/oi/MMDL/summary-MMmodel/figures/interpretability_error_summary.png)

![Current best architecture](/home/oi/MMDL/summary-MMmodel/figures/hcaf_current_best_architecture.png)

### 13.2 正文优先推荐插图

若正文篇幅有限，建议优先放以下 `4` 张图:

1. `summary-MMmodel/figures/primary_performance_progression.png`
2. `summary-MMmodel/figures/frontend_same_split_comparison.png`
3. `summary-MMmodel/figures/fusion_mechanism_ablation.png`
4. `summary-MMmodel/figures/hcaf_current_best_architecture.png`

推荐理由:

- 第 `1` 张负责回答“当前最终模型整体是否更强”
- 第 `2` 张负责回答“为什么最终保留 `PCEN96 + HP80`”
- 第 `3` 张负责回答“多模态增益为什么不是自然出现，而依赖融合机制”
- 第 `4` 张负责回答“当前最终模型的结构到底是什么”

### 13.3 补充材料推荐插图

若作为补充实验图或答辩备份页，建议加入:

1. `summary-MMmodel/figures/filter_strategy_comparison.png`
2. `summary-MMmodel/figures/missing_modality_robustness.png`
3. `summary-MMmodel/figures/window_length_tradeoff.png`
4. `summary-MMmodel/figures/interpretability_error_summary.png`
5. `summary-MMmodel/hcaf_confgate_interpretability/attention_examples.png`
6. `summary-MMmodel/hcaf_confgate_interpretability/gate_vs_expert_confidence.png`

这些图分别对应:

- 滤波策略为何最终选择 `HP80`
- 缺失模态时系统如何退化运行
- `5 s` 与更长窗长的稳定性 / 均值权衡
- gate 行为与错误分布的精炼摘要
- cross-attention 的案例可视化
- confidence-aware gate 与 expert confidence 的关系

### 13.4 当前图版可直接支持的性能结论

基于本轮统一生成的汇总图，可在第四章中更直观地支撑以下几条结论:

- 当前最终模型 `hcaf_confgate_residual_pcen96hp80_5s` 相较上一版 HCAF best 的 session-level macro-F1 提升为 `+0.0593`
- 当前最终模型相较 PQ-only 基线的 session-level macro-F1 提升为 `+0.0889`
- 真正有效的改进主要来自 `PCEN96 + HP80`，而不是所有音频前端替换都有效
- 多模态收益依赖于 `confidence-aware gate + expert residual` 的组合，而不是普通融合自然产生
- 当前模型存在一定边界效应，但其 gate 行为与错误结构已经能够提供较一致的机制证据

## 14. 2026-04-01 自主迭代补充

在将 `hcaf_confgate_residual_pcen96hp80_5s` 作为当前最终模型后，又按“单点聚焦修改 -> 训练/评估 -> 判断是否继续”的方式补做了两轮快速验证，目标是确认当前 best 是否还能通过小改动继续刷新。

### 14.1 迭代 1：去掉 modality dropout

- 配置文件: `configs/hcaf_moddrop_search.yaml`
- 改动: 保持 `PCEN96 + HP80 + confidence-aware gate + expert residual` 全部不变，仅把 `modality_dropout` 从 `0.1` 改为 `0.0`
- 运行环境: `conda run -n dl`
- 已完成结果:
  - `repeat1_fold1`
  - window macro-F1: `0.8344`
  - session macro-F1: `0.8222`

结果解读:

- 该折明显低于当前 best 在同类实验中的首折表现。
- 更关键的是，在 `1 x 3 folds` 的平均口径下，就算剩余两折都达到 `1.0`，最终 session-level macro-F1 也最多只能追平 `0.9407`，不能严格超过当前 best。
- 因此本轮实验在首折后提前停止，不再继续消耗完整三折训练预算。

本轮结论:

- 对当前“全模态可用”的主结果口径来说，直接去掉 `modality_dropout` 没有带来收益。
- `modality_dropout=0.1` 仍更适合作为当前最终模型的默认设置。

### 14.2 迭代 2：在最终模型上试 focal loss

- 配置文件: `configs/hcaf_loss_search.yaml`
- 改动: 保持最终模型结构和前端不变，仅将损失函数改为 `focal loss (gamma=1.5)`
- 运行环境: `conda run -n dl`
- 已完成结果:
  - `repeat1_fold1`
  - window macro-F1: `0.8213`
  - session macro-F1: `0.8222`

结果解读:

- 首折 session-level macro-F1 与上一轮 `modality_dropout=0.0` 一样，仍为 `0.8222`，并没有修复当前最主要的 `0 -> 2` 边界混淆。
- 同样地，在三折均值口径下，本轮剩余两折即使全满分，也不能严格超过现有最佳结果。
- 因此本轮也在首折后提前停止。

本轮结论:

- 对当前 HCAF 最终模型而言，`focal loss` 没有复制另一条模型线上“提升 window 指标”的效果。
- 当前最佳模型仍应保持 `cross_entropy` 作为默认损失。

### 14.3 当前最终判断

综合已有主实验、补充结构搜索和本次两轮自主迭代，可以得到更稳妥的结论:

- 当前最优模型仍是 `hcaf_confgate_residual_pcen96hp80_5s`
- 新补充的两轮配置级优化都未能继续刷新结果
- 因此现阶段更值得保留的是:
  - 现有最终模型
  - 与其对应的机制解释链
  - 以及“为什么其他看似合理的改动没有继续提升”的负结果记录

下一步更有价值的方向不再是继续微调当前小超参，而是:

- 若继续提性能: 引入事件级标注或更精细的时序对齐，减少 `5 s` 固定窗边界效应
- 若继续增强证据: 扩充数据规模或增加跨采集条件验证，检验当前增益的外部稳定性
