# 硕士论文正文草稿：当前主模型结构说明

本文当前保留的主模型为 `hcaf_audio_r18img_pq_xattn_5s`。该模型以呼吸音、压力与流量三种模态为输入，在固定非对齐 `5 s` 窗口上完成 `0 / 2 / 4` 三分类任务。与早期采用轻量二维卷积音频编码器的版本不同，当前模型在呼吸音分支中使用经过 `ImageNet` 初始化的 `ResNet18` 作为特征提取骨干，并在融合阶段采用 `PQ + audio cross-attention` 结构，而非简单的向量拼接。现有实验结果表明，在统一 split manifest、统一训练预算与统一音频前端条件下，该模型在窗口级指标上同时优于 `PQ-only`、`audio-only` 和 `direct concat PQ+audio` 三类对照模型，因此可作为当前论文写作与系统说明的默认口径。

## 1. 呼吸音编码分支

当前主模型中的呼吸音并不直接以原始波形输入神经网络，而是首先转换为时频表示。具体处理流程为：先对原始音频施加 `80 Hz` 高通滤波，以抑制超低频漂移和接触噪声；随后计算 Mel 频谱，并进一步使用 `PCEN`（Per-Channel Energy Normalization）进行通道级动态归一化。该流程保留了呼吸音事件的瞬态结构，同时削弱了不同记录之间的幅值尺度差异。

音频前端参数如下：

- 采样率：`16 kHz`
- `n_fft = 1024`
- `win_length = 400`
- `hop_length = 160`
- Mel 频带数：`96`
- 频率范围：`80 Hz ~ 6000 Hz`
- 特征形式：`PCEN`

因此，单个窗口的音频输入可表示为

\[
X^{(a)} \in \mathbb{R}^{B\times 1\times 96\times T_a},
\]

其中 \(B\) 表示 batch size，96 表示 Mel 频带数，\(T_a\) 表示时间帧数。

在当前模型中，音频编码器采用 `AudioTokenEncoder(encoder_type="resnet18", pretrained=True)`。也就是说，音频分支并非使用从头训练的浅层卷积网络，而是使用经 `ImageNet` 初始化的 `ResNet18` 作为二维特征主干，并将首层卷积修改为单通道输入。若将 `ResNet18` 主干的输出特征记为

\[
F^{(a)} \in \mathbb{R}^{B\times 512\times F_a\times T_a'},
\]

则在该高层特征上，模型进一步分成 Token 分支与 Summary 分支：

1. **Token 分支**  
   通过 `AdaptiveAvgPool2d((1, 12))` 将时间维统一压缩为 12 个位置，再通过 \(1\times1\) 卷积将通道数从 512 投影到 128，得到音频 Token 序列

\[
Z^{(a)} \in \mathbb{R}^{B\times 12\times 128}.
\]

2. **Summary 分支**  
   使用 `AdaptiveAvgPool2d((1, 1))` 做全局池化，再经过线性映射与 GELU 激活，得到音频窗口级摘要向量

\[
z^{(a)} \in \mathbb{R}^{B\times 128}.
\]

这种 “Token + Summary” 的双支路设计，使音频分支既能保留局部时频结构，又能提供整窗级语义表示，为后续跨模态交互提供支持。

## 2. PQ 时序编码分支

压力与流量信号均为一维时间序列，因此在当前主模型中，二者分别使用结构同构、参数独立的 `SensorTemporalEncoder(encoder_type="tcn")` 进行建模。当前传感器采样率为 `100 Hz`，窗口长度为 `5 s`，因此单路传感器输入可表示为

\[
X^{(p)},X^{(q)} \in \mathbb{R}^{B\times 1\times 500}.
\]

PQ 编码器首先通过三层一维卷积构成的卷积前端提取局部波形结构：

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

设卷积前端后的中间特征为

\[
H_{\mathrm{stem}} \in \mathbb{R}^{B\times 64\times 125}.
\]

随后，模型使用 `TCN` 主干进一步建模局部时序结构的演化关系。当前主模型中 `tcn_layers = 2`，因此共有两个带空洞卷积的残差块，其扩张率分别为 `1` 与 `2`。若记第 \(l\) 个 TCN block 的输出为 \(H_l\)，则可写为

\[
H_l = H_{l-1} + \mathcal{D}\big(\mathcal{F}_{\mathrm{TCN}}(H_{l-1}; d_l)\big),
\]

其中 \(\mathcal{F}_{\mathrm{TCN}}\) 表示双层空洞卷积与非线性映射，\(\mathcal{D}\) 表示 Dropout 操作。该结构使模型在有限层数内获得较大的时间感受野，同时保留对局部波形变化的敏感性。

与音频分支相同，PQ 编码器最终同样输出 Token 表示和 Summary 表示：

1. **Token 分支**  
   经 `AdaptiveAvgPool1d(16)` 与 \(1\times1\) 卷积投影后，得到

\[
Z^{(p)}, Z^{(q)} \in \mathbb{R}^{B\times 16\times 128}.
\]

2. **Summary 分支**  
   经全局池化与线性映射后，得到

\[
z^{(p)}, z^{(q)} \in \mathbb{R}^{B\times 128}.
\]

因此，当前 PQ 编码器不是简单输出一个全局向量，而是同时保留传感器的局部时序 token 和窗口级 summary，为后续内部融合和跨模态交互提供两种尺度的信息。

## 3. PQ 内部融合

在当前模型中，Pressure 与 Flow 并非被直接拼接后再送入异构模态融合，而是先在传感器内部完成一次双向交互。设两路传感器 Token 序列分别为 \(Z^{(p)}\) 与 \(Z^{(q)}\)，则模型首先执行双向 cross-attention：

\[
\tilde{Z}^{(p)} = \mathrm{CA}_{p\leftarrow q}(Z^{(p)}, Z^{(q)}), \qquad
\tilde{Z}^{(q)} = \mathrm{CA}_{q\leftarrow p}(Z^{(q)}, Z^{(p)}).
\]

随后，对交互后的 Token 沿时间维平均池化，并与各自的 Summary 向量相加，再做层归一化，构造得到两路传感器的模态级表示：

\[
r^{(p)}=\mathrm{LN}\big(\mathrm{Mean}(\tilde{Z}^{(p)}) + z^{(p)}\big), \qquad
r^{(q)}=\mathrm{LN}\big(\mathrm{Mean}(\tilde{Z}^{(q)}) + z^{(q)}\big).
\]

在此基础上，模型使用门控融合模块分别在 Token 层和表示层执行融合，得到联合传感器 Token 序列 \(Z^{(s)}\) 与联合传感器表示 \(r^{(s,0)}\)。这种“先同质整合，再异质交互”的层次结构，能够先在呼吸力学信号内部完成语义协调，再与音频模态进行跨模态融合。

## 4. 音频与 PQ 的跨模态交互

当前主模型与 `direct concat` 基线最本质的区别，就在于音频与 PQ 之间是否存在显式的 cross-attention 交互。

设音频 Token 序列为 \(Z^{(a)}\)，联合传感器 Token 序列为 \(Z^{(s)}\)，则当前主模型中采用双向 cross-attention：

\[
\hat{Z}^{(a)} = \mathrm{CA}_{a\leftarrow s}(Z^{(a)}, Z^{(s)}), \qquad
\hat{Z}^{(s)} = \mathrm{CA}_{s\leftarrow a}(Z^{(s)}, Z^{(a)}).
\]

随后，模型将更新后的两路 Token 拼接，并送入一层轻量 self-attention，进一步建立统一的联合时序表示。经该步骤后，再分别构造音频侧与传感器侧的模态级表示：

\[
r^{(a)}=\mathrm{LN}\big(\mathrm{Mean}(J^{(a)}) + z^{(a)}\big), \qquad
r^{(s)}=\mathrm{LN}\big(\mathrm{Mean}(J^{(s)}) + r^{(s,0)}\big).
\]

在融合决策层面，当前主模型保留了 `confidence-aware gate + expert residual` 结构。音频表示与传感器表示分别经过独立 expert 得到两组模态级 logits：

\[
l^{(a)}=\mathrm{Expert}_a(r^{(a)}), \qquad
l^{(s)}=\mathrm{Expert}_s(r^{(s)}).
\]

模型再依据表示本身及 expert 的置信特征，输出两路权重 \(w_a\) 与 \(w_s\)，据此构造融合表示：

\[
r^{(f)}=\psi\big(w_a r^{(a)} + w_s r^{(s)}\big).
\]

最终输出形式为：

\[
y=\mathrm{Classifier}(r^{(f)}) + \lambda\big(w_a l^{(a)} + w_s l^{(s)}\big),
\]

其中 \(\lambda = 0.3\) 为 expert residual 的缩放系数。该设计使模型不仅能够完成模态级融合，还能保留各模态单独判别的辅助证据，从而提升窗口级预测稳定性。

## 5. 与 Direct Concat 基线的差异

为验证 `cross-attention` 的必要性，当前版本还专门引入了一个同编码器、同前端、同训练预算下的 `direct concat` 基线：

- 保留同样的 `ResNet18` 音频分支
- 保留同样的 `PQ TCN` 传感器分支
- 保留 `Pressure <-> Flow` 内部交互
- 去掉 `audio <-> sensor` cross-attention
- 改为直接拼接 `audio_repr` 与 `sensor_repr` 后分类

因此，这个基线并不是“完全不同的模型”，而是只移除了音频与 PQ 之间的显式跨模态交互。换句话说，它是一个更公平的消融对照。

## 6. 当前结果对应的结构结论

在统一的 `5 s` 固定非对齐窗条件下，正式三折结果如下：

| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `pressure_flow_5s` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` |
| `hcaf_audio_r18img_audio_only_5s` | `0.8709 ± 0.0722` | `0.9407 ± 0.0838` |
| `hcaf_audio_r18img_pq_directconcat_5s` | `0.7800 ± 0.1610` | `0.7852 ± 0.1923` |
| `hcaf_audio_r18img_pq_xattn_5s` | `0.9145 ± 0.0745` | `0.9407 ± 0.0838` |

窗口级差值为：

- `cross-attention - audio-only = +0.0436`
- `cross-attention - PQ-only = +0.1646`
- `cross-attention - direct concat = +0.1345`

这表明：

1. 当前主模型并非只是“音频很强，PQ 随便加一点”  
   因为如果仅仅依赖音频，`audio-only` 已经足够强，但仍低于 `cross-attention`。

2. 当前主模型也不是“只靠传感器”  
   因为 `PQ-only` 明显低于 `cross-attention`。

3. 真正起作用的并不是“同样 encoder 下简单拼接”  
   因为 `direct concat` 明显低于 `cross-attention`。

因此，当前可以合理地将 `PQ + audio cross-attention` 视为窗口级判别任务下的默认主结构。

## 7. 当前主模型的局限

尽管当前模型在窗口级上已经优于 `PQ-only`、`audio-only` 和 `direct concat`，但仍存在以下限制：

- `session-level` 上，`cross-attention` 与 `audio-only` 当前打平
- 更长固定非对齐窗的完整三折结果仍未全部补齐
- 周期级表示虽然有生理合理性，但在当前 `ResNet18` 路线下尚未形成更强的多模态主线
- 当前 `sensor` 分支仍可能是后续进一步提升的主要瓶颈

## 8. 可直接用于论文的简洁表述

> 在固定非对齐 `5 s` 窗条件下，本文采用 `ResNet18`（ImageNet 初始化）作为呼吸音编码骨干，并使用 `PQ + audio cross-attention` 进行多模态融合。实验结果表明，该模型在窗口级 macro-F1 上达到 `0.9145 ± 0.0745`，高于 `audio-only` 的 `0.8709 ± 0.0722`、`PQ-only` 的 `0.7499 ± 0.2513`，也高于移除跨模态交互后的 `direct concat PQ+audio` 基线 `0.7800 ± 0.1610`。这说明，对当前任务而言，音频与呼吸力学信号之间的显式跨模态交互比简单的特征拼接更有效。
