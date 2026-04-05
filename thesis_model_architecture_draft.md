# 硕士论文正文草稿：当前主模型结构详述

> 权威声明：论文正文当前默认模型统一为 `HCAF-PCEN-DualXAttn`，实验 ID 为 `hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s`。如果仓库其他历史文档还出现 `HCAF-LogMel96-DualXAttn`，那是旧口径，不应再作为正文默认模型。

本文当前保留的默认主模型在正文中记为 `HCAF-PCEN-DualXAttn`，其对应实验配置 ID 为 `hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s`。该模型以呼吸音、气道压力（Pressure）与流量（Flow）三种模态为输入，在固定非对齐 `5 s` 时间窗上完成 `0 / 2 / 4` 三分类任务。它建立在最终 HCAF 主线的基础上，并保留了最值得强调的三组核心机制：`PCEN96 + HP80` 音频前端、`Pressure-Flow` 内部交互与 `audio-sensor` 双向 cross-attention、以及决策阶段的 `confidence-aware gate + expert residual`。在最终统一证据表中，该模型的 window-level macro-F1 为 `0.9196 ± 0.0469`，高于同前端的 `audio_only` 与 `pressure_flow` 基线，因此当前更适合作为论文正文中的默认模型口径。

为便于后续表述，本文约定如下记号：

- 音频时频特征张量记为 \([B,C,F,T]\)，其中 \(B\) 为 batch size，\(C\) 为通道数，\(F\) 为频率维，\(T\) 为时间帧数。
- 压力与流量一维信号张量记为 \([B,C,L]\)，其中 \(L\) 为时间长度。
- Token 序列记为 \([B,N,D]\)，其中 \(N\) 为 token 个数，\(D\) 为嵌入维度。

## 1. 整体结构与数据流

当前主模型的整体处理流程可概括为：

1. 对每个 `session` 的三种模态裁剪到共同可覆盖的最短时长。
2. 按固定非对齐 `5 s` 窗进行切窗，窗移同样为 `5 s`，即窗口之间不重叠。
3. 呼吸音窗口经 `HP80 + Mel + PCEN + 标准化` 形成二维时频输入，再送入 `ResNet18` 音频编码器。
4. 压力与流量窗口分别经 `1D CNN stem + TCN` 编码，得到各自的局部时序 token 与全局摘要向量。
5. 先对 `Pressure` 与 `Flow` 做双向 cross-attention，再通过门控融合形成联合传感器表示。
6. 再对 `audio` 与 `sensor` 做双向 cross-attention；在当前默认压缩版中，不再对拼接后的联合 token 追加 self-attention。
7. 最后用 `confidence-aware gate` 自适应融合音频侧与传感器侧决策证据，并通过 `expert residual` 保留模态专家的直接判别能力。

若以张量流表示，则主干可写为：

\[
\{X^{(a)}, X^{(p)}, X^{(q)}\}
\xrightarrow{\text{encoders}}
\{Z^{(a)}, z^{(a)}, Z^{(p)}, z^{(p)}, Z^{(q)}, z^{(q)}\}
\xrightarrow{\text{PQ interaction}}
\{Z^{(s)}, r^{(s,0)}\}
\xrightarrow{\text{audio-sensor interaction}}
\{r^{(a)}, r^{(s)}\}
\xrightarrow{\text{conf-aware fusion}}
y.
\]

其中，\(Z\) 表示 token 序列，\(z\) 表示分支自身的 summary 向量，\(r\) 表示经过交互与归一化后的模态级表示，\(y\) 为最终分类 logits。

当前主模型的关键结构超参数如下：

- 嵌入维度：`128`
- cross-attention 头数：`4`
- dropout：`0.3`
- 音频 token 数：`12`
- 传感器 token 数：`16`
- 联合 self-attention 层数：`0`
- modality dropout：`0.1`
- use_summary_in_repr：`false`
- expert residual scale：`0.3`

## 2. 输入构建与窗口级对齐方式

### 2.1 多模态切窗原则

当前任务并不是先独立处理每个模态，再在后期粗略对齐，而是在切窗前先对三种模态进行共同裁剪。设原始音频长度为 \(L_a^{\text{raw}}\)，压力长度为 \(L_p^{\text{raw}}\)，流量长度为 \(L_q^{\text{raw}}\)，则首先根据各自采样率转换为秒级时长，并取三者最小值：

\[
T_{\text{common}} = \min\left(\frac{L_a^{\text{raw}}}{16000}, \frac{L_p^{\text{raw}}}{100}, \frac{L_q^{\text{raw}}}{100}\right).
\]

之后仅保留三种模态在 \([0, T_{\text{common}}]\) 内共同存在的片段，从而保证同一窗口中的呼吸音、压力、流量严格对应于同一时间段。这一设计避免了多模态后融合时由于尾部长度不一致带来的伪对齐问题。

### 2.2 窗口大小

当前固定窗长度为 `5 s`，因此：

- 呼吸音窗口长度为 `5 x 16000 = 80000` 个采样点；
- 压力窗口长度为 `5 x 100 = 500` 个采样点；
- 流量窗口长度为 `5 x 100 = 500` 个采样点。

因此，原始输入可分别记为

\[
\mathbf{x}^{(a)} \in \mathbb{R}^{B\times 80000}, \qquad
\mathbf{x}^{(p)}, \mathbf{x}^{(q)} \in \mathbb{R}^{B\times 1\times 500}.
\]

### 2.3 输入标准化

在进入模型前，各模态还进行了基础标准化处理：

- 呼吸音波形先做 `z-score` 标准化；
- 压力与流量分别按各自整段序列做 `z-score` 标准化；
- 呼吸音在前端变换后，还会再对最终时频特征做一次零均值单位方差标准化。

这样做的目的不是替代模型学习，而是削弱不同记录之间的绝对量纲和增益差异，使网络更关注呼吸事件形态、波形结构及跨模态对应关系。

## 3. 呼吸音编码分支

呼吸音分支是当前主模型中最强的单模态支路之一，但其作用并不只是提供一个最终向量，而是同时输出跨模态交互所需的 token 序列和整窗摘要表示。该分支由“音频前端”和“二维卷积骨干”两部分组成。

### 3.1 音频前端处理流程

当前主模型的音频前端参数如下：

- 采样率：`16 kHz`
- `n_fft = 1024`
- `win_length = 400`
- `hop_length = 160`
- Mel 频带数：`96`
- 频率范围：`80 Hz ~ 6000 Hz`
- 特征类型：`PCEN`
- 高通滤波：`80 Hz`
- `delta_order = 0`

其处理顺序为：

1. 对原始波形施加 `80 Hz` 高通滤波，抑制超低频漂移、机械接触噪声和基线波动；
2. 计算 `96` 维 Mel 时频能量图；
3. 对 Mel 能量图执行 `PCEN`（Per-Channel Energy Normalization）；
4. 对最终特征做全局标准化；
5. 作为单通道二维张量送入 `ResNet18`。

需要说明的是，本文仅将 Mel 频谱视为构建时频输入的中间表示，不再展开其原理；真正需要重点解释的是 `PCEN`，因为它是当前主模型音频前端的重要技术亮点之一。

由于当前未引入 `delta` 或 `delta-delta` 通道，因此音频前端输出通道数为 `1`，这也与后续 `ResNet18` 首层单通道卷积设置保持一致。

### 3.2 PCEN 的定义与作用

设 Mel 频谱能量为 \(S_{f,t}\)，其中 \(f\) 表示频带索引，\(t\) 表示时间帧索引。`PCEN` 先对每个频带构造一个时间递推的平滑包络：

\[
M_{f,t} =
\begin{cases}
S_{f,0}, & t = 0, \\
(1-s) M_{f,t-1} + s S_{f,t}, & t > 0,
\end{cases}
\]

其中 \(s\) 为平滑系数。当前实现中：

\[
s = 0.025.
\]

在得到平滑包络 \(M_{f,t}\) 后，`PCEN` 输出定义为

\[
\mathrm{PCEN}(S_{f,t})
=
\left(
\frac{S_{f,t}}{(\varepsilon + M_{f,t})^{\alpha}} + \delta
\right)^r
- \delta^r,
\]

其中当前主模型采用：

\[
\alpha = 0.98,\qquad
\delta = 2.0,\qquad
r = 0.5,\qquad
\varepsilon = 10^{-6}.
\]

从形式上看，`PCEN` 具有三个作用：

1. 用 \(M_{f,t}\) 对当前帧做自适应归一化，减弱慢变背景能量和录音增益差异；
2. 用 \(\delta\) 与 \(r\) 引入可调的动态范围压缩，避免少数高能量帧支配特征；
3. 强调相对突变和局部增强结构，使吸气、呼气、液体相关异常声等瞬态事件更容易被后续二维卷积捕获。

与简单对数压缩相比，`PCEN` 并非固定的静态映射，而是带有时间递推平滑项的动态归一化，因此更适合处理不同记录间振幅尺度波动较大的呼吸音。

### 3.3 呼吸音输入尺寸

对单个 `5 s` 音频窗口，原始波形长度为 `80000` 点。按照当前 `STFT/Mel` 参数，在默认中心填充设置下，时间帧数为：

\[
T_a
=
\left\lfloor
\frac{80000 + 2\times(1024/2) - 1024}{160}
\right\rfloor + 1
= 501.
\]

因此，经过前端后的音频输入张量尺寸为

\[
X^{(a)} \in \mathbb{R}^{B\times 1\times 96\times 501}.
\]

其中：

- `1` 表示单通道时频图；
- `96` 表示 Mel 频带数；
- `501` 表示时间帧数。

### 3.4 ResNet18 音频骨干

当前主模型中的音频编码器为：

```text
AudioTokenEncoder(
    encoder_type="resnet18",
    pretrained=True,
    in_channels=1,
    token_frames=12,
    embedding_dim=128
)
```

这意味着音频分支不是从头训练的浅层 CNN，而是直接采用 `torchvision ResNet18` 的主干结构：

```text
conv1 -> bn1 -> relu -> maxpool
-> layer1 -> layer2 -> layer3 -> layer4
```

但与标准图像模型相比，有一个重要改动：原始 `ResNet18` 的首层卷积接收 `RGB 3` 通道输入，而当前模型的音频时频图只有单通道，因此将 `conv1` 改为单通道输入，并在 `ImageNet` 预训练场景下，将原 RGB 卷积核在通道维上求均值后复制到新卷积层。这一设计兼顾了两点：

1. 保留 `ImageNet` 预训练带来的稳定初始化优势；
2. 使模型能够直接适配单通道呼吸音时频图，而无需伪造三通道输入。

对当前输入尺寸 \(B\times1\times96\times501\)，`ResNet18` 主干输出的高层特征尺寸为

\[
F^{(a)} \in \mathbb{R}^{B\times 512\times 3\times 16}.
\]

该输出具有两个关键含义：

- 通道维 `512` 表示主干已经提取到较高层次的时频语义；
- 时间维从 `501` 压缩到 `16`，意味着网络已完成较强的时间抽象；
- 频率维从 `96` 压缩到 `3`，使后续模块更关注高层局部模式，而不是逐频带细节。

### 3.5 Token 分支与 Summary 分支

为了兼顾局部交互与全局判别，音频骨干输出后被拆成两条支路。

#### （1）Token 分支

首先对 \(F^{(a)}\) 进行自适应池化：

\[
\bar{F}^{(a)} = \mathrm{AdaptiveAvgPool2d}_{(1,12)}(F^{(a)})
\in \mathbb{R}^{B\times512\times1\times12}.
\]

随后压缩掉频率维，并使用 \(1\times1\) 卷积做通道投影：

\[
Z^{(a)} =
\mathrm{Transpose}\left(
\mathrm{Conv1d}_{512\rightarrow128}(\bar{F}^{(a)}_{\text{squeeze}})
\right)
\in \mathbb{R}^{B\times12\times128}.
\]

因此，音频分支最终输出 `12` 个 token，每个 token 的维度为 `128`。这些 token 对应于整窗内经过主干压缩后的 `12` 个时间位置，是后续跨模态交互的核心载体。

#### （2）Summary 分支

与此同时，音频分支还保留一个整窗级摘要向量：

\[
z^{(a)} =
\mathrm{GELU}\left(
W_a \cdot \mathrm{GAP}(F^{(a)})
\right)
\in \mathbb{R}^{B\times128},
\]

其中 `GAP` 表示 `AdaptiveAvgPool2d((1,1))` 后展平，\(W_a\) 为线性映射。该向量用于保留整窗级的稳定语义，在最后构建模态表示时与 token 平均池化结果相加。

### 3.6 呼吸音分支的技术亮点

当前音频分支的关键优势不只是“用了更强的骨干”，而是以下几项设计共同作用的结果：

1. `PCEN` 使前端具备动态归一化能力，而非固定对数压缩；
2. `ResNet18 + ImageNet init` 提升了二维时频特征提取能力；
3. 首层单通道改造保留了预训练权重的可迁移性；
4. `Token + Summary` 双输出使音频分支既能参与细粒度跨模态交互，又能保留整窗稳定判别语义。

## 4. Pressure / Flow 时序编码分支

压力与流量均为一维呼吸力学信号，因此当前模型为这两种传感器分别使用结构同构、参数独立的 `SensorTemporalEncoder(encoder_type="tcn")`。这种设计既允许二者学习各自的物理形态特征，又保证它们在融合前处于同一表示空间。

### 4.1 输入尺寸

当前传感器采样率为 `100 Hz`，窗口长度为 `5 s`，因此单路传感器输入为：

\[
X^{(p)}, X^{(q)} \in \mathbb{R}^{B\times1\times500}.
\]

其中 `500` 表示窗口内的时间采样点数。

### 4.2 卷积前端（1D CNN stem）

每一路传感器首先经过三层一维卷积前端：

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

该前端的作用可以概括为：

1. 第一层用较大卷积核 `9` 捕获局部波形形态与呼吸相位边界附近的短时模式；
2. 后两层用步长 `2` 逐步下采样，降低时间分辨率并扩大有效感受野；
3. 通道数逐步从 `1` 提升到 `64`，使模型从原始波形过渡到高维时序特征。

长度变化过程为：

\[
500 \xrightarrow{\text{stride }1} 500
\xrightarrow{\text{stride }2} 250
\xrightarrow{\text{stride }2} 125.
\]

因此卷积前端输出为

\[
H_{\text{stem}}^{(p)}, H_{\text{stem}}^{(q)}
\in \mathbb{R}^{B\times64\times125}.
\]

### 4.3 TCN 主干

在卷积前端之后，模型使用 `TCN` 建模中尺度时序演化关系。当前主模型配置为：

- `tcn_layers = 2`
- dilation 分别为 `1` 和 `2`
- 每个 block 内部包含两层 `kernel_size=3` 的空洞卷积

若记第 \(l\) 个 TCN block 的输入为 \(H_{l-1}\)，则其输出可表示为：

\[
H_l = H_{l-1} + \mathcal{F}_{\text{TCN}}(H_{l-1}; d_l),
\]

其中 \(d_l\) 为第 \(l\) 层空洞率，\(\mathcal{F}_{\text{TCN}}\) 表示

\[
\mathcal{F}_{\text{TCN}}(\cdot; d)
=
\mathrm{GELU}
\circ \mathrm{BN}
\circ \mathrm{Conv1d}_{k=3,d}
\circ \mathrm{Dropout}
\circ \mathrm{GELU}
\circ \mathrm{BN}
\circ \mathrm{Conv1d}_{k=3,d}.
\]

因此当前两层 `TCN` 的扩张率依次为

\[
d_1 = 1,\qquad d_2 = 2.
\]

该结构相较于普通连续卷积有两个优点：

1. 用有限层数获得更大的时间感受野；
2. 通过残差连接保留原始局部波形细节，减轻深层时序建模带来的信息损失。

在当前实现中，`TCN` 不改变通道数与长度，因此主干输出仍为

\[
H^{(p)}, H^{(q)} \in \mathbb{R}^{B\times64\times125}.
\]

### 4.4 Token 与 Summary 输出

与音频分支一致，传感器编码器并不只输出一个全局向量，而是同时输出 token 序列和 summary 向量。

#### （1）Token 分支

对时序特征执行自适应池化：

\[
\bar{H}^{(p)} = \mathrm{AdaptiveAvgPool1d}_{16}(H^{(p)}),
\qquad
\bar{H}^{(q)} = \mathrm{AdaptiveAvgPool1d}_{16}(H^{(q)}),
\]

并通过 \(1\times1\) 卷积投影到统一嵌入维度：

\[
Z^{(p)}, Z^{(q)}
\in \mathbb{R}^{B\times16\times128}.
\]

这意味着每一路传感器最终保留 `16` 个局部时序 token，用于后续的 cross-attention。

#### （2）Summary 分支

同时，编码器对整段时序特征做全局池化与线性映射：

\[
z^{(p)} = \mathrm{GELU}(W_p \cdot \mathrm{GAP}(H^{(p)})) \in \mathbb{R}^{B\times128},
\]
\[
z^{(q)} = \mathrm{GELU}(W_q \cdot \mathrm{GAP}(H^{(q)})) \in \mathbb{R}^{B\times128}.
\]

因此，PQ 编码器保留了两个粒度的信息：

- token 级局部时序结构；
- 窗口级全局动力学摘要。

这也是后续能够先做 `Pressure-Flow` 内部交互，再做异构模态融合的重要基础。

### 4.5 PQ 分支的技术亮点

PQ 分支的优势并不只是“对两个传感器分别编码”，而是体现在：

1. `1D CNN stem` 先提取局部波形形态并完成时间压缩；
2. `TCN` 在保持长度不变的同时扩展时序感受野；
3. 每一路都输出 `token + summary`，使传感器分支具备与音频对等的交互能力；
4. `Pressure` 与 `Flow` 参数独立，避免二者被强行绑定到同一卷积滤波器上。

## 5. PQ 内部融合：先同质交互，再形成联合传感器表示

在当前主模型中，`Pressure` 与 `Flow` 并不是编码完成后直接拼接，而是先进行显式双向交互。这是因为二者同属于呼吸力学模态，具有更强的物理耦合关系，先完成同质模态内部语义协调，再与音频交互，通常比三路特征一次性混合更稳定。

### 5.1 Cross-Attention Block 的数学形式

当前模型中的 cross-attention block 采用 `pre-norm + multi-head attention + residual + FFN` 结构。给定 query 序列 \(Q\in\mathbb{R}^{B\times N_q\times D}\) 与 context 序列 \(C\in\mathbb{R}^{B\times N_c\times D}\)，定义：

\[
\mathrm{CA}(Q,C)
=
\mathrm{FFN}\left(
Q + \mathrm{Drop}\left(
\mathrm{MHA}(\mathrm{LN}(Q), \mathrm{LN}(C), \mathrm{LN}(C))
\right)
\right).
\]

其中 `MHA` 为 `4` 头多头注意力，嵌入维度 \(D=128\)。其后的前馈网络 `FFN` 采用 `128 -> 512 -> 128` 的两层映射，即：

\[
\mathrm{FFN}(X)
=
X + \mathrm{Drop}\left(
W_2\,
\mathrm{Drop}\left(
\mathrm{GELU}(W_1\,\mathrm{LN}(X))
\right)
\right),
\]

其中 \(W_1:128\rightarrow512\)，\(W_2:512\rightarrow128\)。

### 5.2 Pressure 与 Flow 的双向交互

设两路传感器 token 分别为 \(Z^{(p)}\) 与 \(Z^{(q)}\)，对应模态可用性掩码为 \(m_p,m_q\in\{0,1\}\)，则代码中的双向交互可写为：

\[
\tilde{Z}^{(p)} = \mathrm{CA}(Z^{(p)}, Z^{(q)}), \qquad
\tilde{Z}^{(q)} = \mathrm{CA}(Z^{(q)}, Z^{(p)}).
\]

但真实实现并不是无条件替换，而是采用掩码控制的更新策略：

\[
Z_{\text{new}}^{(p)}
=
m_p\big(m_q\tilde{Z}^{(p)} + (1-m_q)Z^{(p)}\big),
\]
\[
Z_{\text{new}}^{(q)}
=
m_q\big(m_p\tilde{Z}^{(q)} + (1-m_p)Z^{(q)}\big).
\]

这一步的含义是：

- `Pressure <- Flow`：压力 token 在流量上下文中更新；
- `Flow <- Pressure`：流量 token 在压力上下文中更新。
- 若某一路在训练时被 `modality dropout` 掩掉，则不会强行执行无意义的跨模态更新，而是退化为保留仍然可用的那一路表示。

因此，交互后的每一路 token 不再是“孤立传感器特征”，而是已经吸收了另一种呼吸力学信号提供的上下文信息。

### 5.3 分支级表示构造

交互后的 token 先沿 token 维求平均，再与本支路 summary 向量相加，并通过层归一化得到模态级表示：

\[
r^{(p)}
=
\mathrm{LN}\left(
\mathrm{Mean}(Z_{\text{new}}^{(p)}) + z^{(p)}
\right),
\]
\[
r^{(q)}
=
\mathrm{LN}\left(
\mathrm{Mean}(Z_{\text{new}}^{(q)}) + z^{(q)}
\right).
\]

其中，`Mean(token) + summary` 的组合有明确目的：

- `Mean(token)` 保留交互后的局部时序统计；
- `summary` 保留该模态自身的全局信息；
- 二者相加后再归一化，可以在不引入额外参数的情况下形成稳定的模态级表示。

### 5.4 Mask-aware 门控融合

当前主模型在传感器内部融合时没有直接平均，而是采用 `MaskedTokenGate`，并且在 token 层与 repr 层各执行一次。设左右输入分别为 \(\mathbf{u}\) 与 \(\mathbf{v}\)，对应可用性掩码为 \(m_u,m_v\in\{0,1\}\)，则先定义：

\[
m_{\text{both}} = m_u m_v,\qquad
m_{u\text{-only}} = m_u(1-m_v),\qquad
m_{v\text{-only}} = m_v(1-m_u).
\]

当两侧都存在时，门控值由拼接表示产生：

\[
g = \sigma(\mathrm{MLP}([\mathbf{u};\mathbf{v}])).
\]

在当前实现中，该门控网络具体采用：

\[
\mathrm{LayerNorm}(256)
\rightarrow
\mathrm{Linear}(256,128)
\rightarrow
\mathrm{GELU}
\rightarrow
\mathrm{Dropout}
\rightarrow
\mathrm{Linear}(128,128)
\rightarrow
\mathrm{Sigmoid}.
\]

融合结果写为：

\[
\mathbf{h}
=
m_{\text{both}}\cdot
\big(g\odot \mathbf{u} + (1-g)\odot \mathbf{v}\big)
+
m_{u\text{-only}}\cdot \mathbf{u}
+
m_{v\text{-only}}\cdot \mathbf{v}.
\]

最后再经输出映射得到：

\[
\mathrm{Fuse}(\mathbf{u},\mathbf{v})
=
\phi(\mathbf{h})\cdot \min(1,m_u+m_v).
\]

这一设计有两个非常关键的意义：

1. 当两路传感器都可用时，由网络自适应决定保留多少 `Pressure` 信息、保留多少 `Flow` 信息；
2. 当某一路被训练时的 `modality dropout` 掩掉时，融合模块会自动退化为“只传递仍然存在的那一路”，而不是生成无意义的混合表示。

经过 token 级和表示级融合后，得到：

\[
Z^{(s)} \in \mathbb{R}^{B\times16\times128},\qquad
r^{(s,0)} \in \mathbb{R}^{B\times128}.
\]

更具体地说，

\[
Z^{(s)} = \mathrm{Fuse}_{\text{token}}\big(Z_{\text{new}}^{(p)}, Z_{\text{new}}^{(q)}\big),
\]
\[
r^{(s,0)} = \mathrm{Fuse}_{\text{repr}}\big(r^{(p)}, r^{(q)}\big).
\]

这两个量分别表示联合传感器 token 序列与联合传感器初始表示。

## 6. 音频与联合传感器的跨模态交互

在完成 `Pressure-Flow` 内部融合后，模型才进入异构模态交互阶段。当前主模型与 `direct concat` 基线的根本差别，就在于此处是否存在显式的 `audio-sensor cross-attention`。

### 6.1 双向 cross-attention

设音频 token 为 \(Z^{(a)}\in\mathbb{R}^{B\times12\times128}\)，联合传感器 token 为 \(Z^{(s)}\in\mathbb{R}^{B\times16\times128}\)，音频与传感器可用性掩码分别为 \(m_a,m_s\in\{0,1\}\)，则双向交互的候选更新为：

\[
\hat{Z}^{(a)} = \mathrm{CA}(Z^{(a)}, Z^{(s)}), \qquad
\hat{Z}^{(s)} = \mathrm{CA}(Z^{(s)}, Z^{(a)}).
\]

真实实现中的最终更新则为：

\[
Z_{\text{new}}^{(a)}
=
m_a\big(m_s\hat{Z}^{(a)} + (1-m_s)Z^{(a)}\big),
\]
\[
Z_{\text{new}}^{(s)}
=
m_s\big(m_a\hat{Z}^{(s)} + (1-m_a)Z^{(s)}\big).
\]

这意味着：

- 音频 token 在传感器上下文中被重加权；
- 传感器 token 在音频上下文中被重加权。
- 若训练时某一侧模态被掩掉，则另一侧保持原有 token，而不会被迫和零表示做交互。

因此，模型不再把“音频特征”和“PQ 特征”视为两个互不相干的向量，而是允许一个模态在编码后的 token 层主动读取另一模态的局部证据。

### 6.2 联合 token 组织

双向 cross-attention 完成后，模型将两路 token 拼接为统一序列：

\[
J = [Z_{\text{new}}^{(a)};Z_{\text{new}}^{(s)}]
\in \mathbb{R}^{B\times(12+16)\times128}
= \mathbb{R}^{B\times28\times128}.
\]

在当前默认模型中，拼接后的联合 token 不再继续送入额外的 self-attention 层，而是直接保留为

\[
J' = J.
\]

这样处理的原因是：补充压缩实验表明，保留前面的双阶段 cross-attention 已足以完成主要的跨模态证据交换，而去掉 concat 之后的联合 self-attention 既能减少参数与计算量，也没有破坏当前最佳窗口级结果。

### 6.3 交互后的模态级表示

设从 \(J'\) 中重新切分出的音频 token 与传感器 token 分别为 \(J^{(a)}\) 与 \(J^{(s)}\)，则最终模态级表示构造为：

\[
r^{(a)}
=
\mathrm{LN}\left(
\mathrm{Mean}(J^{(a)})
\right),
\]
\[
r^{(s)}
=
\mathrm{LN}\left(
\mathrm{Mean}(J^{(s)})
\right).
\]

也就是说，当前默认模型在表示层不再显式叠加原始 `summary` 残差，而是直接使用跨模态更新后的 token 均值作为模态级表示。补充压缩实验说明，这样的表示构造反而更稳定，也更符合当前正文希望强调的“核心证据来自前端与跨模态交互本身”这一叙述。

## 7. 决策层：Confidence-Aware Gate 与 Expert Residual

当前主模型并未在得到 \(r^{(a)}\) 与 \(r^{(s)}\) 后直接拼接分类，而是采用“模态专家 + 置信感知门控 + 专家残差”的层次化决策方式。这是当前结构中非常重要、也很容易在简写中被忽略的技术亮点。

### 7.1 模态专家

首先，音频侧与传感器侧分别通过独立线性分类头得到模态级 logits：

\[
l^{(a)} = \mathrm{Expert}_a(r^{(a)}), \qquad
l^{(s)} = \mathrm{Expert}_s(r^{(s)}),
\]

其中

\[
l^{(a)}, l^{(s)} \in \mathbb{R}^{B\times3}.
\]

这里的作用是让每个模态先给出“自己单独怎么看”的判别证据，而不是一开始就完全交给融合层决定。

### 7.2 置信特征构造

对任一模态 logits \(l\)，令

\[
p = \mathrm{softmax}(l).
\]

随后提取三类置信特征：

1. 最大类别概率

\[
c_{\text{top1}} = \max_c p_c;
\]

2. 前两类概率差值（margin）

\[
c_{\text{margin}} = p_{(1)} - p_{(2)};
\]

3. 归一化熵对应的置信度

\[
H(p) = -\sum_{c=1}^{3} p_c \log p_c,
\qquad
c_{\text{ent}} = 1 - \frac{H(p)}{\log 3}.
\]

因此，每个模态的置信特征为

\[
c^{(a)} = [c_{\text{top1}}^{(a)}, c_{\text{margin}}^{(a)}, c_{\text{ent}}^{(a)}] \in \mathbb{R}^{B\times3},
\]
\[
c^{(s)} = [c_{\text{top1}}^{(s)}, c_{\text{margin}}^{(s)}, c_{\text{ent}}^{(s)}] \in \mathbb{R}^{B\times3}.
\]

这意味着门控网络并不是只看表示向量本身，而是同时利用了“该模态当前判断是否确定”这一决策层信息。

### 7.3 Confidence-Aware Gate

当前门控网络以两路模态表示与对应置信特征拼接作为输入：

\[
[r^{(a)}; r^{(s)}; c^{(a)}; c^{(s)}]
\in \mathbb{R}^{B\times(128+128+3+3)}.
\]

经过两层感知机后得到两路原始权重，再经 softmax 归一化为：

\[
[w_a, w_s]
=
\mathrm{softmax}
\left(
\mathrm{MLP}\big([r^{(a)}; r^{(s)}; c^{(a)}; c^{(s)}]\big)
\right).
\]

其对应的门控网络结构为：

\[
\mathrm{LayerNorm}(262)
\rightarrow
\mathrm{Linear}(262,128)
\rightarrow
\mathrm{GELU}
\rightarrow
\mathrm{Dropout}
\rightarrow
\mathrm{Linear}(128,2).
\]

随后得到融合表示：

\[
r^{(f)} = \psi\big(w_a r^{(a)} + w_s r^{(s)}\big),
\]

其中 \(\psi(\cdot)\) 表示门控后的输出映射，即 `LayerNorm + Linear + GELU + Dropout`。

与普通 softmax gate 相比，这里的门控至少有两个优势：

1. 不仅根据表示内容决定权重，还会参考模态当前的分类置信度；
2. 当某一路模态在训练时被 `modality dropout` 掩掉时，门控会借助掩码自动退化为单模态决策。

### 7.4 Expert Residual

在得到融合表示 \(r^{(f)}\) 后，模型先通过最终分类器输出主干 logits：

\[
l^{(f)} = \mathrm{Classifier}(r^{(f)}).
\]

当前主模型的最终分类器结构为：

\[
\mathrm{Linear}(128,128)
\rightarrow
\mathrm{GELU}
\rightarrow
\mathrm{Dropout}
\rightarrow
\mathrm{Linear}(128,3).
\]

然后再将模态专家 logits 以门控权重加权后作为残差项叠加：

\[
l^{(\text{exp})} = w_a l^{(a)} + w_s l^{(s)}.
\]

最终输出为

\[
y = l^{(f)} + \lambda\, l^{(\text{exp})},
\]

其中当前主模型中

\[
\lambda = 0.3.
\]

`expert residual` 的作用是：即使最终融合表示已经形成，模型仍然保留“音频专家”和“传感器专家”的直接判别证据，防止融合层过度平滑掉某一路模态已经十分明确的判断。

## 8. Modality Dropout 与缺失模态鲁棒性

当前模型在训练阶段还使用了 `modality_dropout = 0.1`。其实现方式不是在输入层简单加噪，而是在编码后对每种模态采样可用性掩码：

\[
m_a, m_p, m_q \in \{0,1\}.
\]

对当前启用的模态，保留概率为

\[
P(m=1) = 0.9.
\]

并且会额外保证每个样本至少保留一种模态，避免出现三路全部被置零的无效训练样本。

得到掩码后，音频、压力、流量的 token 与 summary 都会被对应掩码相乘。随后，`MaskedTokenGate` 与 `confidence-aware gate` 会根据掩码自动处理“单模态可用”或“部分模态缺失”的情形。也就是说，当前主模型不仅追求多模态最优性能，还显式考虑了训练时的模态扰动鲁棒性。

这也是为什么当前融合模块采用 mask-aware 设计，而不是默认所有模态始终存在。

## 9. 与 Direct Concat 基线的差异

为了证明性能提升来自“显式跨模态交互”，而不是仅仅来自“模态更多”或“编码器更大”，当前版本保留了一个高度公平的 `direct concat` 基线，即 `hcaf_audio_r18img_pq_directconcat_5s`。该基线与主模型的相同点包括：

- 相同的呼吸音前端：`HP80 + 96-mel + PCEN`
- 相同的音频骨干：`ResNet18 (ImageNet init)`
- 相同的压力与流量编码器：`1D CNN stem + TCN`
- 相同的 `Pressure <-> Flow` 双向交互
- 相同的数据划分、训练轮数和训练预算

不同点则集中在最后的音频-传感器融合阶段：

1. 移除了 `audio <-> sensor` 双向 cross-attention；
2. 不再对拼接后的联合 token 做 self-attention；
3. 不再使用 `confidence-aware gate`；
4. 不再使用 `expert residual`；
5. 改为直接拼接音频表示与传感器表示后分类：

\[
y_{\text{concat}}
=
\mathrm{MLP}\big([r^{(a)}; r^{(s)}]\big).
\]

其中该 `MLP` 的结构为：

\[
\mathrm{Linear}(256,128)
\rightarrow
\mathrm{GELU}
\rightarrow
\mathrm{Dropout}
\rightarrow
\mathrm{Linear}(128,3).
\]

因此，该基线并不是一个完全不同范式的模型，而是仅删除“显式异构交互与分层决策”后得到的消融版本。这使得 `cross-attention` 主模型的收益更具解释力。

## 10. 当前主模型的核心技术亮点

综合上述结构，当前主模型相较于普通多模态拼接网络的技术亮点可以概括为以下几点：

1. **音频前端采用 PCEN 而非简单对数压缩**  
   它通过时间递推的平滑包络完成动态归一化，更适合处理呼吸音记录间的增益差异和背景能量波动。

2. **呼吸音分支采用单通道改造的 ImageNet 预训练 ResNet18**  
   该设计兼顾预训练迁移能力与单通道声学输入的适配性。

3. **三路编码器全部输出 Token + Summary 双尺度表示**  
   这样既支持局部跨模态交互，也保留整窗级稳定摘要。

4. **先做同质模态融合，再做异质模态融合**  
   `Pressure` 与 `Flow` 先在传感器内部协调，再与呼吸音交互，更符合呼吸力学与声学之间的层次关系。

5. **显式双向 cross-attention 替代直接拼接**  
   模型不是简单把模态向量堆在一起，而是在 token 层让模态之间相互读取证据。

6. **核心交互集中在双阶段 cross-attention，而非额外堆叠联合注意力**
   模型重点保留 `Pressure-Flow` 内部交互和 `audio-sensor` 异构交互，让跨模态证据交换发生在最关键的位置。

7. **confidence-aware gate 将表示信息与分类置信信息共同用于融合决策**  
   这比只依赖表示内容的普通 softmax gate 更细粒度。

8. **expert residual 保留单模态专家的直接判别证据**  
   防止多模态融合层过度平滑掉强证据模态。

9. **modality dropout + mask-aware fusion 提升缺失模态鲁棒性**  
   使模型在训练时就学会处理部分模态不可靠或缺失的情形。

## 11. 当前结果对应的结构结论

在统一的固定非对齐 `5 s` 窗条件下，当前默认模型相关的压缩对照结果如下：

| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `HCAF compressed base SA0 PCEN96 HP80` | `0.8968 ± 0.0495` | `0.8815 ± 0.0838` |
| `HCAF-PCEN-DualXAttn` | `0.9155 ± 0.0133` | `0.9407 ± 0.0838` |
| `HCAF compressed SA0 summary token attention` | `0.8298 ± 0.0805` | `0.8815 ± 0.0838` |
| `HCAF compressed SA0 PCEN64 HP80` | `0.8773 ± 0.0458` | `0.8815 ± 0.0838` |

窗口级差值为：

- `HCAF-PCEN-DualXAttn - SA0 base = +0.0187`
- `HCAF-PCEN-DualXAttn - summary token = +0.0857`
- `HCAF-PCEN-DualXAttn - PCEN64 HP80 = +0.0382`

这些结果说明：

1. 当前主模型的优势并不来自额外堆叠更多模块；
   因为在保留核心结构的同时去掉联合 self-attention 后，结果并未下降，反而仍保持当前最佳。

2. 当前主模型也不是单纯依赖某个更小或更简单的前端；
   因为直接压缩到 `PCEN64` 后，window-level macro-F1 明显低于当前默认模型。

3. 真正起作用的是保留下来的核心机制组合；
   即 `PCEN96 + HP80`、双阶段 cross-attention 与 `confidence-aware gate + expert residual` 的协同，而不是名字里写出了多少被移除的部件。

因此，在当前任务设置下，可以将 `HCAF-PCEN-DualXAttn` 视为最具代表性的主结构，即“`ResNet18` 呼吸音骨干 + `TCN` 传感器编码器 + `PQ` 内部交互 + `audio-sensor cross-attention` + `confidence-aware gate + expert residual`”。

## 12. 可直接写入论文正文的总结性表述

> 在固定非对齐 `5 s` 窗条件下，本文采用 `ResNet18`（`ImageNet` 初始化）作为呼吸音编码骨干，采用 `1D CNN stem + TCN` 对压力与流量信号进行建模，并通过“先 `Pressure-Flow` 内部交互、再 `audio-sensor` 双向 cross-attention”的层次化融合策略完成多模态判别。在音频前端中，本文使用 `PCEN96 + HP80` 强化呼吸音表征，并在决策层引入 `confidence-aware gate` 与 `expert residual`，使融合权重同时依赖模态表示与分类置信度。补充压缩实验表明，保留上述核心结构的 `HCAF-PCEN-DualXAttn` 在窗口级 macro-F1 上达到 `0.9155 ± 0.0133`，说明当前任务中的主要性能收益来自关键前端、层次化跨模态交互与置信感知决策机制，而不是额外叠加的非核心模块。
