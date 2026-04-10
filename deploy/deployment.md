# Deployment Guide

本文档基于两个外部来源整理：

- `C:\Users\Wangshuai\Desktop\SYNC\RespireNet_train`
- `C:\Users\Wangshuai\Desktop\MMDataset\MMDAQ\MultiModalSync.ipynb`

目标不是复述训练流程，而是提炼其中对部署真正有用的内容，并把它改写成适合当前多模态模型在 ARM 主板上落地的实施说明。

## 1. 两个来源分别提供了什么

### 1.1 RespireNet_train 提供的部署思路

这个仓库里最有部署价值的部分不是训练代码，而是下面几项：

- `resave.py`
  - 先把训练好的 PyTorch checkpoint 加载到网络中
  - 再用 `torch.jit.trace(...)` 导出成 `TorchScript .pt`
- `pt2RKNN.py`
  - 用 `rknn.load_pytorch(...)` 读取 `.pt`
  - 指定输入尺寸
  - 指定 `target_platform="rk3588"`
  - 构建并导出 `.rknn`
- `infer_check.py`
  - 在 PC 端验证 `TorchScript` 推理结果
  - 在 RKNN 侧验证 `.rknn` 推理结果
  - 用同一张图像比较输出是否合理
- `requirements_推理.txt`
  - 暗示板端推理环境包含 `rknn_toolkit_lite2`
  - 同时包含 `PyAudio`，说明原方案考虑过实时音频输入

这套方案本质上是一条很清晰的部署链：

`训练得到 checkpoint -> 重建模型 -> 导出 TorchScript .pt -> 转换成 RKNN -> 在板端做推理验证`

这条链路对你当前项目仍然有参考价值，只是不能原样照搬。原因是 RespireNet 是单模态音频图像分类，而你现在是音频 + Pressure + Flow 的多模态实时系统。

### 1.2 MultiModalSync.ipynb 提供的采集思路

这个 notebook 对部署最重要，因为它定义了“数据是怎么来的”。

它当前的采集架构是：

- 音频
  - 用 `sounddevice.InputStream` 采集
  - 默认 `audio_rate=22050`
  - `audio_channels=1`
  - `audio_dtype="int16"`
  - `audio_blocksize=1024`
- 传感器
  - 用 `nidaqmx` 从 NI DAQ 读两路模拟量
  - `pressure_channel="ai0"`
  - `flow_channel="ai1"`
  - 默认 `sample_rate=100 Hz`
- 同步方式
  - 不是硬件同步
  - 是在同一软件流程里同时启动音频流和 DAQ 任务
  - 用 `perf_counter` 推导统一时间轴
- 数据落盘
  - 音频保存为 `audio.wav`
  - 压力/流量保存为 `daq.csv`
  - 元数据保存为 `metadata.json`
- 标定支持
  - `params.json` 中保存
    - `pressure_slope`
    - `pressure_intercept`
    - `flow_slope`
    - `flow_intercept`
    - `sample_rate`

notebook 里的一个非常关键的说明是：

> `audio 和 daq 通过同一软件流程启动，属于软件同步，不是硬件时钟级同步`

这句话对部署判断很重要。它意味着你当前数据采集系统能满足实验和标注，但如果以后要求严格时序一致性，就要考虑主板侧统一时钟、统一采样触发或者后处理重采样对齐。

## 2. 当前多模态模型部署时，哪些内容可以直接借鉴

可以直接借鉴的主要有 4 点。

### 2.1 借鉴导出链路

RespireNet 的思路完全可以保留为：

1. 加载训练好的权重
2. 构造推理态模型
3. 导出中间格式
4. 在目标硬件专用工具链里转换
5. 在板端做一致性检查

只是你当前模型不一定最终导出到 RKNN，取决于主板 SoC。

如果主板是 Rockchip RK3588 / RK356x 这类 NPU 平台，可以继续走：

`PyTorch -> TorchScript/ONNX -> RKNN -> rknn_toolkit_lite2`

如果主板只是普通 ARM CPU，没有 Rockchip NPU，就更适合：

`PyTorch -> ONNX -> ONNX Runtime / TFLite / NCNN`

### 2.2 借鉴“先离线验证，再上板”

`infer_check.py` 的思想是对的：

- 先用同一输入跑原始 PyTorch 模型
- 再跑部署格式模型
- 比较 logits / softmax / top-k 结果是否一致

对你当前项目，这一步必须升级为多输入一致性验证：

- 输入 1: 音频特征
- 输入 2: pressure 序列
- 输入 3: flow 序列

不能只验证“能跑”，还要验证：

- 形状一致
- 预处理一致
- 标签映射一致
- 数值误差在可接受范围内

### 2.3 借鉴推理环境最小化

RespireNet 的板端依赖相对克制。这个思路非常对。

部署版不应把训练依赖全部搬到板子上。主板侧通常只保留：

- 推理引擎
- 音频采集库
- ADC / 驱动接口
- 少量数值处理依赖
- 必要的日志与故障恢复模块

### 2.4 借鉴“单独写一个部署脚本”

不要把部署逻辑散在 notebook 里。应单独拆成：

- `export_model.py`
- `benchmark_infer.py`
- `runtime_capture.py`
- `runtime_infer.py`

RespireNet 仓库里虽然实现比较简单，但这个方向是对的。

## 3. 哪些内容不能直接照搬

### 3.1 不能直接照搬 NI DAQ

`MultiModalSync.ipynb` 依赖：

- `nidaqmx`
- NI DAQ 设备
- Windows/桌面侧驱动环境

而你当前目标是 ARM 主板。大多数 ARM 主板并不会原生接 NI 采集卡，因此这一段必须替换。

你自己的判断是对的：板端更现实的是 ADC 方案。

### 3.2 不能直接照搬单输入图像模型部署

RespireNet 的输入是固定尺寸图像，`pt2RKNN.py` 里直接写了：

- `input_size_list = [[1, 3, 192, 753]]`

你的当前模型不是这个接口。你的默认 HCAF 模型是三输入：

- 音频特征张量
- pressure 序列
- flow 序列

所以你后续导出时要处理的是多输入模型，而不是单图像分类模型。

### 3.3 不能直接照搬采样率和窗长

采集 notebook 里的音频率是 `22050 Hz`，DAQ 是 `100 Hz`。  
而你当前主模型训练配置里是：

- 音频采样率 `16000 Hz`
- 传感器采样率 `100 Hz`
- 窗长 `5 s`

因此部署时要保证“板端实时预处理”与“训练时预处理”一致。否则模型即使成功运行，结果也会漂。

## 4. 面向当前项目的部署总体架构

推荐把系统拆成 5 层。

### 4.1 采集层

负责实时拿到三路原始数据：

- 呼吸音
- 压力
- 流量

在 ARM 主板上更合理的做法是：

- 音频
  - USB 声卡 / I2S 麦克风 / 板载音频 codec
- 压力与流量
  - 外接 ADC 芯片
  - 或 MCU 先采集，再通过串口 / USB / SPI / CAN 发给主板

### 4.2 标定层

负责把原始 ADC 计数或电压值转成物理量：

- Pressure: `pressure = slope * raw + intercept`
- Flow: `flow = slope * raw + intercept`

这部分可以直接借鉴 `MultiModalSync.ipynb` 的 `params.json` 思路。

部署时建议保留一个独立配置文件，例如：

```json
{
  "pressure_slope": 12.34,
  "pressure_intercept": -1.23,
  "flow_slope": 8.76,
  "flow_intercept": 0.45,
  "sensor_sample_rate": 100,
  "audio_sample_rate": 16000
}
```

### 4.3 对齐与缓冲层

负责形成送给模型的固定长度窗口。

建议使用环形缓冲区：

- 音频缓冲区
  - 长度至少 `5 s`
- pressure 缓冲区
  - 长度至少 `5 s`
- flow 缓冲区
  - 长度至少 `5 s`

然后按固定 hop 输出窗口，例如：

- 上下文窗口 `5 s`
- 更新步长 `1 s`

这样做的好处是：

- 不必每 `5 s` 才出一次结果
- 保持和当前模型训练窗长一致
- 兼顾稳定性和响应速度

### 4.4 预处理层

负责把原始信号变成模型输入。

对当前项目，这一层通常包含：

- 音频
  - 重采样到训练采样率
  - 高通或其他滤波
  - log-Mel 或 PCEN 特征
- pressure / flow
  - 截取 `5 s`
  - 必要时归一化
  - 整形成训练时同样的长度与张量形状

这部分是部署成败的关键。  
模型本体通常不是最容易出错的，最容易出错的是“板端前处理”和“训练前处理”不一致。

### 4.5 推理与决策层

负责：

- 调用部署模型
- 输出类别概率
- 做阈值、平滑、报警逻辑

建议不要直接使用单窗结果做最终报警，而是加一层时间平滑，例如：

- 最近 `3~5` 个窗多数投票
- 概率滑动平均
- 状态机去抖

## 5. ARM 主板上的采集替代方案

由于 NI DAQ 不适合直接搬到 ARM 主板，推荐优先考虑下面两种方案。

### 5.1 方案 A: 主板直接接 ADC

适合主板有足够驱动支持、采样率要求不高的情况。

建议结构：

- 麦克风 -> 音频接口
- 压力传感器 -> 模拟调理 -> ADC -> ARM
- 流量传感器 -> 模拟调理 -> ADC -> ARM

要求：

- ADC 至少两路同步或准同步采样
- 传感器通道有效采样率满足 `100 Hz`
- 驱动能够稳定持续读数

优点：

- 系统简单
- 少一个中间控制器

缺点：

- Linux 驱动和时序调试可能比较麻烦

### 5.2 方案 B: MCU + ADC 前端，ARM 只负责推理

这是我更推荐的工程方案。

建议结构：

- MCU 侧
  - 负责 ADC 采集
  - 负责基础滤波
  - 负责时间戳
  - 通过串口/USB/SPI 发给 ARM
- ARM 侧
  - 负责音频采集
  - 负责多模态缓冲和对齐
  - 负责模型推理和显示/报警

优点：

- 采样更稳定
- ADC 时序更容易控制
- ARM 负担更轻

缺点：

- 需要设计板间通信协议

如果你的最终目标是一个能稳定跑很久的设备，通常 `MCU 采集 + ARM 推理` 比“所有事情都压在主板 Linux 用户态里”更稳。

## 6. 当前项目建议的部署路线

建议分三阶段推进。

### 6.1 第一阶段：先做 PC 端部署闭环

先不要急着上板，先在 PC 上把部署链路闭环跑通。

步骤：

1. 固化当前最终模型
   - 明确使用哪个配置：`configs/final_model_unified_evidence.yaml`
   - 明确实验 ID：`hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s`
   - 明确展示名：`HCAF-PCEN-DualXAttn`
   - 明确权重文件
2. 写一个部署版前处理脚本
   - 输入 `audio.wav + pressure.csv + flow.csv`
   - 输出模型输入张量
3. 写一个导出脚本
   - 优先导出 `ONNX`
   - 如目标确定是 RK3588，再补 `RKNN`
4. 写一个一致性验证脚本
   - 同一组样本分别跑 PyTorch 与部署模型
   - 比较输出差异

完成这一阶段后，你才能确认真正要部署的不是“训练代码”，而是“推理系统”。

### 6.2 第二阶段：做 ARM 主板离线推理

这一步不接实时采集，只把事先录好的文件送到主板。

流程：

- 主板读取一段音频文件
- 主板读取一段传感器 csv
- 主板做前处理
- 主板跑模型
- 输出分类结果与延时

要测 4 个指标：

- 前处理耗时
- 模型推理耗时
- 端到端耗时
- 持续运行内存占用

### 6.3 第三阶段：接入实时采集

最后才接音频和 ADC 实时流。

原因很现实：

- 实时采集问题和模型问题会互相干扰
- 如果一开始就一起调，定位会非常慢

正确做法是：

1. 先确认模型部署没问题
2. 再确认采集链路没问题
3. 最后再做在线联调

## 7. 对当前多模态模型的具体部署建议

### 7.1 先不要直接做蒸馏

现阶段先做原模型部署验证更合理。

原因：

- 当前默认模型已经是压缩过一版的结构
- 真实瓶颈大概率不在 attention，而在前处理和系统集成
- 没有板端 profiling 之前就做蒸馏，容易过早优化

建议顺序：

1. 原模型部署
2. 量化
3. 如果仍然不够，再做蒸馏

### 7.2 优先导出 ONNX

除非你已经确认主板 SoC 是 Rockchip 并且要用 RKNN，否则优先 ONNX。

原因：

- ONNX 更通用
- 更适合多输入模型
- 更方便先在 PC 上做数值对比

推荐路线：

- PyTorch -> ONNX
- PC 上用 ONNX Runtime 验证
- 再根据主板 SoC 决定转 RKNN / TFLite / NCNN

### 7.3 板端前处理要尽量“去 notebook 化”

`MultiModalSync.ipynb` 非常适合实验，不适合长期部署。

板端代码应该重写为常规脚本或服务，至少拆成：

- `capture_audio.py`
- `capture_sensor.py`
- `preprocess.py`
- `infer.py`
- `main.py`

不要把实时显示、数据采集、模型推理全写在一个 notebook 里。

## 8. 从 MultiModalSync 中应保留的设计

下面这些设计值得保留。

### 8.1 保留 metadata.json

每次采集都应记录：

- 开始时间
- 请求采样率
- 实际采样率
- 样本数
- 时长
- 设备名
- 通道号
- 标定参数版本

这对排查板端问题非常关键。

### 8.2 保留一致性检查

notebook 里已经做了：

- `audio_duration_sec`
- `daq_duration_sec`
- `audio_to_daq_duration_ratio`

这个思路很好，板端应继续保留。

如果 `audio_to_daq_duration_ratio` 明显偏离 `1.0`，说明：

- 音频设备实际采样率漂移
- ADC 读数堵塞
- 缓冲区丢帧
- 对齐逻辑有 bug

### 8.3 保留参数文件

标定参数、采样率、设备 ID 不要写死在代码里，应该外置到配置文件。

### 8.4 实时图仅作为调试工具

notebook 里的三通道实时图适合调试，不适合作为正式部署主逻辑的一部分。

正式设备可以把它改成：

- 可选调试页面
- 简单状态灯
- 日志统计

## 9. 建议的部署目录结构

可以参考下面的结构重构部署代码：

```text
deployment/
├── configs/
│   ├── hardware.json
│   ├── calibration.json
│   └── model.json
├── export/
│   ├── export_onnx.py
│   ├── export_rknn.py
│   └── check_parity.py
├── runtime/
│   ├── capture_audio.py
│   ├── capture_sensor.py
│   ├── ring_buffer.py
│   ├── preprocess.py
│   ├── infer_engine.py
│   └── main.py
└── tools/
    ├── benchmark.py
    └── replay_from_files.py
```

## 10. 建议的实施顺序

下面这个顺序最稳。

1. 固定当前最终模型及权重
2. 写 PC 端“离线前处理 + 推理”脚本
3. 导出 ONNX
4. 做 PyTorch / ONNX 一致性验证
5. 在 ARM 主板上跑离线样本 benchmark
6. 确认主板采集方案
   - 直接 ADC
   - 或 MCU + ADC
7. 接入实时音频
8. 接入实时 pressure / flow
9. 做多模态对齐
10. 加入平滑决策、日志和异常恢复

## 11. 结论

从 `RespireNet_train` 学到的核心不是某一段具体代码，而是一条部署方法论：

- 训练模型与部署模型分离
- 先导出中间格式
- 再转目标硬件格式
- 用独立脚本验证推理一致性

从 `MultiModalSync.ipynb` 学到的核心不是 notebook 本身，而是采集系统设计：

- 音频和传感器分别采集
- 通过统一软件流程做时间对齐
- 记录完整 metadata
- 保留标定参数和一致性检查

对你当前项目，最合理的落地方案是：

- 在 PC 上先完成多输入模型的部署闭环
- 在 ARM 主板上优先采用 `音频接口 + ADC` 或 `音频接口 + MCU/ADC`
- 不再依赖 NI DAQ
- 先做离线推理，再做实时接入
- 先验证原模型可部署，再决定是否量化或蒸馏

如果后续确认你的主板就是 Rockchip RK3588 一类平台，那么可以继续沿用 RespireNet 里的 RKNN 思路；如果不是，就应优先转向 ONNX Runtime 或其他更通用的 ARM 推理后端。

## 12. 硕士论文中边缘部署部分建议保留的成果

这一部分建议写得简洁，不要把重点放在“已经做成完整产品”，而是放在“已经验证具备边缘部署可行性”。

建议至少保留下面 4 类结果。

### 12.1 部署前后精度对比

应当把固定测试集或固定测试子集复制到边缘主板上，分别在：

- PC 端原始模型
- 边缘主板部署模型

上各跑一次，然后比较：

- Accuracy
- Macro-F1
- Precision / Recall
- 混淆矩阵

这一组结果用来说明部署后模型是否仍然保持原有判别能力。

### 12.2 实时性指标

建议至少测量：

- 单窗前处理时间
- 单窗模型推理时间
- 单窗端到端总时间

如果系统采用滑窗机制，还要写清楚：

- 窗长
- hop
- 是否满足实时运行要求

### 12.3 资源占用指标

建议至少给出：

- 参数量
- 模型文件大小
- 运行时内存占用
- CPU 占用率

如果硬件支持 NPU，也可以补充 NPU 使用情况；如果无法测功耗，可以在论文中明确说明未做功耗测试。

### 12.4 稳定性与一致性

建议至少补一个简短稳定性实验，例如：

- 连续运行一段时间，确认系统无崩溃、无明显丢帧
- 检查 audio 与 pressure/flow 时长比值是否接近 1

这组结果用于说明该系统不仅能跑一次，而且能稳定运行。

## 13. 最小可答辩成果包

如果时间有限，边缘部署部分至少应完成下面这些内容：

1. 将固定测试集复制到主板并完成整套推理。
2. 给出主板端与 PC 端的精度对比。
3. 给出主板端单窗延时。
4. 给出主板端内存或 CPU 占用。
5. 说明传感器输入链路将由 NI DAQ 改为 ADC 或 MCU+ADC 方案。

对于你当前课题，更稳妥的论文表述是：

- 当前模型在模拟采集数据上完成训练与验证。
- 边缘部署实验主要验证模型推理的工程可行性。
- 当前结果表明该系统具备边缘实时部署潜力，但尚未完成真实临床在线场景的大规模验证。
