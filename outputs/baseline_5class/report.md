# 多模态 5 分类基线实验报告

## 1. 数据组织方式

数据根目录为 `data`，每个 session 目录默认包含：

- `audio.wav`
- `daq.csv`
- `metadata.json`

每个 recording/session 作为一个独立样本单元，严格先按 session 划分 train/val/test，再切 10 秒窗口。

## 2. 标签解析方式

标签优先从 `metadata.json` 中的 `label` 字段解析，失败时回退到文件夹名。统一映射到：

- 0: no / no secretion / 0 ml
- 1: 1 ml
- 2: 2 ml
- 3: 3 ml
- 4: 4 ml

## 3. 数据划分策略

固定随机种子：`20260327`。

划分规则：

- 每个类别优先保留 1 个 session 到 test
- 剩余 session 中约 25% 进入 val
- 其余进入 train

当前 session 级划分如下：

- train: 11 sessions, labels=[4, 1, 2, 0, 1, 0, 4, 0, 3, 0, 2], ids=['MMdata_1012.15s_0321_003441_yumi_4ml', 'MMdata_1100.00s_0327_170159_1ml', 'MMdata_1200.00s_0322_232833_2ml_yumi', 'MMdata_195.00s_0322_230906_no_secretion', 'MMdata_1964.50s_0327_175903_1ml', 'MMdata_318.75s_0327_175326_no', 'MMdata_474.50s_0323_001822_4ml_yumi', 'MMdata_598.25s_0322_224923_no_secretion', 'MMdata_600.00s_0327_190504_3ml', 'MMdata_660.75s_0327_164500_no_secretion', 'MMdata_820.00s_0327_172321_2ml']
- val: 3 sessions, labels=[0, 0, 2], ids=['MMdata_235.00s_0320_224031_no_secretion', 'MMdata_265.10s_0322_224132_no_secretion', 'MMdata_272.75s_0327_174501_2ml']
- test: 5 sessions, labels=[3, 2, 0, 1, 4], ids=['MMdata_1000.00s_0327_191853_3ml', 'MMdata_1136.50s_0327_183428_2ml', 'MMdata_1973.40s_0320_224508_no_secretion', 'MMdata_554.50s_0322_231552_1ml_yumi', 'MMdata_949.50s_0323_000227_4ml_yumi']

## 4. 预处理方法

- audio: 重采样到 `16000` Hz，单声道，逐 recording 做 z-score，转 log-Mel 频谱
- pressure / flow: 使用 `daq.csv` 原始序列，按 recording 做 z-score，采样率按 `100` Hz 处理
- 时间对齐: 对三模态按共同有效时长截断到最短模态
- 窗口: 固定 `10.0` 秒，步长 `10.0` 秒，不重叠
- 对不足一个窗口的 recording，默认跳过，不做补齐

## 5. 模型结构说明

- `audio_only`: 轻量 2D CNN 编码 log-Mel 频谱，GAP 后线性分类
- `pressure_flow`: pressure 和 flow 各自通过 1D CNN encoder，concat 后接 MLP
- `multimodal`: audio 2D CNN + pressure 1D CNN + flow 1D CNN，三分支中间层融合后分类

## 6. 训练配置

- epoch 上限: `12`
- batch size: `16`
- optimizer: Adam
- learning rate: `0.001`
- weight decay: `0.0001`
- weighted sampler: `True`
- early stopping patience: `4`

## 7. 测试集结果表

| Model | Window Acc | Window F1 | Session Acc | Session F1 | Best Epoch |
| --- | --- | --- | --- | --- | --- |
| audio_only | 0.4580 | 0.2857 | 0.2000 | 0.0800 | 1 |
| pressure_flow | 0.5528 | 0.3610 | 0.4000 | 0.2667 | 6 |
| multimodal | 0.8640 | 0.7550 | 0.8000 | 0.7333 | 3 |

## 8. 混淆矩阵图

### 窗口级

![audio_only](audio_only/confusion_matrix.png)

![pressure_flow](pressure_flow/confusion_matrix.png)

![multimodal](multimodal/confusion_matrix.png)

### Session 聚合

![audio_only_session](audio_only/confusion_matrix_session.png)

![pressure_flow_session](pressure_flow/confusion_matrix_session.png)

![multimodal_session](multimodal/confusion_matrix_session.png)

## 9. 结果分析与结论

- 五分类任务可以完整跑通，当前最佳模型是 `multimodal`。
- 类别 3 只有 2 个 session，因此无法同时覆盖 train/val/test，验证集不保证所有类别都出现。
- 由于按 session 先划分再切窗，避免了同一 recording 的窗口泄漏到不同数据集。
- 最佳模型窗口级指标: accuracy=0.8640, macro-F1=0.7550；session 聚合指标: accuracy=0.8000, macro-F1=0.7333。
- 相邻组别混淆情况（最佳模型，窗口级）: 1 vs 2: 21 windows, 2 vs 3: 0 windows, 3 vs 4: 0 windows。
- `0 ml` 与 `4 ml` 在最佳模型上没有出现互相混淆（0<->4 共 0 个窗口），说明负荷两端更容易区分。
- 主要混淆集中在: 1 -> 0: 55 windows, 2 -> 1: 21 windows。
- 从当前测试集看，`1 ml` 是最难的类别，最佳模型把该类 session 预测成了 `0 ml`；`2 ml` 有一部分窗口被预测成 `1 ml`，而 `3 ml` 与 `4 ml` 基本稳定。

- 最佳模型窗口级 confusion matrix: `[[197, 0, 0, 0, 0], [55, 0, 0, 0, 0], [0, 21, 92, 0, 0], [0, 0, 0, 100, 0], [0, 0, 0, 0, 94]]`。
- 最佳模型 session 聚合 confusion matrix: `[[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]`。
