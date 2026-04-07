# RK3588 Runtime Notes

这个目录不是直接照抄 `breathe_v0.3`，而是只复用了两条已经在板端验证过的工程路径：

- `PyAudio` 实时读取 USB 音频
- 通过串口连续接收 `CH0/CH1`，再用标定参数换算成 `pressure/flow`

## 和旧项目的关系

旧项目里的 `demo_multimode.py` 并没有把实时 `pressure/flow` 序列直接送进主进程做多模态推理。
它实际做的是：

- 启动 `R_Identification/ADC_Realtime.py`
- 由子进程读串口
- 子进程自己计算 `R/C/MP`
- 主进程只轮询 CSV 里的最新 `R`

这对旧任务够用，但对当前模型不够，因为当前模型需要的是完整的 `pressure` 与 `flow` 时间序列。

## 当前文件用途

- `sensor_serial.py`
  - 直接沿用旧项目的串口协议：`CH0` 对应压力，`CH1` 对应流量
  - 直接读取 `params.json` 里的标定参数与串口配置
  - 输出实时 `pressure/flow` 样本
- `runtime_infer_onnx.py`
  - 板端预处理与 ONNX 推理封装
  - 优先使用训练同款 `torch/torchaudio` 预处理
  - 如果板端没有这些依赖，再退回 `librosa/scipy` 近似预处理
- `demo_multimodal.py`
  - 复用旧项目的实时音频采集方式
  - 同时维护音频环形缓冲区和传感器环形缓冲区
  - 每个 `hop` 直接做一次多模态 ONNX 推理
  - 可选保存 `5 s` 快照做调试

## 当前阶段的定位

这里已经接入了板端实时 `ONNX Runtime` 推理。
建议顺序仍然是：

1. PC 侧先完成 `PyTorch` / `ONNX` 离线一致性验证
2. 板端再跑这个实时 demo
3. 如果板端性能仍不够，再转 `RKNN`

## 依赖建议

为了尽量保持和训练时一致，板端优先建议安装：

- `onnxruntime`
- `torch`
- `torchaudio`
- `pyyaml`
- `pyaudio`
- `pyserial`

如果板端暂时装不上 `torch/torchaudio`，也可以退回：

- `onnxruntime`
- `librosa`
- `scipy`
- `pyyaml`
- `pyaudio`
- `pyserial`

但这条近似预处理路线更适合作为临时验证，不如同款预处理稳。

## 运行方式

```bash
python3 demo_multimodal.py \
  --onnx-model ../artifacts/hcaf_pcen_dualxattn.onnx \
  --config ../../configs/final_model_unified_evidence.yaml \
  --audio-rate 16000 \
  --window-sec 5 \
  --hop-sec 1 \
  --sensor-params R_Identification/params.json
```

默认行为会周期性输出：

- predicted label
- predicted index
- probabilities
- inference latency

如果加 `--save-snapshots`，还会额外保存：

- `*_audio.wav`
- `*_daq.csv`
- `*_meta.json`
