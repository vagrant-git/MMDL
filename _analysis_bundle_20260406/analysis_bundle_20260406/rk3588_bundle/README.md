# RK3588 Bundle

这个目录是给 RK3588 板子使用的最小部署包。模型和配置已经包含在包内，板端主要工作是准备运行环境、串口参数和音频设备。

先把整个目录拷到板子：

```bash
scp -r rk3588_bundle firefly@<board-ip>:~/
```

## 1. 板端需要准备什么

默认运行方式：

- USB 音频实时采集
- 串口 `CH0/CH1` 实时读取 `pressure/flow`
- `ONNX Runtime` 做多模态推理

你还需要：

- Python 3.9
- 老项目里的 `R_Identification/params.json`
- 可被 `PyAudio` 读到的音频输入设备

`params.json` 里至少要有：

- `serial_port`
- `sample_rate`
- `pressure_slope` / `pressure_intercept`
- `flow_slope` / `flow_intercept`

## 2. 安装依赖

```bash
cd ~/rk3588_bundle
python3.9 -m pip install -r requirements_board.txt
```

如果板子能装 `torch` 和 `torchaudio`，启动时会优先使用更接近训练侧的前处理；否则会退回 `librosa/scipy`。

## 3. 先做部署自检

先列出板子的音频输入设备：

```bash
cd ~/rk3588_bundle
./RUN_ON_BOARD.sh --list-audio-devices
```

如果你想先确认 ADC 串口链路本身是否通，再跑下面这条：

```bash
python3.9 runtime/serial_probe.py \
  --sensor-params /path/to/R_Identification/params.json
```

这条命令只看串口 `CH0/CH1` 数据和换算后的 `pressure/flow`，不启动模型。

再做完整自检：

```bash
./RUN_ON_BOARD.sh \
  --sensor-params /path/to/R_Identification/params.json \
  --audio-device-index 1
```

如果不指定 `--audio-device-index`，默认使用系统默认输入设备。

## 4. 启动实时推理

```bash
cd ~/rk3588_bundle
./RUN_ON_BOARD.sh \
  --sensor-params /path/to/R_Identification/params.json \
  --audio-device-index 1 \
  --hop-sec 1
```

也可以用环境变量传入参数文件：

```bash
SENSOR_PARAMS=/path/to/R_Identification/params.json ./RUN_ON_BOARD.sh
```

加上 `--save-snapshots` 后，会在 `runtime_debug/` 里保存 `*_audio.wav`、`*_daq.csv` 和 `*_meta.json`。

## 5. 启动后看什么

启动时会先输出 `audio_preprocess_backend=torch` 或 `audio_preprocess_backend=librosa`，方便判断当前板端走的是哪条前处理路径。
同时会输出：

- `audio_preprocess_backend_mode=training_consistent_torch` 或 `librosa_fallback`
- `input_normalization_mode=rolling` 或 `window`
- 当使用滚动统计时，还会输出 `input_normalization_buffer_sec=...`

当前默认使用 `rolling` 长缓冲区统计归一化，让实时板端输入更接近离线 PC 侧分布。
如果你想临时切回原来的逐窗口归一化，可以这样启动：

```bash
cd ~/rk3588_bundle
./RUN_ON_BOARD.sh \
  --sensor-params /path/to/R_Identification/params.json \
  --audio-device-index 1 \
  --normalization-mode window
```

正常运行后，终端会周期性输出：

- `pred`
- `idx`
- `probs`
- `latency_ms`

## 6. 当前已验证内容

- ONNX 模型已导出完成：`models/hcaf_pcen_dualxattn.onnx`
- 板端运行配置已打包：`configs/final_model_unified_evidence.yaml`
- PC 侧已验证单样本 `PyTorch / ONNX Runtime` 一致性
- 最大 logits 差异约 `9.5e-07`
