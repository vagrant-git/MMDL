# RK3588 Bundle

这个文件夹是给板子用的最小部署包。

你需要把仓库里的 `deploy/rk3588_bundle/` 目录整体拷到 RK3588 板子上，例如：

```bash
scp -r deploy/rk3588_bundle firefly@<board-ip>:~/
```

拷到板子后，建议目录形态是：

```text
~/rk3588_bundle/
├── README.md
├── requirements_board.txt
├── configs/
│   └── final_model_unified_evidence.yaml
├── models/
│   ├── hcaf_pcen_dualxattn.onnx
│   └── hcaf_pcen_dualxattn.json
└── runtime/
    ├── demo_multimodal.py
    ├── runtime_infer_onnx.py
    └── sensor_serial.py
```

## 1. 板端先准备什么

这个包默认使用：

- USB 音频实时采集
- 串口 `CH0/CH1` 实时读取 `pressure/flow`
- `ONNX Runtime` 做多模态推理

你还需要板子上已有或准备好：

- 一个可用的 Python 3.9 环境
- 原来老项目里的 `R_Identification/params.json`
  - 里面要有串口号和 `pressure/flow` 标定参数
- 音频输入设备已经能被 `PyAudio` 正常读到

## 2. 建议安装依赖

先进入板子：

```bash
cd ~/rk3588_bundle
python3.9 -m pip install -r requirements_board.txt
```

如果板子上已经有部分包，可以按需跳过。

## 3. 最推荐的运行方式

如果板子能装上 `torch` 和 `torchaudio`，优先走这条。
因为这样板端前处理最接近训练时逻辑。

运行命令：

```bash
cd ~/rk3588_bundle/runtime
python3.9 demo_multimodal.py \
  --onnx-model ../models/hcaf_pcen_dualxattn.onnx \
  --config ../configs/final_model_unified_evidence.yaml \
  --sensor-params /path/to/R_Identification/params.json \
  --audio-rate 16000 \
  --window-sec 5 \
  --hop-sec 1
```

如果你想保留每个窗口的调试快照，再加：

```bash
  --save-snapshots
```

## 4. 运行后会看到什么

终端会周期性输出：

- `predicted_label`
- `predicted_index`
- `probabilities`
- `latency_ms`

如果启用了 `--save-snapshots`，还会在 `../runtime_debug/` 里保存：

- `*_audio.wav`
- `*_daq.csv`
- `*_meta.json`

## 5. 如果板子装不上 torch/torchaudio

程序会自动退回到 `librosa/scipy` 的近似前处理路线。

这条路线可以先用于联调，但不如训练同款前处理稳。
如果你发现板端结果和 PC 侧差得比较多，优先检查是不是退回到了近似前处理。

## 6. 这个包已经在 PC 上验证过什么

已完成：

- ONNX 导出成功
- 单个 `MMdata_*` 离线样本的 `PyTorch / ONNX Runtime` 一致性验证成功
- 最大 logits 差异约 `9.5e-07`
- 预测类别一致

## 7. 板端首次联调建议顺序

1. 先确认 `params.json` 指向的串口能打开。
2. 先确认板子音频设备能被 `PyAudio` 读到。
3. 先不加 `--save-snapshots`，直接看实时输出是否持续刷新。
4. 如果推理能跑，再加 `--save-snapshots` 检查窗口内容。
5. 如果结果明显异常，再回头比对 PC 离线推理结果。
