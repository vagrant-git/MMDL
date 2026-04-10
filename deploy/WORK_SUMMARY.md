# Deployment Work Summary

## 1. 本轮工作的目标

本轮工作的目标不是直接在 RK3588 板子上完成全部联调，而是先把当前多模态模型的部署链路整理清楚，并完成下面三件关键事情：

1. 将部署相关代码从实验仓库中独立出来。
2. 在 PC 端完成 `PyTorch -> ONNX` 的离线部署闭环。
3. 基于旧 `breathe_v0.3` 项目的工程经验，搭建适用于当前多模态模型的 RK3588 运行骨架。

## 2. 已完成的主要工作

### 2.1 整理部署目录

已将部署相关内容集中到 [deploy](/home/oi/MMDL/deploy) 目录下，避免继续散落在仓库根目录。

当前主要文件包括：

- [README.md](/home/oi/MMDL/deploy/README.md)
- [edge_deploy_utils.py](/home/oi/MMDL/deploy/edge_deploy_utils.py)
- [export_onnx.py](/home/oi/MMDL/deploy/export_onnx.py)
- [offline_multimodal_infer.py](/home/oi/MMDL/deploy/offline_multimodal_infer.py)
- [README.md](/home/oi/MMDL/deploy/rk3588_runtime/README.md)
- [demo_multimodal.py](/home/oi/MMDL/deploy/rk3588_runtime/demo_multimodal.py)
- [runtime_infer_onnx.py](/home/oi/MMDL/deploy/rk3588_runtime/runtime_infer_onnx.py)
- [sensor_serial.py](/home/oi/MMDL/deploy/rk3588_runtime/sensor_serial.py)

### 2.2 完成 ONNX 导出脚本

已实现 [export_onnx.py](/home/oi/MMDL/deploy/export_onnx.py)，可将当前默认模型导出为 ONNX。

默认模型来源：

- 配置文件: [final_model_unified_evidence.yaml](/home/oi/MMDL/configs/final_model_unified_evidence.yaml)
- 默认权重: `summary-MMmodel/final_model_unified_evidence/runs/hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s/repeat1_fold1/best_model.pt`

实际导出产物：

- [hcaf_pcen_dualxattn.onnx](/home/oi/MMDL/deploy/artifacts/hcaf_pcen_dualxattn.onnx)
- [hcaf_pcen_dualxattn.json](/home/oi/MMDL/deploy/artifacts/hcaf_pcen_dualxattn.json)

### 2.3 完成离线多模态推理脚本

已实现 [offline_multimodal_infer.py](/home/oi/MMDL/deploy/offline_multimodal_infer.py)，用于对单个 `MMdata_*` 目录做离线单窗推理。

它支持：

- `PyTorch` 推理
- `ONNX Runtime` 推理
- 两者数值对比

### 2.4 完成离线一致性验证

已在 PC 端使用样本：

- `data/MMdata_235.00s_0320_224031_no_secretion`

完成 `PyTorch / ONNX Runtime` 一致性验证，结果如下：

- `predicted_index` 一致
- `same_predicted_index = true`
- `max_abs_diff = 9.5367431640625e-07`
- `mean_abs_diff = 3.97364289028701e-07`

这说明：

- 当前 ONNX 导出结果可用
- 当前三输入组织方式可用
- 当前离线部署链闭环已成立

## 3. RK3588 侧已完成的工作

### 3.1 复用旧项目的工程经验

已参考旧项目：

- `C:\Users\WangShuai\Desktop\code in RK3588s\breathe_v0.3`

但没有直接照抄其旧音频模型逻辑，而是只复用了两条已经在板端被验证过的路径：

- `PyAudio` 实时采集 USB 音频
- 串口 `CH0/CH1` 连续读取传感器数据

### 3.2 确认旧 `demo_multimode.py` 的局限

已确认旧项目中的 `demo_multimode.py` 并不是把完整 `pressure/flow` 序列直接送入模型。

旧逻辑实际是：

1. 启动 `ADC_Realtime.py`
2. 子进程读串口
3. 子进程计算 `R/C/MP`
4. 主进程轮询 CSV 中的最新 `R`

这对旧任务足够，但不满足当前模型需求，因为当前模型需要整段 `pressure` 与 `flow` 序列。

### 3.3 实现 RK3588 多模态实时运行骨架

已实现：

- [sensor_serial.py](/home/oi/MMDL/deploy/rk3588_runtime/sensor_serial.py)
  - 直接沿用旧串口协议
  - 直接解析 `CH0/CH1`
  - 直接结合 `params.json` 做 `pressure/flow` 标定
- [runtime_infer_onnx.py](/home/oi/MMDL/deploy/rk3588_runtime/runtime_infer_onnx.py)
  - 板端 ONNX 推理封装
  - 负责音频预处理、传感器标准化、ONNX 输入组织和推理
- [demo_multimodal.py](/home/oi/MMDL/deploy/rk3588_runtime/demo_multimodal.py)
  - 负责实时音频采集
  - 负责维护音频与传感器环形缓冲区
  - 负责每个 `hop` 直接触发一次 ONNX 推理

## 4. 关于板端预处理的处理策略

这是本轮工作中特别关键的一点。

当前板端预处理采用“双路径策略”：

1. 优先使用训练同款的 `torch/torchaudio` 预处理。
2. 如果板端无法安装 `torch/torchaudio`，再退回 `librosa/scipy` 的近似预处理。

这样做的原因是：

- 单靠 `librosa/scipy` 近似前处理，数值可能漂移
- 当前模型对音频前处理一致性较敏感
- 因此板端若能使用训练同款预处理，更稳妥

已在 PC 上验证：

- 使用同款预处理时，实时 ONNX 推理的预测类别回到正确类别

## 5. 已准备好的板端可拷贝目录

已生成板端部署包：

- [rk3588_bundle](/home/oi/MMDL/deploy/rk3588_bundle)

主要内容包括：

- [README.md](/home/oi/MMDL/deploy/rk3588_bundle/README.md)
- [requirements_board.txt](/home/oi/MMDL/deploy/rk3588_bundle/requirements_board.txt)
- [RUN_ON_BOARD.sh](/home/oi/MMDL/deploy/rk3588_bundle/RUN_ON_BOARD.sh)
- [hcaf_pcen_dualxattn.onnx](/home/oi/MMDL/deploy/rk3588_bundle/models/hcaf_pcen_dualxattn.onnx)
- [final_model_unified_evidence.yaml](/home/oi/MMDL/deploy/rk3588_bundle/configs/final_model_unified_evidence.yaml)
- [demo_multimodal.py](/home/oi/MMDL/deploy/rk3588_bundle/runtime/demo_multimodal.py)

这个目录的目标是：

- 可以整体拷到 RK3588 板子上
- 不要求保留完整训练仓库
- 带有板端依赖清单与运行说明

## 6. 本轮工作的实际结论

本轮工作已经完成了当前多模态模型部署中的“PC 端部署闭环 + 板端运行骨架”两部分。

可以明确认为已经完成的结论有：

- 当前默认模型已经可以成功导出为 ONNX。
- 当前多模态三输入 `audio + pressure + flow` 的离线部署链是可行的。
- 当前 RK3588 侧已经具备实时音频采集与实时传感器读取的运行骨架。
- 当前工程上已经具备继续开展板端联调的条件。

## 7. 后续仍需完成的工作

后续工作重点已经从“继续写部署代码”转为“板端真实联调”。

建议按以下顺序继续推进：

1. 将 [rk3588_bundle](/home/oi/MMDL/deploy/rk3588_bundle) 整体拷到 RK3588 板子。
2. 在板子上安装 `requirements_board.txt` 中的依赖。
3. 准备好原有项目中的 `R_Identification/params.json`。
4. 在板子上运行 `runtime/demo_multimodal.py`。
5. 观察实时输出的：
   - `predicted_label`
   - `probabilities`
   - `latency_ms`
6. 如果结果异常，再开启 `--save-snapshots` 做窗口级排查。
7. 如果板端性能不足，再考虑继续转向 `RKNN`。

## 8. 当前阶段的判断

截至本轮工作结束，可以认为：

- 当前课题的边缘部署已经不再停留在“概念设计”
- 已经进入“可执行、可验证、可板端联调”的阶段

但也应保持边界清晰：

- 当前已经验证的是部署工程可行性
- 还没有完成真实硬件场景下的长期稳定性联调
- 也还没有完成最终的 RKNN/NPU 路线优化

因此当前最合理的表述是：

> 已完成多模态模型在 ARM/RK3588 部署链路中的 ONNX 导出、一致性验证与板端运行骨架搭建，初步证明该系统具备边缘端实时部署的工程可行性；后续工作重点为板端真实采集链路联调与运行稳定性验证。
