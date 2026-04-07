# Deploy Directory

这个目录集中放当前项目的部署脚本与板端运行骨架。

## 文件说明

- `edge_deploy_utils.py`
  - 部署侧公共函数
  - 负责加载配置、恢复权重、构造单窗三输入
- `export_onnx.py`
  - 将当前默认多模态模型导出为 ONNX
- `offline_multimodal_infer.py`
  - 对单个 `MMdata_*` 目录做离线推理
  - 可直接对比 `PyTorch` 与 `ONNX Runtime`
- `rk3588_runtime/`
  - 基于旧 `breathe_v0.3` 经验改写的 RK3588 板端采集骨架

## 建议用法

在 PC 上：

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate onnx311

python deploy/export_onnx.py --output deploy/artifacts/hcaf_pcen_dualxattn.onnx

python deploy/offline_multimodal_infer.py \
  data/MMdata_235.00s_0320_224031_no_secretion \
  --backend both \
  --onnx deploy/artifacts/hcaf_pcen_dualxattn.onnx
```

在 RK3588 上：

```bash
python3 deploy/rk3588_runtime/demo_multimodal.py
```

## 当前状态

- 离线 `PyTorch` 推理已验证通过
- ONNX 导出已验证通过
- `PyTorch` 与 `ONNX Runtime` 的离线结果已验证一致
- RK3588 实时部分目前先负责稳定采集音频与 `pressure/flow` 窗口
  - 现在已经接入实时 `ONNX Runtime` 推理
  - 优先建议板端使用训练同款 `torch/torchaudio` 预处理
