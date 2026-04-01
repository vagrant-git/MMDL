# MMDL 5-Class Baseline

运行全部实验：

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
python run_all.py --config configs/baseline.yaml
```

单独运行某个模型：

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
python train.py --config configs/baseline.yaml --modality audio_only
```

输出目录默认为 `outputs/baseline_5class/`。

全仓库通用的数据过滤规则写在配置文件的 `session_filter` 段，例如：

```yaml
session_filter:
  exclude_session_ids:
    - "MMdata_265.10s_0322_224132_no_secretion"
```

凡是走当前数据发现与训练流程的脚本，都会先应用这条规则，再做任务映射和 train/val/test 或 grouped CV 划分。

运行 grouped cross-validation：

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
python grouped_cv.py --config configs/baseline.yaml
python generate_grouped_cv_report.py --config configs/baseline.yaml
```

Grouped CV 输出目录为 `outputs/grouped_cv_5class/`，根目录结果汇总为 `EXPERIMENT_SUMMARY.md`。

在现有 grouped CV 结果上运行 session-level aggregation：

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
python session_aggregation_cv.py --config configs/baseline.yaml
python generate_session_aggregation_report.py --config configs/baseline.yaml
```

Session aggregation 输出目录为 `outputs/grouped_cv_5class_session_agg/`。

运行第四章 `0/2/4` 三分类主实验：

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
python chapter4_024_experiments.py --config configs/chapter4_024.yaml
python generate_chapter4_024_report.py --config configs/chapter4_024.yaml
```

输出目录为 `outputs/chapter4_024/`。

运行 `summary-MMmodel` 主实验：

```bash
source /home/oi/miniforge3/etc/profile.d/conda.sh
conda activate dl
python summary_mmmodel_experiments.py --config configs/summary_mmmodel.yaml
python generate_summary_mmmodel_report.py --config configs/summary_mmmodel.yaml
```

输出目录为 `summary-MMmodel/`。

如果启动时看到 `torch 2.x+cpu` 或 `cuda_available=False`，说明当前并没有跑在 GPU 环境里，先切到 `dl` 再重新执行。
