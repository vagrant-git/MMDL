# Canonical Model Identity

本仓库当前唯一正式默认模型为：

- display name: `HCAF-PCEN-DualXAttn`
- canonical experiment lineage: `SA=0 + no-summary`
- implementation ids:
  - `hcaf_comp_sa0_no_summary_5s`：完整模型正式成绩来源
  - `hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s`：当前实现 / 部署对齐配置名
- aligned config: `configs/final_model_unified_evidence.yaml`
- canonical full-model result source: `outputs/hcaf_confgate_compression_search`
- supplementary evidence source: `summary-MMmodel/final_model_unified_evidence`
- task: `0 / 2 / 4` 三分类
- primary metric: `window macro-F1`

关键结构口径：

- audio frontend: `PCEN96 + HP80`
- PQ branch: `1D CNN stem + TCN`
- interactions: `PQ cross-attention + audio-sensor cross-attention`
- decision: `confidence-aware gate + expert residual`
- compression switches: `self_attention_layers = 0`, `use_summary_in_repr = false`

当前文档统一引用的完整模型正式成绩为：

- window accuracy: `0.9260 ± 0.0161`
- window macro-F1: `0.9155 ± 0.0133`
- session macro-F1: `0.9407 ± 0.0838`

说明：

- 完整模型成绩统一引用 `outputs/hcaf_confgate_compression_search/overall_results.csv` 中的 `hcaf_comp_sa0_no_summary_5s`
- Audio-only、Pressure+Flow-only、缺失模态等补充对照可继续引用各自实验目录
- 不再把 `summary-MMmodel/final_model_unified_evidence` 中那组较低完整模型成绩当作默认模型正式结果

为避免误读，以下模型或配置目前都不是默认模型：

- `HCAF-LogMel96-DualXAttn`
- `hcaf_confgate_residual_logmel96_sa0_nosummary_5s`
- `configs/final_model_logmel.yaml`
- `configs/final_model_logmel_only.yaml`

如果文档中出现与本文件冲突的旧表述，以本文件和 `README.md` 为准。
