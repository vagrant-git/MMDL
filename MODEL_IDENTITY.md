# Canonical Model Identity

本仓库当前唯一正式默认模型为：

- display name: `HCAF-PCEN-DualXAttn`
- experiment id: `hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s`
- config: `configs/final_model_unified_evidence.yaml`
- result dir: `summary-MMmodel/final_model_unified_evidence`
- task: `0 / 2 / 4` 三分类
- primary metric: `window macro-F1`

关键结构口径：

- audio frontend: `PCEN96 + HP80`
- PQ branch: `1D CNN stem + TCN`
- interactions: `PQ cross-attention + audio-sensor cross-attention`
- decision: `confidence-aware gate + expert residual`
- compression switches: `self_attention_layers = 0`, `use_summary_in_repr = false`

为避免误读，以下模型或配置目前都不是默认模型：

- `HCAF-LogMel96-DualXAttn`
- `hcaf_confgate_residual_logmel96_sa0_nosummary_5s`
- `configs/final_model_logmel.yaml`
- `configs/final_model_logmel_only.yaml`

如果文档中出现与本文件冲突的旧表述，以本文件和 `README.md` 为准。
