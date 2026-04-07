# All Experiment Results

当前默认模型展示名为 `HCAF-PCEN-DualXAttn`，对应实验配置 ID 为 `hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s`。

唯一正式默认模型的短声明见 [MODEL_IDENTITY.md](/home/wangshuai/MMDL/MODEL_IDENTITY.md)。

## Current Default

| name | experiment id | source | window macro-F1 | session macro-F1 |
| --- | --- | --- | ---: | ---: |
| `HCAF-PCEN-DualXAttn` | `hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s` | `summary-MMmodel/final_model_unified_evidence` | `0.8225 ± 0.1681` | `0.8148 ± 0.2619` |

## Unified Evidence

| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `Audio-only PCEN96 HP80` | `0.7052 ± 0.0667` | `0.8296 ± 0.1362` |
| `Pressure+Flow-only` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` |
| `HCAF-PCEN-DualXAttn` | `0.8225 ± 0.1681` | `0.8148 ± 0.2619` |
| `HCAF-PCEN-DualXAttn without audio` | `0.7578 ± 0.1570` | `0.7333 ± 0.1257` |
| `HCAF-PCEN-DualXAttn without pressure` | `0.9379 ± 0.0221` | `0.9407 ± 0.0838` |
| `HCAF-PCEN-DualXAttn without flow` | `0.9116 ± 0.0359` | `0.9407 ± 0.0838` |
| `HCAF-PCEN-DualXAttn audio only` | `0.7086 ± 0.0726` | `0.7926 ± 0.1826` |

Source:

- [`summary-MMmodel/final_model_unified_evidence/overall_results.csv`](summary-MMmodel/final_model_unified_evidence/overall_results.csv)

## Historical Search Records

| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `SA=0 base` | `0.8968 ± 0.0495` | `0.8815 ± 0.0838` |
| `SA=0 + no-summary` | `0.9155 ± 0.0133` | `0.9407 ± 0.0838` |
| `SA=0 + summary-token` | `0.8298 ± 0.0805` | `0.8815 ± 0.0838` |
| `SA=0 + PCEN64 HP80` | `0.8773 ± 0.0458` | `0.8815 ± 0.0838` |
| `SA=0 + PCEN96 nofilter` | `0.8891 ± 0.0644` | `0.9407 ± 0.0838` |
| `SA=0 + simplegate` | `0.8307 ± 0.0650` | `0.8296 ± 0.1362` |

Source:

- [`outputs/hcaf_confgate_compression_search/overall_results.csv`](outputs/hcaf_confgate_compression_search/overall_results.csv)

## Audio Frontend Follow-up

| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `HCAF confgate+residual base` | `0.8671 ± 0.0547` | `0.8815 ± 0.0838` |
| `HCAF confgate+residual + preemphasis 16k` | `0.8541 ± 0.0405` | `0.8815 ± 0.0838` |
| `HCAF confgate+residual + preemphasis 12k` | `0.7705 ± 0.1494` | `0.7926 ± 0.1826` |
| `HCAF confgate+residual + PCEN96 HP80` | `0.9207 ± 0.0261` | `0.9407 ± 0.0838` |

Source:

- [`summary-MMmodel/hcaf_confgate_improve_search/overall_results.csv`](summary-MMmodel/hcaf_confgate_improve_search/overall_results.csv)

## Earlier Structural References

| source | note |
| --- | --- |
| `outputs/hcaf_audioresnet_joint_sa_ablation` | joint self-attention ablation on earlier `audioresnet_xattn` line |
| `outputs/hcaf_audioresnet_one_vs_two_xattn` | one-stage vs two-stage cross-attention on earlier `audioresnet_xattn` line |
| `summary-MMmodel/pq_vs_multimodal_check` | PQ-only vs earlier HCAF comparison |
| `summary-MMmodel/hcaf_audioresnet_xattn_vs_concat` | earlier `audioresnet_xattn` route comparison |

## Default Recommendation

- 当前默认模型使用 `HCAF-PCEN-DualXAttn`
- 当前正式成绩只引用 `summary-MMmodel/final_model_unified_evidence`
- 历史搜索表保留作补充，不再与默认结果混写
