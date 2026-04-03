# All Experiment Results

当前默认模型展示名为 `HCAF-PCEN-XAttn`，对应实验配置 ID 为 `hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s`。

为了避免历史实验结果分散在多个目录中，这个文件把各个 `EXPERIMENT_SUMMARY.md` 的已整理结果集中到一个 Markdown 中。未跑完的实验暂不写入“当前默认模型”结论，只保留已完成结果。

## Current Default

| name | experiment id | source | window macro-F1 | session macro-F1 |
| --- | --- | --- | ---: | ---: |
| `HCAF-PCEN-XAttn` | `hcaf_confgate_residual_pcen96hp80_sa0_nosummary_5s` | `outputs/hcaf_confgate_compression_search` | `0.9155 ± 0.0133` | `0.9407 ± 0.0838` |

## outputs/hcaf_audioresnet_joint_sa_ablation/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `Audio R18 + PQ cross-attention 5 s without joint self-attention` | `0.9018 ± 0.0796` | `0.8815 ± 0.0838` |
| `Audio R18 + PQ cross-attention 5 s with joint self-attention` | `0.9130 ± 0.0385` | `0.8815 ± 0.0838` |

## outputs/hcaf_audioresnet_one_vs_two_xattn/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `Audio R18 + PQ one cross-attention stage 5 s` | `0.8729 ± 0.0617` | `0.8815 ± 0.0838` |
| `Audio R18 + PQ two cross-attention stages 5 s` | `0.9158 ± 0.0703` | `0.9407 ± 0.0838` |

## outputs/hcaf_audioresnet_pq_resnet_iter/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `Audio R18 ImageNet + PQ R18 long tokens + attn2` | `0.8993 ± 0.0663` | `0.9407 ± 0.0838` |

## outputs/hcaf_audioresnet_pq_seqmodels_iter2/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `Audio R18 ImageNet + PQ TCN (baseline)` | `0.9145 ± 0.0745` | `0.9407 ± 0.0838` |
| `Audio R18 ImageNet + PQ TCN (ReduceLROnPlateau)` | `0.9111 ± 0.0826` | `0.9407 ± 0.0838` |

## outputs/hcaf_audioresnet_pq_seqmodels_iter3/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `Audio R18 ImageNet + PQ TCN (baseline)` | `0.9145 ± 0.0745` | `0.9407 ± 0.0838` |
| `Audio R18 ImageNet + PQ TCN (Focal loss)` | `0.9232 ± 0.0902` | `0.9407 ± 0.0838` |

## outputs/hcaf_confgate_compression_search/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `HCAF compressed base SA0 PCEN96 HP80` | `0.8968 ± 0.0495` | `0.8815 ± 0.0838` |
| `HCAF-PCEN-XAttn` | `0.9155 ± 0.0133` | `0.9407 ± 0.0838` |
| `HCAF compressed SA0 summary token attention` | `0.8298 ± 0.0805` | `0.8815 ± 0.0838` |
| `HCAF compressed SA0 PCEN64 HP80` | `0.8773 ± 0.0458` | `0.8815 ± 0.0838` |
| `HCAF compressed SA0 PCEN96 no filter` | `0.8891 ± 0.0644` | `0.9407 ± 0.0838` |

## outputs/iter_pressure_flow_fusion/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `Pressure+Flow (gated)` | `0.9357 ± 0.0272` | `0.9407 ± 0.0838` |
| `Pressure+Flow (softmax-gate)` | `0.9461 ± 0.0215` | `1.0000 ± 0.0000` |
| `Pressure+Flow (concat-mlp)` | `0.9310 ± 0.0201` | `0.9407 ± 0.0838` |

## summary-MMmodel/audio_frontend_followup/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `HCAF base audio` | `0.6194 ± 0.1627` | `0.6630 ± 0.2435` |
| `HCAF + preemphasis` | `0.5738 ± 0.2860` | `0.6630 ± 0.3584` |
| `HCAF + PCEN96 HP80` | `0.7755 ± 0.1674` | `0.7926 ± 0.1826` |

## summary-MMmodel/audio_frontend_hcaf_round2/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `HCAF base audio` | `0.9077 ± 0.0774` | `0.9407 ± 0.0838` |
| `HCAF + PCEN96 HP80` | `0.8880 ± 0.0820` | `0.9407 ± 0.0838` |
| `HCAF + preemphasis + fixed top crop` | `0.8155 ± 0.1374` | `0.8815 ± 0.0838` |
| `HCAF + preemphasis 12k` | `0.8941 ± 0.0673` | `0.9407 ± 0.0838` |

## summary-MMmodel/final_model_unified_evidence/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `Audio-only PCEN96 HP80` | `0.7052 ± 0.0667` | `0.8296 ± 0.1362` |
| `Pressure+Flow-only` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` |
| `HCAF final full multimodal` | `0.9207 ± 0.0261` | `0.9407 ± 0.0838` |
| `HCAF final without audio` | `0.9394 ± 0.0379` | `0.9407 ± 0.0838` |
| `HCAF final without pressure` | `0.8152 ± 0.1187` | `0.8148 ± 0.2619` |
| `HCAF final without flow` | `0.8982 ± 0.0656` | `0.9407 ± 0.0838` |
| `HCAF final audio only` | `0.8734 ± 0.0496` | `0.8815 ± 0.0838` |

## summary-MMmodel/hcaf_arch_search/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `HCAF PCEN base` | `0.9259 ± 0.0079` | `0.9407 ± 0.0838` |
| `HCAF PCEN batch 8` | `0.9022 ± 0.0154` | `0.8815 ± 0.0838` |
| `HCAF PCEN batch 32` | `0.8343 ± 0.0892` | `0.8296 ± 0.1362` |
| `HCAF PCEN short attention` | `0.8504 ± 0.0701` | `0.8259 ± 0.1406` |
| `HCAF PCEN long attention` | `0.8477 ± 0.0807` | `0.8296 ± 0.1362` |
| `HCAF PCEN ResNet18 encoders` | `0.8933 ± 0.0271` | `0.8815 ± 0.0838` |

## summary-MMmodel/hcaf_audioresnet_pq_1dmodels/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `Audio R18 ImageNet + PQ TCN` | `0.9145 ± 0.0745` | `0.9407 ± 0.0838` |

## summary-MMmodel/hcaf_audioresnet_pq_complex/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `Audio R18 ImageNet + PQ R18` | `0.8966 ± 0.0711` | `0.9407 ± 0.0838` |

## summary-MMmodel/hcaf_audioresnet_pq_seqmodels/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `Audio R18 ImageNet + PQ TCN` | `0.9145 ± 0.0745` | `0.9407 ± 0.0838` |
| `Audio R18 ImageNet + PQ GRU` | `0.8837 ± 0.0659` | `0.8222 ± 0.0000` |
| `Audio R18 ImageNet + PQ CNN-GRU` | `0.9181 ± 0.0746` | `0.9407 ± 0.0838` |

## summary-MMmodel/hcaf_audioresnet_unified_evidence/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `HCAF Audio R18 ImageNet audio only` | `0.8709 ± 0.0722` | `0.9407 ± 0.0838` |
| `Pressure+Flow-only` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` |
| `Audio R18 ImageNet + PQ TCN` | `0.9145 ± 0.0745` | `0.9407 ± 0.0838` |

## summary-MMmodel/hcaf_audioresnet_xattn_vs_concat/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `Audio R18 ImageNet audio only 5 s` | `0.8709 ± 0.0722` | `0.9407 ± 0.0838` |
| `Pressure+Flow-only` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` |
| `Audio R18 + PQ direct concat 5 s` | `0.7800 ± 0.1610` | `0.7852 ± 0.1923` |
| `Audio R18 + PQ cross-attention 5 s` | `0.9145 ± 0.0745` | `0.9407 ± 0.0838` |

## summary-MMmodel/hcaf_confgate_filter_lowpass300/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `HCAF confgate+residual + PCEN96 LP300` | `0.8570 ± 0.0769` | `0.8815 ± 0.0838` |
| `HCAF confgate+residual + PCEN96 BP80-300` | `0.7709 ± 0.1303` | `0.8259 ± 0.1406` |

## summary-MMmodel/hcaf_confgate_filter_search/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `HCAF confgate+residual base` | `0.8671 ± 0.0547` | `0.8815 ± 0.0838` |

## summary-MMmodel/hcaf_confgate_improve_search/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `HCAF confgate+residual base` | `0.8671 ± 0.0547` | `0.8815 ± 0.0838` |
| `HCAF confgate+residual + preemphasis 16k` | `0.8541 ± 0.0405` | `0.8815 ± 0.0838` |
| `HCAF confgate+residual + preemphasis 12k` | `0.7705 ± 0.1494` | `0.7926 ± 0.1826` |
| `HCAF confgate+residual + PCEN96 HP80` | `0.9207 ± 0.0261` | `0.9407 ± 0.0838` |

## summary-MMmodel/hcaf_confgate_window_lengths/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `HCAF confgate+residual 5 s` | `0.6745 ± 0.0918` | `0.7545 ± 0.0958` |
| `HCAF confgate+residual 10 s` | `0.6613 ± 0.1246` | `0.7619 ± 0.1695` |
| `HCAF confgate+residual 20 s` | `0.7577 ± 0.1943` | `0.7926 ± 0.1826` |

## summary-MMmodel/hcaf_confgate_window_lengths_6_8_15/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `HCAF confgate+residual 6 s` | `0.6419 ± 0.1336` | `0.7026 ± 0.0868` |
| `HCAF confgate+residual 8 s` | `0.7099 ± 0.2337` | `0.7407 ± 0.2516` |
| `HCAF confgate+residual 15 s` | `0.6651 ± 0.1116` | `0.6868 ± 0.0958` |

## summary-MMmodel/hcaf_fusion_gate_followup/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `HCAF legacy shared norm` | `0.7919 ± 0.1489` | `0.8815 ± 0.0838` |
| `HCAF norm fix` | `0.8728 ± 0.0529` | `0.8815 ± 0.0838` |
| `HCAF confidence-aware gate` | `0.6791 ± 0.1195` | `0.6857 ± 0.1931` |
| `HCAF confidence gate + expert residual` | `0.8847 ± 0.0733` | `0.8815 ± 0.0838` |

## summary-MMmodel/hcaf_missing_modalities/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `HCAF full multimodal` | `0.8065 ± 0.1489` | `0.8296 ± 0.1362` |
| `HCAF without audio` | `0.7028 ± 0.2052` | `0.8042 ± 0.2769` |
| `HCAF without pressure` | `0.8662 ± 0.0910` | `0.8815 ± 0.0838` |
| `HCAF without flow` | `0.7266 ± 0.1130` | `0.7026 ± 0.0868` |
| `HCAF audio only` | `0.6255 ± 0.1125` | `0.7386 ± 0.1319` |

## summary-MMmodel/hcaf_resnet18_freeze_schedule/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `ResNet18 ImageNet full finetune` | `0.8966 ± 0.0711` | `0.9407 ± 0.0838` |
| `ResNet18 ImageNet freeze 1 epoch` | `0.8757 ± 0.0560` | `0.8815 ± 0.0838` |

## summary-MMmodel/hcaf_resnet18_imagenet_only/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `HCAF audio ResNet18 scratch` | `0.8348 ± 0.0433` | `0.8222 ± 0.0000` |
| `HCAF audio ResNet18 ImageNet` | `0.9111 ± 0.0839` | `0.9407 ± 0.0838` |

## summary-MMmodel/pq_vs_multimodal_check/EXPERIMENT_SUMMARY.md
| model | window macro-F1 | session macro-F1 |
| --- | ---: | ---: |
| `Pressure+Flow-only` | `0.7499 ± 0.2513` | `0.8519 ± 0.2095` |
| `HCAF norm fix` | `0.8011 ± 0.1424` | `0.8259 ± 0.1406` |
| `HCAF confidence gate + expert residual` | `0.7760 ± 0.0972` | `0.8815 ± 0.0838` |
