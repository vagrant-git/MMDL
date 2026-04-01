## summary_mmmodel/hcaf_resnet18_scratch_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_resnet18_scratch_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8882, macro-F1=0.8924, precision=0.9291, recall=0.8867
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_resnet18_scratch_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_resnet18_scratch_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.8722, macro-F1=0.8240, precision=0.9012, recall=0.8176
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_resnet18_scratch_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_resnet18_scratch_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.8655, macro-F1=0.7880, precision=0.8053, recall=0.7774
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel | hcaf_resnet18_scratch_5s

- model: `HCAF audio ResNet18 scratch`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.8753 ± 0.0095, macro-F1=0.8348 ± 0.0433, precision=0.8785 ± 0.0530, recall=0.8272 ± 0.0452
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8333 ± 0.0000, macro-F1=0.8222 ± 0.0000, precision=0.8889 ± 0.0000, recall=0.8333 ± 0.0000

## summary_mmmodel/hcaf_resnet18_imagenet_5s/repeat1_fold1 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_resnet18_imagenet_5s/repeat1_fold1` 设定训练评估。
- result_window: acc=0.8000, macro-F1=0.7938, precision=0.8685, recall=0.7927
- result_session: acc=0.8333, macro-F1=0.8222, precision=0.8889, recall=0.8333

## summary_mmmodel/hcaf_resnet18_imagenet_5s/repeat1_fold2 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_resnet18_imagenet_5s/repeat1_fold2` 设定训练评估。
- result_window: acc=0.9613, macro-F1=0.9546, precision=0.9650, recall=0.9494
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel/hcaf_resnet18_imagenet_5s/repeat1_fold3 | multimodal

- python_env: `dl`
- model: `multimodal`
- method: 保持 baseline_5class 的预处理与模型结构不变，按 `summary_mmmodel/hcaf_resnet18_imagenet_5s/repeat1_fold3` 设定训练评估。
- result_window: acc=0.9823, macro-F1=0.9850, precision=0.9881, recall=0.9823
- result_session: acc=1.0000, macro-F1=1.0000, precision=1.0000, recall=1.0000

## summary_mmmodel | hcaf_resnet18_imagenet_5s

- model: `HCAF audio ResNet18 ImageNet`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.9146 ± 0.0815, macro-F1=0.9111 ± 0.0839, precision=0.9405 ± 0.0518, recall=0.9081 ± 0.0827
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

