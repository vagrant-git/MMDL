## summary_mmmodel | audio_only_pcen96hp80_5s

- model: `Audio-only PCEN96 HP80`
- modality: `audio_only`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.7170 ± 0.0578, macro-F1=0.7052 ± 0.0667, precision=0.7535 ± 0.0784, recall=0.7171 ± 0.0525
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8333 ± 0.1361, macro-F1=0.8296 ± 0.1362, precision=0.9074 ± 0.0693, recall=0.8333 ± 0.1361

## summary_mmmodel | pressure_flow_5s

- model: `Pressure+Flow-only`
- modality: `pressure_flow`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.7692 ± 0.2274, macro-F1=0.7499 ± 0.2513, precision=0.7495 ± 0.2684, recall=0.8192 ± 0.1529
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.1571, macro-F1=0.8519 ± 0.2095, precision=0.8333 ± 0.2357, recall=0.8889 ± 0.1571

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_5s

- model: `HCAF final full multimodal`
- modality: `multimodal`
- group: `main`
- window_sec: `5.0`
- window_mean_std: acc=0.9303 ± 0.0197, macro-F1=0.9207 ± 0.0261, precision=0.9435 ± 0.0185, recall=0.9105 ± 0.0259
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_minus_audio_5s

- model: `HCAF final without audio`
- modality: `multimodal_minus_audio`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.9398 ± 0.0412, macro-F1=0.9394 ± 0.0379, precision=0.9551 ± 0.0219, recall=0.9357 ± 0.0391
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_minus_pressure_5s

- model: `HCAF final without pressure`
- modality: `multimodal_minus_pressure`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.8358 ± 0.1046, macro-F1=0.8152 ± 0.1187, precision=0.8558 ± 0.0816, recall=0.8234 ± 0.1096
- best_session_method: `mean_probability_pooling`
- session_mean_std: acc=0.8333 ± 0.2357, macro-F1=0.8148 ± 0.2619, precision=0.8333 ± 0.2357, recall=0.8333 ± 0.2357

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_minus_flow_5s

- model: `HCAF final without flow`
- modality: `multimodal_minus_flow`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.9132 ± 0.0530, macro-F1=0.8982 ± 0.0656, precision=0.9302 ± 0.0452, recall=0.8872 ± 0.0666
- best_session_method: `majority_voting`
- session_mean_std: acc=0.9444 ± 0.0786, macro-F1=0.9407 ± 0.0838, precision=0.9630 ± 0.0524, recall=0.9444 ± 0.0786

## summary_mmmodel | hcaf_confgate_residual_pcen96hp80_audio_only_5s

- model: `HCAF final audio only`
- modality: `multimodal`
- group: `ablation`
- window_sec: `5.0`
- window_mean_std: acc=0.8823 ± 0.0525, macro-F1=0.8734 ± 0.0496, precision=0.9040 ± 0.0334, recall=0.8685 ± 0.0489
- best_session_method: `majority_voting`
- session_mean_std: acc=0.8889 ± 0.0786, macro-F1=0.8815 ± 0.0838, precision=0.9259 ± 0.0524, recall=0.8889 ± 0.0786

