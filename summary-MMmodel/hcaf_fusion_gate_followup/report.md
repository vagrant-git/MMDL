# HCAF Fusion/Gate Follow-up

## Setup

- Date: `2026-03-30`
- Task: `0 / 2 / 4` three-class classification
- Split strategy: grouped CV at `session` level, fixed before windowing
- Window length: `5 s`
- Audio frontend: base `log-Mel 64`
- Goal: isolate fusion/gating changes without changing data split, windowing, or training budget

## Model Variants

1. `hcaf_legacy_sharednorm_5s`
   - reproduces the previous HCAF normalization path where pressure/flow branch representations reuse the audio `LayerNorm`
2. `hcaf_normfix_5s`
   - fixes the branch normalization path with dedicated sensor-side norms
3. `hcaf_confgate_5s`
   - adds a confidence-aware reliability gate using modality-specific expert logits
4. `hcaf_confgate_residual_5s`
   - adds confidence-aware gate plus expert-logit residual fusion

## Results

| Experiment | Window macro-F1 | Session macro-F1 |
| --- | ---: | ---: |
| HCAF legacy shared norm | `0.7919 ± 0.1489` | `0.8815 ± 0.0838` |
| HCAF norm fix | `0.8728 ± 0.0529` | `0.8815 ± 0.0838` |
| HCAF confidence-aware gate | `0.6791 ± 0.1195` | `0.6857 ± 0.1931` |
| HCAF confidence gate + expert residual | `0.8847 ± 0.0733` | `0.8815 ± 0.0838` |

## Findings

- The shared-norm path was a real weak point. Fixing it improved window-level macro-F1 by about `+0.081` and sharply reduced variance.
- Confidence-aware gating alone was unstable and degraded both window-level and session-level performance.
- Adding expert-logit residuals recovered the window-level performance and slightly exceeded the pure norm-fix variant.
- Session-level macro-F1 remained tied between `legacy shared norm`, `norm fix`, and `confidence gate + expert residual`.
- Under the current grouped split, majority voting was still the best session aggregation method for all variants.

## Interpretation

- Audio information is not the only bottleneck anymore.
- The main gain from this round came from stabilizing representation geometry, not from making the final gate more aggressive.
- Confidence signals can easily over-amplify the stronger modality early in training; without a stronger regularizer, this hurts robustness.
- The current session-level metric is coarse because each fold has very few test sessions, so meaningful improvements are easier to observe at the window level first.

## Recommendation

- Keep the normalization fix.
- Do not adopt confidence-aware gate alone.
- If continuing HCAF refinement, prefer `norm fix` or `norm fix + expert residual` as the new base for the next round.
