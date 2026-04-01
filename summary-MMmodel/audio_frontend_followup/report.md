# HCAF Audio Frontend Follow-up

## Setup

- task: `0 / 2 / 4` 三分类
- split: grouped CV，先按 session 划分再切窗
- window: `5 s`
- model: `HCAF-Net`
- compared audio frontends:
  - base
  - `log-Mel 64 + preemphasis`
  - `PCEN 96 + high-pass 80 Hz`

## Results

| Variant | Window Macro-F1 | Session Macro-F1 |
| --- | --- | --- |
| HCAF base audio | 0.6194 ± 0.1627 | 0.6630 ± 0.2435 |
| HCAF + preemphasis | 0.5738 ± 0.2860 | 0.6630 ± 0.3584 |
| HCAF + PCEN96 HP80 | 0.7755 ± 0.1674 | 0.7926 ± 0.1826 |

## Conclusion

- `PCEN 96 + high-pass 80 Hz` 是目前唯一在 HCAF 上形成稳定收益的音频前端
- `preemphasis` 虽然能提升 audio-only，但没有稳定迁移到 HCAF
- 改善音频前端后，HCAF 的 session-level macro-F1 已提升到 `0.7926`
- 该结果已经明显优于原 gated fusion baseline，但仍低于 `pressure_flow-only`
- 这说明当前主要瓶颈已经从“音频完全无信息”转为“如何让音频增益稳定地进入融合层”
