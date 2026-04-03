# HCAF Compression Search Results

Source config: `configs/hcaf_confgate_compression_search.yaml`

Completed variants:

| Variant | Window macro-F1 | Window std | Session macro-F1 | Session std |
| --- | ---: | ---: | ---: | ---: |
| `hcaf_comp_sa0_base_5s` | `0.8968` | `0.0495` | `0.8815` | `0.0838` |
| `hcaf_comp_sa0_no_summary_5s` | `0.9155` | `0.0133` | `0.9407` | `0.0838` |
| `hcaf_comp_sa0_summary_token_5s` | `0.8298` | `0.0805` | `0.8815` | `0.0838` |
| `hcaf_comp_sa0_pcen64_hp80_5s` | `0.8773` | `0.0458` | `0.8815` | `0.0838` |

Additional completed variants:

| Variant | Window macro-F1 | Window std | Session macro-F1 | Session std |
| --- | ---: | ---: | ---: | ---: |
| `hcaf_comp_sa0_pcen96_nofilter_5s` | `0.8891` | `0.0644` | `0.9407` | `0.0838` |
| `hcaf_comp_sa0_simplegate_5s` | `0.8307` | `0.0650` | `0.8296` | `0.1362` |

Recommended compressed variant from this round:

- `self_attention_layers = 0`
- `use_summary_in_repr = false`
- keep `PCEN96 + HP80`
- keep `confidence-aware gate + expert residual`
