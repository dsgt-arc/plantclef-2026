# Submission Log — Experiment 003: TileQ-Decoder

Source-of-truth log for every CSV submitted to the PlantCLEF 2026 Kaggle
leaderboard from this experiment. Records the exact `-m` message, submission
date, and public LB score so the paper can cite verbatim and post-submission
deltas vs expectation are auditable.

## Submission convention

```bash
kaggle competitions submit \
    -c plantclef-2026 \
    -f submissions/<csv-name> \
    -m "<tag>: <method-detail> (<context>, expect F1=<X.XXXXX>)"
```

- `<tag>` ∈ {A0a, A0b, A1, A2, A3} — see spec.md §5 ablation table.
- `<method-detail>` is brief: `sorted-prob top-9 grid_4x4`,
  `decoder samplerA topk10 seed42`, etc.
- `<context>` calibrates the LB read: "2025 reproduction", "organizers'
  approach", "decoder@k=9 vs A0b", "α=0.50 blend".
- Expected F1 always included so the LB delta vs expectation is logged.

Kaggle daily submission cap: **100/day** (verified 2026-05-05; earlier
spec/CONTEXT note of "~5/day" was wrong). Plenty of headroom — exp 003's
full plan (4 headlines + 12 k-sweep + 11 A3 = 27 submissions) fits in one day.

## Log

| Tag | CSV | Kaggle message | Submitted | LB F1 |
|---|---|---|---|---:|
| A0a | `A0a_max_prob_grid4.csv` | `A0a: max-prob top-10 grid_4x4 (organizers' approach, expect F1≈0.23)` | 2026-05-03 | **0.22986** |
| A0b | `A0b_sorted_prob_topk9_grid4.csv` | `A0b: sorted-prob top-9 grid_4x4 (2025 reproduction, expect F1=0.30810)` | 2026-05-03 | **0.30810** |
| A1@k9 | `A1_samplerA_topk9_seed42.csv` | `A1: decoder samplerA topk9 seed42 (H1 anchor vs A0b 0.30810)` | 2026-05-05 | **0.12180** |
| A1@k10 | `A1_samplerA_topk10_seed42.csv` | `A1: decoder samplerA topk10 seed42 (organizers' k)` | 2026-05-05 | **0.11640** |
| A2@k9 | `A2_samplerB_topk9_seed42.csv` | `A2: decoder samplerB tau=0.65 topk9 seed42 (H2 contrast)` | 2026-05-05 | **0.09060** |
| A2@k10 | `A2_samplerB_topk10_seed42.csv` | `A2: decoder samplerB tau=0.65 topk10 seed42` | 2026-05-05 | **0.08445** |
| A3-α0.00 | `A3_alpha0p00.csv` | _pending — α=0, pure max_prob baseline (sanity check)_ | — | — |
| A3-α0.25 | `A3_alpha0p25.csv` | _pending_ | — | — |
| A3-α0.50 | `A3_alpha0p50.csv` | _pending_ | — | — |
| A3-α0.75 | `A3_alpha0p75.csv` | _pending_ | — | — |
| A3-α1.00 | `A3_alpha1p00.csv` | _pending — α=1, pure decoder (sanity check)_ | — | — |

A3 rows assume best of {A1, A2} as the decoder source per spec §2.5; α=0 and
α=1 are sanity-check duplicates of the cached `max_prob` baseline and the
chosen decoder respectively — the LB scores should match exactly.
