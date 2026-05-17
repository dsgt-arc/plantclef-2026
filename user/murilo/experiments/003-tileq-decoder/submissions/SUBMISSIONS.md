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
| A3-α0.00 | `A3_samplerA_alpha0p00_topk10_seed42.csv` | `A3: alpha=0.00 blend (samplerA + max_prob), topk=10, seed=42` | 2026-05-06 | **0.22144** |
| A3-α0.10 | `A3_samplerA_alpha0p10_topk10_seed42.csv` | `A3: alpha=0.10 blend (samplerA + max_prob), topk=10, seed=42` | 2026-05-06 | 0.21952 |
| A3-α0.20 | `A3_samplerA_alpha0p20_topk10_seed42.csv` | `A3: alpha=0.20 blend (samplerA + max_prob), topk=10, seed=42` | 2026-05-06 | 0.21827 |
| A3-α0.30 | `A3_samplerA_alpha0p30_topk10_seed42.csv` | `A3: alpha=0.30 blend (samplerA + max_prob), topk=10, seed=42` | 2026-05-06 | 0.21001 |
| A3-α0.40 | `A3_samplerA_alpha0p40_topk10_seed42.csv` | `A3: alpha=0.40 blend (samplerA + max_prob), topk=10, seed=42` | 2026-05-06 | 0.19954 |
| A3-α0.50 | `A3_samplerA_alpha0p50_topk10_seed42.csv` | `A3: alpha=0.50 blend (samplerA + max_prob), topk=10, seed=42` | 2026-05-06 | 0.18931 |
| A3-α0.60 | `A3_samplerA_alpha0p60_topk10_seed42.csv` | `A3: alpha=0.60 blend (samplerA + max_prob), topk=10, seed=42` | 2026-05-06 | 0.17292 |
| A3-α0.70 | `A3_samplerA_alpha0p70_topk10_seed42.csv` | `A3: alpha=0.70 blend (samplerA + max_prob), topk=10, seed=42` | 2026-05-06 | 0.15277 |
| A3-α0.80 | `A3_samplerA_alpha0p80_topk10_seed42.csv` | `A3: alpha=0.80 blend (samplerA + max_prob), topk=10, seed=42` | 2026-05-06 | 0.14562 |
| A3-α0.90 | `A3_samplerA_alpha0p90_topk10_seed42.csv` | `A3: alpha=0.90 blend (samplerA + max_prob), topk=10, seed=42` | 2026-05-06 | 0.13154 |
| A3-α1.00 | `A3_samplerA_alpha1p00_topk10_seed42.csv` | `A3: alpha=1.00 blend (samplerA + max_prob), topk=10, seed=42` | 2026-05-06 | **0.11640** |

A3 source: A1 (Sampler A, the better-scoring decoder at 0.12180 LB).
α=1.00 LB = 0.11640 confirmed exact match to A1@k10 (sanity check passed).
α=0.00 LB = 0.22144 (vs A0a's 0.22986 — 0.008 lower due to tie-breaking
differences between numpy `argpartition+argsort` and pandas `sort` for
species with very close softmax probabilities; same top-10 sets in 96 % of
rows, ~80 rows with positions 8/9 or 9/10 swapped, occasionally a different
boundary species at rank 10). The relative A3 curve is internally
consistent (all 11 use identical tie-breaking), so the **monotone decrease
from α=0 to α=1** is the load-bearing finding.

**A3 verdict**: no α > 0 improves over α=0. Decoder probabilities provide
**zero complementary signal** to max_prob — they are purely additive noise
that lowers F1 monotonically with their weight in the blend. This is a
*strong* form of the "decoder is redundant and harmful" finding.
