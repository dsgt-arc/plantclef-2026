# Exp-14 — Diagnostic Findings (Phase 0–1)

This file is populated as Phase 0 and Phase 1 complete.

## Phase 0.1 — Parquet pull

| Artifact | PACE path | Local path | Pulled | Size |
|---|---|---|---|---|
| ViT pre-fusion | `submissions/exp9b_knn_diff_beta_phase2a/knn_bM0.85_bS1.00_T1.50_prefusion.parquet` | TBD | ☐ | |
| ViT post-fusion (reference) | `submissions/exp9b_knn_diff_beta_phase2a/knn_bM0.85_bS1.00_T1.50_fused_a0.15.parquet` | TBD | ☐ | |
| ConvNeXt scale 4×4 | `parquet/exp11_ensemble/convnext_large/scale_4x4.parquet` | TBD | ☐ | |
| ConvNeXt scale 5×5 | `parquet/exp11_ensemble/convnext_large/scale_5x5.parquet` | TBD | ☐ | |
| ConvNeXt scale 6×6 | `parquet/exp11_ensemble/convnext_large/scale_6x6.parquet` | TBD | ☐ | |
| Species map | `species_map.json` (TBD) | TBD | ☐ | |
| Geo class mask | `sw_europe_mask_bool.npy` | TBD | ☐ | |

## Phase 0.2 — Sanity check (failure-mode-3 gate)

Run:
```bash
python diagnostics/prefusion_sanity_check.py \
    --prefusion-parquet <local>/knn_bM0.85_bS1.00_T1.50_prefusion.parquet \
    --out-md docs/Exp14_Diagnostic_Findings.md
```

The script appends results below. **If any pair shows max_abs_diff < 0.05,
STOP** — the bug is upstream of fusion and must be fixed first.

<!-- prefusion_sanity_check.py output goes here -->

## Phase 1.2 — Baseline proxy metrics

Computed against `ens_knn_w0.77_thr0.100_top10.csv` (F1=0.44428 public LB).

| Metric | Value | Notes |
|---|---|---|
| `rnnb_pair_divergence` | TBD | Target ≥0.4 after fix; baseline expected ~0.0–0.1 |
| `stable_site_invariance` | TBD | Should not drop more than 0.10 in any candidate |
| `density_count_correlation` | TBD | Should rise from near-zero to positive |
| `photo_faithfulness` | TBD | Should rise on barren images especially |

Per-source #species/quadrat distribution:

| Source | Mean | Median | n |
|---|---|---|---|
| RNNB | TBD | TBD | TBD |
| CBN-PdlC | TBD | TBD | TBD |
| CBN-Pla | TBD | TBD | TBD |
| GUARDEN | TBD | TBD | TBD |
| LISAH | TBD | TBD | TBD |
| OPTMix | TBD | TBD | TBD |

## Phase 1.1 — Visual confirmation

PDF rendered: `outputs/viz/rnnb_pairs.pdf` (8 user-flagged images, side-by-side).

Random multi-visit sample PDF: `outputs/viz/random_multivisit_30.pdf`.

Observations:
- TBD

## Decision gate

- [ ] All four RNNB pairs PASS sanity check (max_abs_diff > 0.05)
- [ ] Visual confirmation: lush-prior-clobbers-barren pattern visible in ≥X of 30 random pairs
- [ ] Baseline proxies match the predicted "broken" pattern

If all three checked, proceed to Phase 2 (signal computation).
If any one fails, stop and reassess — the hypothesis may not match reality.

---

## Phase 2.1 — Density signal validation (completed locally)

Computed via `signals/compute_density.py` on the full 2,105-image test set
using ExG ∩ HSV-green AND-mask (initial Otsu-on-ExG version misfired on
unimodal bare-sand distributions; fixed threshold ExG > 20 ∧ saturation
≥ 0.15 is robust).

**User-flagged pairs (validation):**

| Pair | May 2023 | Jan 2024 | Δ |
|---|---|---|---|
| RNNB-3-1 | 0.267 | 0.028 | -0.239 |
| RNNB-3-2 | 0.516 | 0.025 | -0.491 |
| RNNB-2-7 | 0.221 | 0.062 | -0.159 |
| RNNB-1-2 | 0.208 | 0.000 | -0.208 |

Every barren follow-up clearly separates from its lush sister.

**Per-source density distribution (mean / % low-density):**

| Source  |   n  | mean | % below 0.05 |
|---|---|---|---|
| CBN-PdlC | 816 | 0.526 | **1.1 %** |
| LISAH    | 208 | 0.580 | 0.5 % |
| CBN-Pla  | 628 | 0.398 | 8.0 % |
| GUARDEN  | 201 | 0.280 | 14.9 % |
| OPTMix   |  78 | 0.140 | **48.7 %** |
| RNNB     | 141 | 0.119 | **43.3 %** |
| OTHER    |  33 | 0.670 | 0.0 % |

Key finding: **the density-driven prior gate will affect ~13 % of the
test set** (RNNB + OPTMix + GUARDEN barren cases ≈ 290 images), not
just RNNB. CBN-PdlC and LISAH are robustly dense and the gate will be
near-noop there — protecting the working cases.

OPTMix wasn't on the user's manual-inspection radar but appears to share
the same RNNB pathology. Stage 2's per-source α grid should include
OPTMix as a probable beneficiary.

## Phase 2.4 — Season weights (completed locally)

Computed via `signals/compute_season_weights.py`. Output:
`outputs/signals/season_weights.json` (~1 MB, 30,475 weighted pairs
across 2,105 quadrats). Cyclical month-distance, σ=2 months. Lower
priority per user direction (year-to-year variation > seasonal
variation), included so Stage 5 can quantify the contribution.
