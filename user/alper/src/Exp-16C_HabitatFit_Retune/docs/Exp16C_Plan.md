# Exp-16C — Re-tune Exp-16 mechanisms for the new ViT-L pipeline

## Goal
Apply the Exp-16 stack (habitat-fit demotion + per-source N-floor admission)
to the new Exp-14C pipeline (ViT-L + LoRA-ArcFace-kNN + ConvNeXt P, with
sim_w on RNNB + aquatic habitat mask), then re-tune Exp-16 hyperparameters
to match or beat the locked production best (F1=0.49842 on the OLD pipeline).

## Why a separate experiment
The Exp-16 mechanisms in the original locked best were heavily LB-tuned to
the OLD pipeline's specific output distribution (β=0.25, R=200, n_near_min=30,
ratio_floor per source, n_floor=2 only on OPTMix). Applying them verbatim to
the new pipeline gave only partial transfer:
- sim_w transferred at 22% (+0.00211 vs old +0.00982)
- aquatic habitat mask transferred at 338% (+0.00548 vs old +0.00162)
- alpine biogeo transferred at 0% (no impact)

The new ViT-L+kNN is a structurally different probability distribution.
Some Exp-16 params will transfer cleanly (habitat-fit β is geographic, so
should be near-full), others may need re-tuning (admission rules depend on
classifier confidence calibration).

## Reference points
| Pipeline | F1 | Notes |
|---|---|---|
| New Exp-14C raw best | 0.45728 | sim_w + habitat mask, no Exp-16 |
| Old + Exp-16 (locked production) | **0.49842** | target to beat |
| Predicted Exp-16C best | 0.490–0.505 | depends on transfer ratios |

## Sweep design (Path B)

### Phase 1 — Transfer measurement (1 LB)
Apply Exp-16 verbatim with locked params:
- β=0.25, R=200km, n_near_min=30, alt-on, refined centroids
- N3_r0.5 globally + LISAH r=0.6 + OPTMix r=0.3 n=2

This measures how the full Exp-16 stack transfers without any re-tuning.

### Phase 2 — habitat-fit β re-tune (4 LB)
β ∈ {0.15, 0.20, 0.30, 0.40} at fixed R=200, n=30, alt-on, B_OPTMix_n2 admission.
β controls demotion strength — sharper model may want different.

### Phase 3 — habitat-fit R re-tune (3 LB, conditional)
R ∈ {150, 250, 300} at locked (β, n, alt). Only run if β doesn't shift.

### Phase 4 — admission re-tune (4 LB)
Test what transfers from per-source admission rules:
- Phase A only (no OPTMix n=2)
- No per-source overrides (pure global)
- Tighter ratio_floor (r=0.7 global)
- Looser n_floor (n=2 global)

Total: ~12 CSVs / ~10 LB submissions across 2-3 days.

## Decision rules

| Phase 1 result | Action |
|---|---|
| ≥ 0.495 | Strong transfer — submit P2 β variants to confirm peak; ship best |
| 0.475-0.495 | Partial — submit P2 + 1-2 P3 variants; aggressive search |
| 0.460-0.475 | Weak — submit P4 admission variants first; then β sweep |
| < 0.460 | Very weak — diagnose input parquet correctness before more LB |

## Inputs

### Required from Exp-14C
- Locked ensemble parquet (saved by `apply_exp14_mechanisms.sbatch`):
  `${PLANTCLEF_ROOT}/aerten3/submissions/exp14c_mechanisms/exp14c_simw_b0.70_w0.88_thr0.085_w0.88_ensemble.parquet`

### Required uploaded files (git-ignored, must scp from local)
| File | Source path | Size |
|---|---|---|
| `species_geo_profile.parquet` | `Exp-16/outputs/habitat_classifier/` | 865 KB |
| `refined_source_ref.json` | `Exp-16/outputs/habitat_classifier/` | 1.4 KB |
| `B_OPTMix_n2.json` | `Exp-16/outputs/per_source/configs/` | tiny |
| `A_combined.json` | `Exp-16/outputs/per_source/configs/` | tiny |

### Reused from existing PACE state
- `Exp-14/outputs/signals/aquatic_blacklist_mask.npy` (already uploaded)
- `Exp-14/outputs/signals/biogeo_per_source.npz` (already uploaded)
- `data/raw/class_id_to_species_mapping/class_to_species_id.json`
- `aerten3/species_masking/sw_europe_mask_bool.npy`

### Scripts reused (no new code)
- `Exp-16_Multi_Species_Discovery/scripts/apply_habitat_fit.py` — applies
  habitat-fit demotion + admission in one pass
