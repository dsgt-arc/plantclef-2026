# Exp-16 — Multi-species discovery on densely-occluded quadrats

## Motivation

User-flagged failure mode: in densely-vegetated quadrat photos (alpine meadow,
GUARDEN holm-oak understory, LISAH multi-herb), 5–15 distinct species share the
frame, several occupying < 5 % of pixel area, often partially occluded by a
dominant canopy species. Manual inspection of the Stage 13 winner shows small
species are systematically under-represented in the top-10 predictions.

## Prior-art constraints (binding — do not re-test)

Survey of all prior experiment docs (Exp-3 through Exp-14):

| Idea | Already tested? | Don't re-test |
|---|---|---|
| Quantile aggregation across tiles (q=0.75–0.95) | Exp-8: lost to max_prob by ~0.044 | yes |
| Coarse tile scales (1×1, 2×2, 3×3) | Exp-7: hurt at every threshold | yes |
| 7×7 tile scale (finer) | Exp-7: +0.0008 for 49 extra tiles | yes |
| Threshold lowering globally | Exp-8 / Exp-14 Stage 10: noise > recall | yes |
| Per-image alpha modulation (density gate) | Exp-14 Stage 7: ±0.001 of winner | yes |
| Global ensemble weight w sweep | Exp-11 / Exp-14 Stage 6: crashed at extremes | yes |
| Mean / noisy-or aggregation | Exp-4: max_prob won | yes |
| Per-source α + per-source aggregator | Exp-14 Stage 4: +0.00982 (locked) | already adopted |
| Aquatic species blacklist | Exp-14 Stage 12: +0.00162 (locked) | already adopted |

## Genuinely virgin ground

Per the same survey:

- **L1: per-tile class-axis vote-share signal** — no prior experiment touched the
  per-tile *class-axis* (only tile-axis aggregators / quantiles). Tested
  here as Phase 1.
- **L3: per-image ensemble weight w schedule** — only fixed `w` was swept;
  per-image conditional weighting (e.g. by source / image entropy / occlusion
  proxy) was never attempted. Phase 2 if Phase 1 leaves headroom.
- **L4: ViT backbone re-finetune with Mixup + CutMix** — Exp-13 plan exists
  but is unexecuted; ConvNeXt (which got Mixup α=0.8 + CutMix α=1.0) is the
  only branch with occlusion-robust training. Phase 3 if Phases 1+2 hit
  ceiling. ViT-B (current production) was never retrained with mixup/cutmix.

## Design — ViT branch only

Per user direction, all phases apply to the ViT branch only. ConvNeXt symmetric
attempts deferred to a later experiment (Exp-16-B), gated on Phase 1+2 results.

## Phase 1 — Vote-share reranking (local CPU, post-hoc, 0 LB cost initially)

**Mechanism (one sentence):** *For each species, compute its multi-tile vote
share — fraction of tiles in which the species appears in the per-tile top-K —
and multiply the existing ViT prefusion probability by `(vote_share + ε)^β`,
so species with broad multi-tile support are preferred over single-tile
noise spikes.*

**Why this is genuinely new vs. Exp-8 quantile aggregation:** quantile
aggregation operates in *probability space across tiles* (e.g. p_75 of
softmax across the 77 tiles); vote-share operates in *rank space* (binary
"is this class in this tile's top-K"). A small species that's #4 on 5 tiles
gets full credit; a noise-spike species that's #1 on 1 tile but absent
elsewhere gets penalized. Quantile aggregation cannot distinguish these.

**Implementation:**

1. From local tile parquets (`scale_4x4.parquet`, `5x5`, `6x6` in
   `Exp-14/pace_downloads/`), compute per-image `vote_share[c]` =
   `#{tiles t : c ∈ argtop-K(softmax(logits_t))} / total_tiles`.
   - Using DINOv2-CLS tile parquets (the local copy). The Exp-9B ArcFace
     tile parquets live on PACE; if Phase 1 shows direction we move to
     PACE for the production-backbone signal.
   - K ∈ {5, 10} sweep.
2. Apply rerank factor to existing production ViT prefusion parquet
   (`knn_bM0.85_bS1.00_T1.50_prefusion.parquet`):
   `p_reranked[c] = p_prefusion[c] × (vote_share[c] + ε)^β`, ε = 0.01.
   - β ∈ {0, 0.25, 0.5, 0.75, 1.0}.
   - β=0 reproduces production prefusion exactly (graceful fallback).
3. Renormalize to keep prefusion sum-conserving (per-image L1 normalization
   to match the original prefusion's L1 sum).
4. Run renormalized prefusion through the locked Stage 13 pipeline:
   `ensemble_v2.py` → `apply_biogeo_mask.py` → candidate CSV.
5. Score every (β, K) variant on internal proxies vs Stage 13 winner.

**Phase 1 → Phase 2 gate (all must hold):**

- `rnnb_pair_divergence` ≥ Stage 13 baseline (no RNNB regression)
- `stable_site_invariance` ≥ Stage 13 baseline − 0.005 (no alpine collapse)
- `density_count_correlation` ≥ Stage 13 baseline − 0.01
- `multi_species_recovery` (new) > 0 (more distinct species discovered on
  high-density cohort vs Stage 13 winner)

If 0 (β, K) variants pass: close Phase 1 with negative-result report;
proceed to Phase 2.
If ≥ 1 passes: pick best by `multi_species_recovery`; submit to LB
(1 submission). β=0 wiring sanity submission optional (must reproduce
0.45611 ≤ 0.001 if submitted).

## Phase 2 — Per-image ensemble weight w schedule (gated, post-hoc, ≤ 2 LB)

**Triggered only if Phase 1 LB submission Δ ≤ +0.003 vs 0.45611 (sub-noise),
i.e. mechanism-true-but-leverage-small.**

**Mechanism (one sentence):** *Increase the ConvNeXt branch weight on
high-occlusion images (image-entropy or tile-disagreement proxy) where
ConvNeXt's mixup-trained occlusion robustness should help, while keeping
the production w=0.77 default elsewhere.*

**Implementation:**

1. Compute per-image **occlusion proxy** from local tile parquets:
   `proxy(image)` = mean per-tile entropy of softmax over top-50 classes
   (high entropy = no clear winning species per tile = many overlapping
   species).
2. Bin images into deciles by proxy.
3. Linear schedule: `w(image) = 0.77 + s × (proxy_decile - 5)`,
   where `s ∈ {0, 0.005, 0.010, 0.015}` (4 variants); negative `s` lifts
   ConvNeXt on high-occlusion (proxy decile > 5); positive `s` does the
   opposite (sanity check direction).
   - `s=0` reproduces baseline; graceful fallback by construction.
4. Run renormalized per-image w through `ensemble_v2.py` →
   `apply_biogeo_mask.py`.
5. Score on internal proxies; pick best by mechanism direction
   (occlusion-correlated improvement, not blanket improvement).
6. Budget: ≤ 2 LB submissions if proxies clear.

## Phase 3 — ViT backbone re-finetune with Mixup + CutMix (gated, GPU-heavy, ≤ 5 LB)

**Triggered only if Phases 1+2 hit a ceiling — i.e. mechanism shows
direction but the post-hoc levers cap the achievable lift.**

**Mechanism (one sentence):** *Re-finetune the ViT-B + LoRA + ArcFace
branch with Mixup α=0.8 + CutMix α=1.0 (matching the ConvNeXt training
recipe) so the high-leverage 0.77-weight branch becomes occlusion-robust
at the backbone level, not just the aggregator level.*

**Reuse Exp-13 scaffolding:**

- `Exp-13_ViT_finetuning/configs/phase2_*.yaml` — already specifies
  `mixup: 0.8, cutmix: 1.0`. Hijack the existing config files.
- `Exp-13_ViT_finetuning/scripts/train_phase2.py` — training script
  already exists per the survey.
- Output: a re-finetuned ViT-B that replaces the production ArcFace branch.

**Cost estimate:** ~30 GPU-h on H100 / RTX-6000 for full Phase 2 finetune
(Exp-13 plan estimates).

**Risk:** OOD risk for the train-test-match invariant. If ViT regresses
on the in-distribution lush alpine cases (which the current train recipe
nails) more than it gains on occluded ones, net F1 drops. Mitigation:
weight the Mixup/CutMix component lower than ConvNeXt's recipe (α=0.4
instead of 0.8) so the in-distribution behavior is partially preserved.

## Anti-LB-overfit guardrails

1. Internal proxies primary; LB confirms only.
2. β=0 / s=0 / α-baseline graceful fallbacks must reproduce production
   ≤ 0.001 numerical tolerance.
3. Mechanism in one sentence per phase (above).
4. Stop rule per phase: Δ < 0.003 → neutral, do not iterate.
5. Coarse parameter precision: β ∈ {0, 0.25, 0.5, 0.75, 1.0} grain, not
   {0.18, 0.22, 0.26, 0.30}.
6. New proxy `multi_species_recovery` must align with the mechanism
   prediction (small species on high-density quadrats), not just total
   F1 lift.

## Files

```
Exp-16_Multi_Species_Discovery/
├── docs/
│   ├── Exp16_Plan.md                      (this file)
│   └── Exp16_Phase1_Results.md            (after Phase 1)
├── scripts/
│   ├── compute_vote_share.py              (Phase 1: per-image vote_share[7806])
│   ├── apply_vote_rerank.py               (Phase 1: rerank prefusion + renormalize)
│   └── compute_occlusion_proxy.py         (Phase 2)
├── diagnostics/
│   └── multi_species_recovery.py          (NEW proxy: high-density discovery delta)
├── outputs/
│   ├── phase1_vote_rerank/
│   └── phase2_per_image_w/
└── slurm/
    ├── slurm_phase1_pace.sbatch           (PACE re-run if local Phase 1 wins)
    └── slurm_phase3_finetune.sbatch       (Phase 3 only)
```

## Files reused (no modification)

- `Exp-14/pace_downloads/scale_{4x4,5x5,6x6}.parquet` — DINOv2 CLS tile parquets
  (Phase 1 vote-share computation)
- `Exp-14/pace_downloads/knn_bM0.85_bS1.00_T1.50_prefusion.parquet` — production
  ViT prefusion (Phase 1 input)
- `Exp-14/scripts/ensemble_v2.py` — ViT × ConvNeXt blend
- `Exp-14/scripts/apply_biogeo_mask.py` — Stage 13 mask application
- `Exp-14/diagnostics/proxy_metrics.py` — RNNB / stable-site / density proxies
- `Exp-13_ViT_finetuning/` — Phase 3 only

## Out of scope

- ConvNeXt-symmetric vote-share rerank (deferred; ViT-only first per user direction)
- Per-source vote rerank (only-on-RNNB, only-on-alpine) — same parameter-fishing
  failure mode as Exp-14 Stage 1 global sim_w
- Tuning K past {5, 10}, β past 5-point grain
- Hierarchical taxonomy heads (L5 — different mechanism, future experiment)
