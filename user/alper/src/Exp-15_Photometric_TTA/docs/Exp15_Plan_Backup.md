# Exp-15 — Photometric Test-Time Augmentation for Low-Contrast Vegetation

## Context

The Exp-14 winner (F1 = 0.45572 public LB) leaves a residual error mode the user has flagged from manual inspection: when plants are brownish or dormant and visually similar in tone to the substrate (sand, dry soil), the model struggles to distinguish them. Photometric preprocessing (sharpening, contrast enhancement, saturation boost) might help by amplifying plant–substrate edges and faint chromatic differences before the backbone sees the tile.

**Critical asymmetry between the two backbones (verified from training scripts):**

- **ViT-B + LoRA + ArcFace (Exp-9B/train_arcface_lora.py:58–75)** trained with `[Resize, CenterCrop, JPEG85, ToTensor, Normalize]` — **zero photometric augmentation by deliberate design** ("matching test distribution at train time is more important than aggressive aug"). Test pipeline matches exactly. Wholesale replacement of test preprocessing pushes ViT OOD; high regression risk.
- **ConvNeXt-Large (Exp-10/configs/phase2_large.yaml)** trained with timm's `rand-m9-mstd0.5-inc1` RandAugment + Mixup 0.8 + CutMix 1.0 — RandAugment includes color jitter, contrast, brightness, sharpness. Has seen photometric variation.

**Leverage observation that drives the design:** ConvNeXt has only w = 0.23 ensemble weight. Even a generous +0.010 standalone F1 lift on ConvNeXt would dilute to ≈ +0.0023 in the final ensemble (within 11 % LB noise floor of ±0.005). ViT carries w = 0.77 → 3.3× higher leverage per F1 point of standalone improvement. The experiment must therefore find a way to safely apply photometric preprocessing to the ViT branch despite its training distribution containing none.

**The mechanism: photometric TTA via probability-level blend on the ViT branch.** Instead of replacing the ViT tile preprocessing, run the ViT backbone twice — once on the original tile pipeline (in-distribution), once on the photometrically enhanced tile pipeline (out-of-distribution but potentially complementary on brownish cases) — and average the post-aggregation per-image probability vectors before feeding into the kNN + temporal fusion + ensemble blend. The in-distribution prediction remains load-bearing; the preprocessed-prediction adds complementary information on cases where the original signal is weak.

This is the photometric analog of test-time augmentation but applied at the post-aggregation per-image probability level (where the ensemble blends), not the per-tile pre-max-prob level (where Exp-4 found multi-view TTA *hurt* with max_prob aggregation).

**Goal:** decide cheap whether photometric TTA on the ViT branch can lift F1 above 0.45572. Target +0.003–0.010 if the mechanism works; null if it doesn't.

## Anti-overfit principles (binding, from Exp-14 lessons)

1. **Internal proxies primary; LB confirms only.** Public LB is 11 % (~232 images), noise floor ≈ ±0.005. The Exp-14 lesson: variants under that band are statistically indistinguishable; do not over-iterate.
2. **Cheap-first staging.** Phase 0 is CPU-only triage; Phases 1+ commit GPU only after positive earlier-phase signal.
3. **Mechanism explicable in one sentence.** *"Average ViT predictions on original tiles with ViT predictions on mildly-sharpened tiles, post-aggregation, to add complementary information on brownish/low-contrast images."*
4. **Negative results report.** A null result on Phase 1 closes the experiment with a documented finding; we don't tune photometric parameters until something positive emerges.

## Phase plan

All paths absolute. New experiment root: `user/alper/src/Exp-15_Photometric_TTA/`.

The experiment is fully data-driven — no manual image review. Phase 1 is the entry point.

### Phase 1 — ViT probe across all 3 variants (PACE, ~1–2 GPU-h, no LB)

**Goal:** does any of the three photometric variants change ViT predictions in a plant-favorable direction without catastrophically pushing the model OOD?

**1a. Auto-select probe sample** (`scripts/build_probe_sample.py`):
- Load `Exp-14_Adaptive_Temporal_Fusion/outputs/signals/density.parquet`.
- Stratified sample of 40 quadrats:
  - 15 from the brownish/low-contrast band (`0.05 ≤ density < 0.15`)
  - 10 from the lush band (`density ≥ 0.40`) to confirm preprocessing does *not* break working cases
  - 10 from the very-barren band (`density < 0.05`) to test whether preprocessing accidentally amplifies substrate noise
  - 5 user-flagged hard cases (the RNNB pairs from Exp-14 manual inspection)
- Output: `outputs/probe/probe_sample.csv`.

**1b. Implement 3 photometric variants** (`preprocess/photometric.py`): each exposes a `transform(img: PIL.Image) -> PIL.Image` callable for reuse in Phase 2.

| Variant | Implementation |
|---|---|
| `unsharp` | `PIL.ImageFilter.UnsharpMask(radius=2.0, percent=120, threshold=3)` |
| `clahe_lab` | cv2 `cvtColor(BGR2LAB)` → `createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` on L → merge → `cvtColor(LAB2BGR)`. Fallback if cv2 unavailable: `PIL.ImageOps.equalize` per-channel. |
| `saturation` | `PIL.ImageEnhance.Color(img).enhance(1.4)` ∘ `PIL.ImageEnhance.Contrast(img2).enhance(1.15)` |

Insertion point in the preprocessing chain: after center-crop-square + resize-to-518, before JPEG85 + ToTensor (mirroring how RandAugment operates on ConvNeXt training inputs).

**1c. Probe script** (`scripts/probe_vit_variants.py`, mirroring `Exp-9B/extract_embeddings_lora_test.py`):
- For each of 40 probe quadrats, tile at scales 4, 5, 6 (production scales).
- For each tile, run ViT-B + LoRA forward pass FOUR times: once with the original preprocessing chain, and once each with `[unsharp, clahe_lab, saturation]` injected before JPEG85.
- Save per-tile teacher_logits for all 4 branches → `outputs/phase1/probe_logits.parquet`.

**1d. Phase-1 proxy metrics** (`scripts/score_phase1.py`), computed per variant V vs the original:
- **`mean_top1_conf_shift`** — mean over (image, tile) of `max_softmax(logits_V) − max_softmax(logits_orig)`. Computed separately for the brownish-band (15 quadrats), lush-band (10), and very-barren-band (10) sub-cohorts. Want positive on brownish, neutral-or-positive on lush, neutral-or-negative on very-barren (preprocessing should not invent confidence on bare sand).
- **`top1_stability`** — fraction of (image, tile) where argmax matches between V and original. Low values = catastrophic OOD; we want ≥ 0.55.
- **`posidonia_logit_shift`** — mean shift in Posidonia (class_idx looked up from `class_to_species_id.json`) softmax probability across all tiles. Negative = preprocessing siphons mass away from the most-known substrate-confusion species.
- **`aquatic_mass_shift`** — mean shift in total softmax mass on the 91 aquatic-blacklist species. Generalization of `posidonia_logit_shift` to the full habitat-impossibility set.
- **`cohort_topk_jaccard`** — per-image (max-aggregated) top-10 Jaccard between V and original. Acceptable range 0.55–0.85 (nudge, not rewrite).
- **`brownish_vs_barren_conf_gap`** — `mean_top1_conf_shift[brownish] − mean_top1_conf_shift[very_barren]`. Mechanism check: preprocessing should help on brownish (real plants) more than on very-barren (no plants). Want > +0.005.

**Phase 1 → Phase 2 gate:** a variant survives if ALL of:
- `top1_stability ≥ 0.55`
- `cohort_topk_jaccard ∈ [0.55, 0.85]`
- `aquatic_mass_shift ≤ 0`
- `brownish_vs_barren_conf_gap > +0.005`
- At least ONE of `mean_top1_conf_shift[brownish] > +0.01` OR `aquatic_mass_shift < −0.005`

If 0 variants pass → close Exp-15 with a negative-result report (`Exp15_Results_Summary.md`).
If 1 variant passes → advance only that one to Phase 2.
If 2+ variants pass → advance all surviving variants to Phase 2 in parallel (extra ~4 GPU-h per variant; user has GPU budget).

The brownish-vs-barren gap is the most important gate: the mechanism only makes sense if preprocessing helps the model on images that *contain* low-contrast plants while NOT helping it hallucinate plants on images that contain none. A variant that uniformly raises confidence everywhere is just sharpening noise, not signal.

### Phase 2 — Full ViT re-extraction with photometric preprocessing (PACE, ~4 GPU-h on RTX-6000, no LB)

**2a. Wrapper script** (`scripts/extract_embeddings_lora_test_photo.py`): clone of `Exp-9B/extract_embeddings_lora_test.py` with a single addition — a `--photometric {none,unsharp,clahe_lab,saturation}` flag that injects `preprocess/photometric.py:transform(...)` between `tile_image` (line ~88) and `apply_jpeg_compression` (line ~92). Original Exp-9B script untouched.

**2b. SLURM job** (`slurm/slurm_extract_vit_photo.sbatch`, copy of `Exp-9B/slurm_extract_lora_test.sbatch`): output to `${PLANTCLEF_ROOT}/aerten3/parquet/exp15_vit_lora_${PHOTO}/scale_{4x4,5x5,6x6}.parquet`. Run for scales 4, 5, 6 (the production scales).

**2c. Sanity check after extraction** (`diagnostics/postextract_sanity.py`):
- Tile count per scale must equal the baseline ViT extraction (33 680 / 52 625 / 75 780 for scales 4/5/6 respectively).
- `max_prob` median should not have shifted by more than ±0.10 vs baseline (sign of distribution collapse if violated).

### Phase 3 — Photometric TTA blend on ViT branch + 1 LB submission

**3a. Blend logic** (`scripts/blend_vit_photo_tta.py`):
- Load the **post-aggregation per-image probability** parquet from each ViT extraction. The Exp-9B pipeline produces `knn_bM0.85_bS1.00_T1.50_prefusion.parquet` after kNN blending and pre-temporal-fusion; we need the analogous file produced by re-running the Exp-9B downstream on the photometric tile parquets.
- Re-run `Exp-9B/multiscale_aggregate_knn.py` (lines 388–402, the prefusion-saving path) on the new photometric tile parquets to produce `knn_bM0.85_bS1.00_T1.50_prefusion_photo.parquet`. This is a CPU-bound step (~10–30 min) — kNN against the existing FAISS index, no model inference.
- Blend at the per-image probability level: `P_blend = β · P_orig + (1−β) · P_photo`. Sweep β ∈ {0.50, 0.65, 0.80} locally (3 candidate parquets, no LB cost).
- Pass each blended ViT parquet through the locked Exp-14 pipeline: `ensemble_v2.py --vit-prefusion-parquet <blend> --convnext-scales <existing>` with the winner controller config + sim_w-on-RNNB + RNNB α=0.30.
- Apply habitat mask via `apply_habitat_mask.py`.

**3b. Pre-LB internal proxy gate** (`diagnostics/proxy_metrics.py` from Exp-14, plus new metric):
- All Exp-14 proxies must pass without regression: `rnnb_pair_divergence ≥ 0`, `stable_site_invariance ≥ −0.02`, `density_count_correlation ≥ 0`, `photo_faithfulness ≥ 0`.
- New Exp-15 metric (`diagnostics/proxy_metrics_photo.py`): `low_density_diff_concentration` = of rows that changed vs the 0.45572 winner, fraction with `density < 0.15`. Required ≥ 0.45 (changes concentrate where the mechanism predicts; they should not be scattered uniformly across CBN-PdlC alpine cases).
- Pick the **single best** β by `density_count_correlation` improvement.

**3c. LB submission #1:** the best-β blend.

### Phase 4 — Bracket and ConvNeXt extension (gated, ≤ 3 LB submissions)

**4a. Bracket the Phase 3 winner (1 LB):** if Phase 3 lifts F1 by ≥ +0.003 vs 0.45572, submit the next-best β as a bracket-confirm. If neither neighbor of the winning β also rises, the lift is suspect.

**4b. ConvNeXt symmetric attempt (1–2 LB, gated):** if Phase 3 wins, repeat the experiment on the ConvNeXt branch with the same chosen photometric variant. ConvNeXt was trained with RandAugment so it should tolerate it natively. Cost: ~8 GPU-h re-extraction. Submit the best ConvNeXt-photo + ViT-photo combo.

**4c. Stop-and-write:** at any point, if proxy gates fail or LB regresses, close the experiment with a negative report.

## Internal proxy metrics (definitive list)

Inherited from Exp-14:
1. `rnnb_pair_divergence` (higher is better)
2. `stable_site_invariance` (must not collapse on CBN-PdlC)
3. `density_count_correlation` (Spearman ρ; **primary mechanism metric**)
4. `photo_faithfulness` (sanity)

New for Exp-15:
5. `low_density_diff_concentration` (Phase 3 gate; rows changed should concentrate on density < 0.15)
6. `posidonia_logit_shift` (Phase 1 gate; ViT softmax probability shift on the most-substrate-confused class)
7. `cohort_topk_jaccard` (Phase 1 gate; nudge-not-rewrite check)

## GPU and LB budget

| Phase | GPU cost (cumulative) | LB submissions | Trigger |
|---|---|---|---|
| 1 (probe, 3 variants × 40 quadrats) | ≤ 2 GPU-h | 0 | unconditional (entry point) |
| 2 (full ViT re-extraction, per surviving variant) | +4 GPU-h × N_surviving | 0 | Phase 1 gates pass for ≥ 1 variant |
| 3 (β sweep + LB) | ~6–10 GPU-h cumulative | up to 7 (β grid) | Phase 2 sanity passes |
| 4a (bracket + cross-variant interactions) | ~6–10 GPU-h | up to 5 | Phase 3 best-β lifts F1 ≥ +0.003 |
| 4b (ConvNeXt symmetric) | +8 GPU-h | up to 5 | Phase 4a confirms direction |

**Worst-case total: ~25 GPU-h (if all 3 variants survive Phase 1 and ConvNeXt symmetric also runs), ~17 LB submissions. Cheap path (negative Phase 1): ~2 GPU-h, 0 LB.**

User has a ceiling of 100 LB submissions/day — effectively unlimited for this experiment. The binding constraint is therefore not budget but the **11 % LB noise floor (≈ ±0.005)**: variants whose F1 differs by less than that band are statistically indistinguishable on the 232-image sample, and iterating to chase sub-noise differences risks private-LB regression. The phase-by-phase design therefore:

- Allows a **denser β sweep in Phase 3** (β ∈ {0.30, 0.45, 0.55, 0.65, 0.75, 0.85, 1.00}, 7 variants — including β=1.00 as parity-check that must reproduce 0.45572 byte-for-byte / ≤ 0.001 tolerance).
- Allows **bracket confirmation in Phase 4a** (3–5 LB to verify the Phase 3 winner is on a real curve, not LB sample artifact — submit β values one grid step on either side of the Phase 3 winner).
- **Still uses internal proxies as Phase 3 entry gate**, because the LB at 232 images cannot reliably distinguish < 0.005 differences and proxies on the full 2105 cohort are higher-resolution evidence.
- **Stops escalating** when the spread of bracket variants is < ±0.003 — this is the noise floor where the LB cannot tell us anything more.

## Risk and failure modes

1. **ViT OOD blowup.** Phase 1 detects this via `top1_stability < 0.55` or `cohort_topk_jaccard < 0.55`. The model loses tile recognition entirely. Mitigation: stop at Phase 1.
2. **Mechanism fires on barren but breaks alpine.** Phase 3 detects this via `stable_site_invariance < −0.02` or `density_count_correlation` decreasing (mechanism inverted). Mitigation: Phase 3 gate kills before LB.
3. **Proxy–LB decoupling (Exp-14 Stage 1 lesson).** Proxies improve, LB drops. Detection: Phase 3 LB. Mitigation: hard cap of 4 LB submissions; the blend with β = 1.0 recovers exactly the 0.45572 winner (graceful fallback by construction).

## Critical files to create

```
user/alper/src/Exp-15_Photometric_TTA/
├── README.md
├── docs/
│   ├── Exp15_Plan.md
│   ├── Exp15_Phase1_Probing_Report.md        (after Phase 1)
│   └── Exp15_Results_Summary.md              (after Phases 3/4)
├── preprocess/
│   ├── __init__.py
│   └── photometric.py                        (3 variants: unsharp, clahe_lab, saturation)
├── diagnostics/
│   ├── postextract_sanity.py                 (Phase 2 sanity)
│   └── proxy_metrics_photo.py                (Phase 3 new metric)
├── scripts/
│   ├── build_probe_sample.py                 (Phase 1a; auto-stratified, no human review)
│   ├── probe_vit_variants.py                 (Phase 1c)
│   ├── score_phase1.py                       (Phase 1d gate)
│   ├── extract_embeddings_lora_test_photo.py (Phase 2 wrapper)
│   ├── blend_vit_photo_tta.py                (Phase 3)
│   └── extract_convnext_photo.py             (Phase 4b, gated)
├── slurm/
│   ├── slurm_probe_phase1.sbatch             (RTX-6000, ≤2h, runs all 3 variants)
│   ├── slurm_extract_vit_photo.sbatch        (RTX-6000, 4h, per surviving variant)
│   └── slurm_extract_cnxt_photo.sbatch       (H200, 8h, Phase 4b gated)
└── outputs/
    ├── probe/                                 (Phase 1 sample CSV + logits parquet)
    ├── phase1/                                (probe scores + chosen variant)
    ├── stage1_vit_photo_tta/                  (Phase 3 outputs + LB CSVs)
    └── stage2_cnxt_photo_tta/                 (Phase 4 outputs)
```

## Files to reuse (no modification)

- `Exp-9B_Improved_KNN_Retrieval/extract_embeddings_lora_test.py` — clone-and-modify for `--photometric` flag
- `Exp-9B_Improved_KNN_Retrieval/multiscale_aggregate_knn.py` — re-runnable on photometric tile parquets to produce blended kNN+aggregation parquet
- `Exp-14_Adaptive_Temporal_Fusion/scripts/ensemble_v2.py` — feed photometric ViT prefusion parquet via `--vit-prefusion-parquet`
- `Exp-14_Adaptive_Temporal_Fusion/scripts/apply_habitat_mask.py` — final post-processing
- `Exp-14_Adaptive_Temporal_Fusion/diagnostics/proxy_metrics.py` — reuse 4 proxies; extend with Phase-3 new metric
- `Exp-14_Adaptive_Temporal_Fusion/outputs/signals/density.parquet` — Phase 0 cohort filter

## Verification

**Phase 1 verification:** `score_phase1.py` outputs JSON with all 6 proxy metrics per variant. JSON exists, all metrics non-NaN, gate decision logged. Surviving variant(s) printed with rationale. The probe sample is auto-stratified by density band (no human selection); reproducibility check: re-running `build_probe_sample.py` with the same density.parquet should produce a deterministic CSV (use a fixed `numpy.random.RandomState(seed)` in the sampling).

**Phase 2 sanity:** `postextract_sanity.py` confirms tile counts match baseline (162 085 total) and `max_prob` median shift < 0.10 vs baseline parquet.

**Phase 3 smoke test:** before LB submission, run `ensemble_v2.py` with `--vit-prefusion-parquet` = baseline (β = 1.0 case) and confirm output matches the 0.45572 winner CSV byte-for-byte (or to the same ≤ 0.001 numerical tolerance as Stage 0 in Exp-14). This confirms the blend infrastructure preserves the baseline when β = 1.0 — the graceful-fallback property.

**Phase 3 LB:** submit best-β CSV to Kaggle; record F1 in `Exp15_Results_Summary.md`.

**End-to-end:** if Phase 3 wins, the diagnostic re-render of the 8 user-flagged RNNB images (using the Exp-14 viz tool) should show the brownish-substrate predictions becoming more sensible — lower Posidonia/dune-confusion mass, higher proportion of plausible halophyte / dormant-grass predictions.

## Out of scope for Exp-15

- ViT replacement (not blend) of the original prefusion parquet — too high OOD risk
- Per-source photometric (e.g., apply only to RNNB images) — adds parameter axis without clear mechanism support
- Geometric TTA (Exp-4 already documented this hurts with max_prob)
- Multi-head taxonomic, SAM masking, EIVE — separate, larger experiments
- Tuning photometric parameters (radius, percent, clipLimit, etc.) past Phase 0 defaults — would be LB-overfit territory
