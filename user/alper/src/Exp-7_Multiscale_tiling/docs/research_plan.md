# Exp-7: Multi-Scale Tiling — Research Plan

## Motivation

Exp-4 optimized tile aggregation within a single-scale 4x4 grid, reaching F1=0.35698 through max_prob aggregation, T=1.5, JPEG85 preprocessing, temporal fusion (alpha=0.6), and threshold tuning (0.04). However, all predictions come from one spatial scale — each tile covers ~750x750 pixels of the original ~3000x3000 image.

The 1st-place PlantCLEF 2025 solution (Espitalier, "Preprocessing Is All You Need") demonstrated that **multi-scale tiling was their single largest performance driver**, achieving private F1=0.35013 with 91 tiles across scales 1–6 — without temporal fusion, temperature tuning, or tile filtering.

Different plant species are optimally detected at different spatial scales:
- **Coarse scales** (1x1, 2x2): dominant ground cover, scene-level context
- **Medium scales** (4x4): individual plants — our current sweet spot
- **Fine scales** (5x5): small/cryptic species at near-native resolution

With `max_prob` aggregation, adding scales is additive — if any scale confidently detects a species, it bubbles up. The multi-scale signal is **orthogonal** to our existing temporal fusion, JPEG85, and temperature improvements, so gains should stack.

## Background

### Current Best Pipeline (Exp-4 Final)

```
Image → 4x4 tiling → 16 tiles
                        ↓
              JPEG85 recompression (Q=85, 4:2:2, Lanczos)
                        ↓
              DINOv2 embedding extraction per tile
                        ↓
              Tile filtering (top-k=3 by embedding similarity)
                        ↓
              max_prob aggregation (T=1.5)
                        ↓
              Temporal fusion (prior_blend, alpha=0.6)
                        ↓
              Geo-masking (sw_europe_mask_bool.npy)
                        ↓
              Threshold=0.04, topk=10 → submission CSV
```

**F1 = 0.35698** (public leaderboard)

### Winner's Multi-Scale Results (PlantCLEF 2025, private F1)

| Scheme | Tiles | Private F1 | Delta vs 1-tile |
|--------|-------|------------|-----------------|
| 1x1 | 1 | 0.23268 | — |
| 3x3 + 1 | 10 | 0.30523 | +0.073 |
| 6x6 + 3x3 + 1 | 46 | 0.33876 | +0.106 |
| **6x6+5x5+4x4+3x3+2x2+1** | **91** | **0.35013** | **+0.117** |
| 8x8+6x6+5x5+4x4+3x3+2x2+1 | 155 | 0.34426 | +0.112 (worse) |

Key observation: scales beyond 6 **hurt**. Diminishing returns after scale 4-5.

### Differentiation from Winner

Our approach differs in several important ways:

1. **Geometric scale spacing** (1, 2, 4, 5) instead of all-integer scales (1–6). Adjacent scales (3 vs 4, 4 vs 5) provide largely redundant information. Geometric spacing maximizes complementary signal.
2. **No upscaling**: scale 5 (2590x2590) keeps images downscaled from ~3000x3000. Scale 6 (3108x3108) requires upscaling — inventing pixels that don't exist.
3. **Combined with temporal fusion, T=1.5, and tile filtering** — all absent from the winner's pipeline.
4. **JPEG85 applied uniformly** — winner found +0.034 private F1 delta from JPEG compression on 91-tile scheme.

### JPEG Compression Considerations

The benefit of JPEG85 recompression varies by scale:
- **Scale 1** (3000→518, heavy downscaling): +0.112 F1 delta — massive
- **Scale 4** (3000→2072, moderate downscaling): significant benefit — already validated
- **Scale 5** (3000→2590, light downscaling): expected moderate benefit
- **Scale 6** (3000→3108, slight upscaling): smallest benefit, test images already have Q80–98 artifacts

Apply JPEG85 uniformly to all scales initially. Per-scale JPEG tuning is a Phase 4 optimization.

## Experimental Design

### Tiling Schemes

Both schemes add complementary scales on top of our existing 4x4:

| Scheme | Composition | Total Tiles | Compute vs baseline |
|--------|-------------|-------------|---------------------|
| Baseline | 4x4 | 16 | 1.0x |
| **B** | **4x4 + 2x2 + 1x1** | **21** | **1.3x** |
| **C** | **4x4 + 5x5 + 2x2 + 1x1** | **46** | **2.9x** |

Scheme B tests whether coarse context (whole-image + quadrant-level) adds signal on top of 4x4. Scheme C adds the fine scale (5x5) to determine whether near-native resolution tiles capture species missed at coarser scales. Comparing B vs C isolates the value of the 5x5 scale.

### Per-tile resolution at each scale

For a 3000x3000 source image:

| Scale | Grid | Tiles | Source area per tile | Resize to 518x518 | Scaling type |
|-------|------|-------|---------------------|-------------------|-------------|
| 1x1 | 1 | 1 | 3000x3000 | 5.8x downscale | heavy compression |
| 2x2 | 2 | 4 | 1500x1500 | 2.9x downscale | moderate compression |
| 4x4 | 4 | 16 | 750x750 | 1.4x downscale | light compression |
| 5x5 | 5 | 25 | 600x600 | 1.2x downscale | minimal compression |

All scales involve downscaling — no pixel invention.

---

## Phase 0: Embedding Extraction (one-time GPU cost)

Extract JPEG85 + Lanczos CLS embeddings for each new scale. Already have scale 4 embeddings.

| Scale | New tiles per image | Total new inferences (2105 images) | Estimated GPU time |
|-------|--------------------|------------------------------------|-------------------|
| 1x1 | 1 | 2,105 | ~5 min |
| 2x2 | 4 | 8,420 | ~20 min |
| 5x5 | 25 | 52,625 | ~2 hrs |
| **Total** | **30** | **63,150** | **~2.5 hrs** |

Current 4x4 extraction: 16 tiles x 2105 = 33,680 inferences. New scales add ~1.9x compute.

**Implementation notes:**
- Reuse the JPEG85 extraction pipeline from Exp-4 (`slurm_extract_test_jpeg85.sbatch`)
- For each scale N: resize source image to N*518 x N*518, crop to square, cut into N*N tiles of 518x518
- Apply JPEG recompression (Q=85, 4:2:2, Lanczos) to the resized image before cutting
- Store CLS embeddings per tile in same format as existing 4x4 embeddings

---

## Phase 1: Scheme x Threshold Sweep (12 submissions)

**Goal**: Determine if multi-scale beats single-scale, whether the 5x5 fine scale adds value, and find the optimal threshold region.

**Fixed parameters for all runs:**
- Aggregation: `max_prob`
- Temperature: `1.5`
- Temporal fusion: `prior_blend`, `alpha=0.6`
- TopK: `10`
- Geo mask: `sw_europe_mask_bool.npy`
- Aug filter: `True`
- Image topk: `64`
- **tile_filter_k: ALL (disabled)** — k=3 would discard 86–93% of tiles in multi-scale schemes

| Run | Scheme | Tiles | Threshold | Rationale |
|-----|--------|-------|-----------|-----------|
| 1 | 4x4 (baseline) | 16 | 0.04 | **Baseline with k=ALL** (compare to k=3 baseline of 0.357) |
| 2 | B: 4x4+2x2+1 | 21 | 0.04 | Multi-scale B, same threshold — isolate scheme effect |
| 3 | B: 4x4+2x2+1 | 21 | 0.06 | Slightly higher threshold |
| 4 | B: 4x4+2x2+1 | 21 | 0.08 | Higher threshold region |
| 5 | B: 4x4+2x2+1 | 21 | 0.10 | Moderate threshold |
| 6 | B: 4x4+2x2+1 | 21 | 0.15 | High threshold |
| 7 | C: 4x4+5x5+2x2+1 | 46 | 0.04 | Full scheme, same threshold as baseline |
| 8 | C: 4x4+5x5+2x2+1 | 46 | 0.06 | Slightly higher threshold |
| 9 | C: 4x4+5x5+2x2+1 | 46 | 0.08 | Higher threshold region |
| 10 | C: 4x4+5x5+2x2+1 | 46 | 0.10 | Moderate threshold |
| 11 | C: 4x4+5x5+2x2+1 | 46 | 0.15 | High threshold |
| 12 | C: 4x4+5x5+2x2+1 | 46 | 0.20 | Aggressive threshold |

**Decision logic after Phase 1:**
- Compare best-of-B vs best-of-C: isolates the value of the 5x5 scale
- If best multi-scale beats baseline: proceed to Phase 2 (fine threshold sweep on winner)
- If neither B nor C beats baseline: multi-scale does not help — stop experiment
- Run 1 also reveals whether k=ALL vs k=3 matters for single-scale (free information)
- Scheme C gets one extra threshold point (0.20) since more tiles may push the optimum higher

---

## Phase 2: Fine Threshold Sweep (5 submissions)

**Goal**: Nail the optimal threshold for the best scheme from Phase 1.

Suppose Phase 1 finds scheme C peaking around threshold 0.10:

| Run | Scheme | Threshold |
|-----|--------|-----------|
| 11 | best | peak - 0.03 |
| 12 | best | peak - 0.015 |
| 13 | best | peak (confirm) |
| 14 | best | peak + 0.015 |
| 15 | best | peak + 0.03 |

Exact values depend on Phase 1 results. Target: pin threshold within ±0.01.

---

## Phase 3: tile_filter_k Reintroduction (5 submissions)

**Goal**: Test whether selective tile filtering helps on top of multi-scale.

Phase 1 disables filtering (k=ALL). But filtering might still help by removing noisy tiles — especially from coarser scales where a single tile covers the whole image and may produce false positives.

| Run | Scheme | k | Effective tiles used | Rationale |
|-----|--------|---|---------------------|-----------|
| 16 | best | ALL | all | Already have from Phase 2 |
| 17 | best | 5 | 5 of 21/46 | Aggressive global filter |
| 18 | best | 10 | 10 of 21/46 | Moderate filter |
| 19 | best | 15 | 15 of 21/46 | Light filter |
| 20 | best | per-scale-top-2 | 2 per scale = 8/10 | Scale-aware: keep top-2 from each scale |

**Note on per-scale filtering:** Instead of a global top-k, select the top-N tiles per scale. This ensures every scale contributes, preventing fine or coarse scales from being completely filtered out.

---

## Phase 4: Optional Refinements (if Phases 1-3 show gains)

### 4a. Alpha re-tuning
Temporal fusion alpha=0.6 was optimized for single-scale 4x4. Multi-scale may change the balance.

| Run | alpha | Notes |
|-----|-------|-------|
| 21 | 0.5 | Less temporal, more photo |
| 22 | 0.7 | More temporal |

### 4b. Temperature re-tuning
T=1.5 was optimal for single-scale. More tiles may benefit from sharper or softer predictions.

| Run | T | Notes |
|-----|---|-------|
| 23 | 1.0 | Sharper per-tile predictions |
| 24 | 2.0 | Softer, more spread |

### 4c. Per-scale JPEG quality (low priority)
Test whether fine scales (5x5) benefit less from JPEG85 recompression.

| Run | JPEG config | Notes |
|-----|-------------|-------|
| 25 | Scale 5: no JPEG, rest: JPEG85 | Fine tiles already near-native quality |
| 26 | Scale 1: Q75, rest: Q85 | Heavier compression for heavy downscale |

---

## Submission Budget Summary

| Phase | Submissions | Cumulative | What you learn |
|-------|-------------|------------|---------------|
| 0 | 0 | 0 | Embeddings extracted |
| 1 | 12 | 12 | B vs C vs baseline + threshold region |
| 2 | 5 | 17 | Optimal threshold (±0.01) |
| 3 | 5 | 22 | Whether tile filtering helps |
| 4 | 6 | 28 | Fine-tuning (optional) |

**Phase 1 is the only critical phase.** If it shows no gain, stop at 12 submissions. Phases 2-4 are refinement.

---

## Expected Outcomes

| Outcome | Estimated probability | Action |
|---------|----------------------|--------|
| Multi-scale gives +0.01 to +0.025 F1 | ~50% | Proceed through Phases 2-4 |
| Multi-scale gives +0.005 to +0.01 F1 | ~25% | Threshold/k tuning may recover more |
| Multi-scale gives ~0 or negative | ~25% | Stop. Move to Exp-6 (adaptive thresholding) |

**Best case target**: F1 = 0.370–0.382 (current 0.357 + 0.01–0.025)

---

## Current Winning Configuration (Pre-Experiment Baseline)

```
Preprocessing:     JPEG compression (quality=85, subsampling=4:2:2, Lanczos)
Tiling:            4x4 grid (single scale, 16 tiles)
View Strategy:     1-view (orig_only, no TTA)
Aggregation:       max_prob
Temperature:       1.5
tile_filter_k:     3
Temporal Fusion:   prior_blend, alpha=0.6
Threshold:         0.04
TopK:              10
Geo Mask:          sw_europe_mask_bool.npy
Aug Filter:        True
Image TopK:        64

F1 Score:          0.35698 (public leaderboard)
```
