# Exp-7: Multi-Scale Tiling — Results Summary

## Executive Summary

Multi-scale tiling improves F1 from **0.35698 → 0.37894** (+0.02196, +6.2% relative) by adding a 5x5 fine-grained scale alongside the existing 4x4 grid. The optimal configuration is **Scheme D (4x4+5x5, 41 tiles)** at **threshold=0.075**.

Key findings:
- **Fine scales help, coarse scales hurt.** The 5x5 scale captures small/cryptic species at near-native resolution that 4x4 misses. Coarse scales (1x1, 2x2) introduce false positives.
- **4x4 and 5x5 are complementary, not redundant.** Neither scale alone matches the combination — 5x5 alone (0.354) is worse than the 4x4 baseline (0.357).
- **Threshold must increase with more tiles.** Single-scale 4x4 peaked at threshold=0.04; multi-scale D peaked at 0.075. More tiles push max probabilities higher.
- Testing of 6x6 scale (4x4+5x5+6x6, Scheme E) is pending.

## Background

### Motivation

The 1st-place PlantCLEF 2025 solution demonstrated that multi-scale tiling was their single largest performance driver (private F1=0.35013 with 91 tiles across scales 1–6). Our pipeline used only a single 4x4 grid (16 tiles). Different plant species are optimally detected at different spatial scales, and with `max_prob` aggregation, adding scales is additive — if any scale confidently detects a species, it bubbles up.

### Training-Test Domain Gap

The DINOv2 model was trained on single-species close-up images (~800x800px). Test images are multi-species vegetation quadrats (~3000x3000px). Finer tiling scales produce tiles that better match the training distribution:

| Scale | Tiles | Area per tile (px) | Resize to 518 | Match to training |
|-------|-------|--------------------|---------------|-------------------|
| 1x1 | 1 | 3000x3000 | 5.8x downscale | Poor (whole scene) |
| 2x2 | 4 | 1500x1500 | 2.9x downscale | Poor (quarter scene) |
| 4x4 | 16 | 750x750 | 1.4x downscale | Good (plant-level) |
| 5x5 | 25 | 600x600 | 1.2x downscale | Better (close-up) |
| 6x6 | 36 | 500x500 | ~1:1 | Best match (but slight upscale for ~3000px images) |

### Fixed Parameters (all runs)

```
Aggregation:       max_prob
Temperature:       1.5
Temporal Fusion:   prior_blend, alpha=0.6
TopK:              10
Geo Mask:          sw_europe_mask_bool.npy
Aug Filter:        True
Image TopK:        64
tile_filter_k:     ALL (disabled)
JPEG:              quality=85, subsampling=4:2:2, Lanczos
```

## Phase 0: Embedding Extraction

Extracted JPEG85 CLS embeddings for scales 1x1, 2x2, and 5x5 using `extract_multiscale_embeddings.py`. Existing 4x4 JPEG85 embeddings reused from Exp-4.

| Scale | Tiles/image | Total inferences | GPU time |
|-------|-------------|------------------|----------|
| 1x1 | 1 | 2,105 | ~5 min |
| 2x2 | 4 | 8,420 | ~20 min |
| 5x5 | 25 | 52,625 | ~2 hrs |
| 6x6 | 36 | 75,780 | ~3 hrs (pending) |

## Phase 1: Scheme Comparison

### Schemes Tested

| Scheme | Composition | Total Tiles | Rationale |
|--------|-------------|-------------|-----------|
| A | 4x4 (baseline) | 16 | Single-scale reference |
| B | 4x4 + 2x2 + 1x1 | 21 | Add coarse context |
| C | 4x4 + 5x5 + 2x2 + 1x1 | 46 | All scales |
| D | 4x4 + 5x5 | 41 | Fine scale only (no coarse noise) |
| G | 5x5 only | 25 | Isolate fine scale contribution |

### Results: Scheme B (4x4 + 2x2 + 1x1, 21 tiles)

Coarse scales only — does adding whole-image and quadrant-level context help?

| Threshold | F1 | Delta vs baseline (0.357) |
|-----------|------|---------------------------|
| 0.04 | 0.35460 | -0.00238 |
| 0.06 | 0.34805 | -0.00893 |
| 0.08 | 0.34874 | -0.00824 |
| 0.10 | 0.33932 | -0.01766 |

**Verdict: Coarse scales HURT at every threshold.** The 1x1 and 2x2 tiles are too zoomed out to identify individual plants and produce confident false positives that max_prob picks up.

### Results: Scheme C (4x4 + 5x5 + 2x2 + 1x1, 46 tiles)

All scales combined — does the 5x5 signal overcome coarse noise?

| Threshold | F1 | Delta vs baseline |
|-----------|------|-------------------|
| 0.04 | 0.34652 | -0.01046 |
| 0.06 | 0.36462 | +0.00764 |
| **0.08** | **0.37596** | **+0.01898** |
| 0.10 | 0.37300 | +0.01602 |
| 0.15 | 0.32701 | -0.02997 |
| 0.20 | 0.30641 | -0.05057 |

**Verdict: YES.** The 5x5 fine scale provides enough true positives to overcome coarse scale noise. But the coarse scales are still dragging the score down (see Scheme D below).

### Results: Scheme D (4x4 + 5x5, 41 tiles)

Fine scale only — does dropping coarse scales improve further?

| Threshold | F1 | Delta vs baseline |
|-----------|------|-------------------|
| 0.04 | 0.34795 | -0.00903 |
| 0.06 | 0.36783 | +0.01085 |
| **0.075** | **0.37894** | **+0.02196** |
| 0.08 | 0.37617 | +0.01919 |
| 0.085 | 0.37674 | +0.01976 |
| 0.09 | 0.37383 | +0.01685 |
| 0.10 | 0.37103 | +0.01405 |

**Verdict: Best scheme so far.** Dropping the noisy 1x1 and 2x2 improves over Scheme C at the optimal threshold (+0.00298 at thr=0.075 vs C's best of 0.37596 at thr=0.08). The fine threshold sweep confirms peak at **threshold=0.075**.

### Results: Scheme G (5x5 only, 25 tiles)

Is 5x5 alone sufficient, or is 4x4 needed?

| Threshold | F1 | Delta vs baseline |
|-----------|------|-------------------|
| 0.04 | 0.35216 | -0.00482 |
| 0.06 | 0.34604 | -0.01094 |
| 0.075 | 0.35447 | -0.00251 |
| 0.08 | 0.35177 | -0.00521 |

**Verdict: 5x5 alone is WORSE than the 4x4 baseline.** The two scales are genuinely complementary — 4x4 captures medium-sized plants better while 5x5 captures fine detail. Neither is sufficient alone. `max_prob` picks the best from each.

## Key Insights

### 1. Scale Complementarity

The combination of 4x4 + 5x5 works because they capture different plant sizes:
- **4x4 tiles** (750x750px → 518x518): Good for medium-sized plants, some scene context
- **5x5 tiles** (600x600px → 518x518): Better for small/cryptic species at near-native resolution

With `max_prob`, each species is detected at whichever scale sees it most clearly. This is genuinely additive information, not redundancy.

### 2. Coarse Scales Are Harmful

Scales 1x1 and 2x2 produce tiles covering 3000x3000 and 1500x1500 pixels respectively, downscaled to 518x518. At this compression ratio, individual plants are too small to identify. The model produces confident but incorrect species predictions that max_prob propagates to the image level.

### 3. Threshold Shifts with Tile Count

| Scheme | Tiles | Optimal threshold |
|--------|-------|-------------------|
| A (4x4) | 16 | 0.04 |
| D (4x4+5x5) | 41 | 0.075 |
| C (4x4+5x5+2x2+1) | 46 | 0.08 |

More tiles means more chances for any tile to produce a high `max_prob` for a given species. The threshold must rise to maintain precision. This relationship is critical — using the wrong threshold can make a better scheme appear worse (e.g., Scheme D at thr=0.04 scores 0.348, worse than baseline).

## Evolution of Best Configuration

| Step | Configuration | F1 | Delta |
|------|--------------|------|-------|
| Exp-3 baseline | T=2.0, k=6, thr=0.03 | 0.28093 | — |
| Exp-4 max_prob | max_prob, T=1.5, k=4 | 0.31950 | +0.03857 |
| + temporal fusion | alpha=0.6 | 0.33946 | +0.01996 |
| + JPEG85 | quality=85, 4:2:2 | 0.35517 | +0.01571 |
| + threshold tuning | thr=0.04 | 0.35698 | +0.00181 |
| **+ multi-scale (Exp-7)** | **4x4+5x5, thr=0.075** | **0.37894** | **+0.02196** |

**Total improvement: +0.09801** (+34.9% relative over Exp-3 baseline)

## Pending: Scheme E (4x4 + 5x5 + 6x6)

6x6 extraction is in progress. The hypothesis: 6x6 tiles (500x500px, near 1:1 with native resolution) may capture the finest detail and better match the single-species training images. Testing as Scheme E (4x4+5x5+6x6, 77 tiles).

Note: for ~3000x3000 source images, 6x6 requires slight upscaling (3000→3108, +3.6%). The winner's paper confirmed 6x6 helped in their pipeline but scales beyond 6 hurt.

## Current Best Configuration

```
Preprocessing:     JPEG compression (quality=85, subsampling=4:2:2, Lanczos)
Tiling:            Multi-scale: 4x4 + 5x5 (41 tiles/image)
View Strategy:     1-view (orig_only, no TTA)
Aggregation:       max_prob
Temperature:       1.5
tile_filter_k:     ALL (disabled)
Temporal Fusion:   prior_blend, alpha=0.6
Threshold:         0.075
TopK:              10
Geo Mask:          sw_europe_mask_bool.npy
Aug Filter:        True
Image TopK:        64

F1 Score:          0.37894 (public leaderboard)
Improvement:       +0.02196 over Exp-4 best (0.35698)
                   +0.09801 over Exp-3 baseline (0.28093)
Relative Gain:     +34.9% over Exp-3 baseline
```
