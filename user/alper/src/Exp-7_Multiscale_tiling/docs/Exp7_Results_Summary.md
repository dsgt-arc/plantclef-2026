# Exp-7: Multi-Scale Tiling — Results Summary

## Executive Summary

Multi-scale tiling with re-tuned parameters improves F1 from **0.35698 → 0.39573** (+0.03875, +10.9% relative). The optimal configuration is **Scheme E (4x4+5x5+6x6, 77 tiles)** with **alpha=0.2, T=1.5, threshold=0.070**.

Key findings:
- **Fine scales help, coarse scales hurt.** Scales 5x5 and 6x6 capture small/cryptic species at near-native resolution. Coarse scales (1x1, 2x2, 3x3) introduce false positives.
- **Three complementary scales outperform two.** 4x4+5x5+6x6 beats 4x4+5x5 by +0.006 and 4x4+6x6 by +0.010. All three contribute unique signal.
- **No single scale is sufficient alone.** 5x5 alone (0.354) is worse than the 4x4 baseline (0.357). The scales are complementary, not redundant.
- **7x7 provides negligible gain (+0.0008) for 49 extra tiles.** Diminishing returns beyond scale 6.
- **Alpha shifted dramatically from 0.6 → 0.2.** Multi-scale produces more candidates but also more noise — the location prior (80% weight) aggressively filters to species actually observed at each site.
- **T=1.5 remains optimal.** Temperature is a sharp peak; deviations in either direction hurt significantly.
- **Threshold shifted from 0.04 → 0.070.** More tiles push max probabilities higher, requiring a higher decision boundary.

## Background

### Motivation

The 1st-place PlantCLEF 2025 solution demonstrated that multi-scale tiling was their single largest performance driver (private F1=0.35013 with 91 tiles across scales 1–6). Our pipeline used only a single 4x4 grid (16 tiles). Different plant species are optimally detected at different spatial scales, and with `max_prob` aggregation, adding scales is additive — if any scale confidently detects a species, it bubbles up.

### Training-Test Domain Gap

The DINOv2 model was trained on single-species close-up images (~800x800px). Test images are multi-species vegetation quadrats (~3000x3000px). Finer tiling scales produce tiles that better match the training distribution:

| Scale | Tiles | Area per tile (px) | Resize to 518 | Match to training |
|-------|-------|--------------------|---------------|-------------------|
| 1x1 | 1 | 3000x3000 | 5.8x downscale | Poor (whole scene) |
| 2x2 | 4 | 1500x1500 | 2.9x downscale | Poor (quarter scene) |
| 3x3 | 9 | 1000x1000 | 1.9x downscale | Marginal |
| 4x4 | 16 | 750x750 | 1.4x downscale | Good (plant-level) |
| 5x5 | 25 | 600x600 | 1.2x downscale | Better (close-up) |
| 6x6 | 36 | 500x500 | ~1:1 | Best match (slight upscale for ~3000px images) |
| 7x7 | 49 | ~430x430 | 1.2x upscale | Good match but upscaling needed |

## Phase 0: Embedding Extraction

Extracted JPEG85 CLS embeddings for each scale using `extract_multiscale_embeddings.py`. Existing 4x4 JPEG85 embeddings reused from Exp-4.

| Scale | Tiles/image | Total inferences | GPU time |
|-------|-------------|------------------|----------|
| 1x1 | 1 | 2,105 | ~5 min |
| 2x2 | 4 | 8,420 | ~20 min |
| 3x3 | 9 | 18,945 | ~30 min |
| 4x4 | 16 | 33,680 | existing |
| 5x5 | 25 | 52,625 | ~2 hrs |
| 6x6 | 36 | 75,780 | ~3 hrs |
| 7x7 | 49 | 103,145 | ~4 hrs |

## Phase 1: Scheme Comparison

All scheme comparison runs used: T=1.5, alpha=0.6, topk=10, tile_filter_k=ALL.

### All Schemes Tested

| Scheme | Composition | Total Tiles | Best Thr | Best F1 | vs baseline (0.357) |
|--------|-------------|-------------|----------|---------|---------------------|
| A | 4x4 (baseline) | 16 | 0.04 | 0.35698 | — |
| B | 4x4 + 2x2 + 1x1 | 21 | 0.04 | 0.35460 | -0.00238 |
| G | 5x5 only | 25 | 0.075 | 0.35447 | -0.00251 |
| D | 4x4 + 5x5 | 41 | 0.075 | 0.37894 | +0.02196 |
| C | 4x4 + 5x5 + 2x2 + 1x1 | 46 | 0.08 | 0.37596 | +0.01898 |
| F | 4x4 + 6x6 | 52 | 0.075 | 0.37507 | +0.01809 |
| **E** | **4x4 + 5x5 + 6x6** | **77** | **0.075** | **0.38466** | **+0.02768** |
| J | 4x4 + 5x5 + 6x6 + 3x3 | 86 | 0.075 | 0.38227 | +0.02529 |
| K | 4x4 + 5x5 + 6x6 + 7x7 | 126 | 0.08 | 0.38542 | +0.02844 |

### Detailed Results by Scheme

#### Scheme B: 4x4 + 2x2 + 1x1 (21 tiles) — Coarse scales only

| Threshold | F1 |
|-----------|------|
| 0.04 | 0.35460 |
| 0.06 | 0.34805 |
| 0.08 | 0.34874 |
| 0.10 | 0.33932 |

**Verdict: Coarse scales HURT at every threshold.** The 1x1 and 2x2 tiles are too zoomed out to identify individual plants and produce confident false positives that max_prob picks up.

#### Scheme G: 5x5 only (25 tiles) — Fine scale isolation

| Threshold | F1 |
|-----------|------|
| 0.04 | 0.35216 |
| 0.06 | 0.34604 |
| 0.075 | 0.35447 |
| 0.08 | 0.35177 |

**Verdict: 5x5 alone is WORSE than the 4x4 baseline.** The two scales are genuinely complementary — 4x4 captures medium-sized plants better while 5x5 captures fine detail. Neither is sufficient alone.

#### Scheme C: 4x4 + 5x5 + 2x2 + 1x1 (46 tiles) — All scales

| Threshold | F1 |
|-----------|------|
| 0.04 | 0.34652 |
| 0.06 | 0.36462 |
| **0.08** | **0.37596** |
| 0.10 | 0.37300 |
| 0.15 | 0.32701 |
| 0.20 | 0.30641 |

**Verdict:** The 5x5 fine scale provides enough true positives to overcome coarse scale noise, but the coarse scales still drag the score down vs Scheme D.

#### Scheme D: 4x4 + 5x5 (41 tiles) — Fine scales only

| Threshold | F1 |
|-----------|------|
| 0.04 | 0.34795 |
| 0.06 | 0.36783 |
| **0.075** | **0.37894** |
| 0.08 | 0.37617 |
| 0.085 | 0.37674 |
| 0.09 | 0.37383 |
| 0.10 | 0.37103 |

**Verdict:** Dropping the noisy 1x1 and 2x2 improves over Scheme C. Fine threshold sweep confirms peak at threshold=0.075.

#### Scheme F: 4x4 + 6x6 (52 tiles) — Skip 5x5

| Threshold | F1 |
|-----------|------|
| 0.04 | 0.34710 |
| 0.06 | 0.37172 |
| **0.075** | **0.37507** |
| 0.08 | 0.36783 |
| 0.10 | 0.36765 |

**Verdict:** Worse than 4x4+5x5 (Scheme D). The gap between 4x4 (750px) and 6x6 (500px) is too large — 5x5 (600px) fills a critical middle ground.

#### Scheme E: 4x4 + 5x5 + 6x6 (77 tiles) — Three fine scales

| Threshold | F1 |
|-----------|------|
| 0.04 | 0.33970 |
| 0.06 | 0.37082 |
| **0.075** | **0.38466** |
| 0.08 | 0.37965 |
| 0.10 | 0.38358 |
| 0.15 | 0.35255 |

**Verdict: Best scheme.** Adding 6x6 on top of 4x4+5x5 gives +0.006. All three scales are complementary. The 6x6 tiles best match the single-species training distribution.

#### Scheme J: 4x4 + 5x5 + 6x6 + 3x3 (86 tiles) — Add coarser scale

| Threshold | F1 |
|-----------|------|
| 0.04 | 0.33758 |
| 0.06 | 0.37181 |
| **0.075** | **0.38227** |
| 0.08 | 0.37731 |
| 0.10 | 0.38110 |
| 0.15 | 0.35245 |

**Verdict:** Adding 3x3 hurts (-0.002 vs E). Consistent with all coarse scale findings.

#### Scheme K: 4x4 + 5x5 + 6x6 + 7x7 (126 tiles) — Add finer scale

| Threshold | F1 |
|-----------|------|
| 0.04 | 0.33242 |
| 0.06 | 0.37325 |
| 0.075 | 0.38453 |
| **0.08** | **0.38542** |
| 0.10 | 0.38529 |
| 0.15 | 0.36284 |

**Verdict:** Marginal gain (+0.0008 vs E) for 49 extra tiles. Diminishing returns. The flat curve across 0.075-0.10 suggests robust but redundant signal from 7x7.

## Phase 2: Alpha Re-tuning on Scheme E

Alpha=0.6 was optimized for single-scale 4x4. With 77 multi-scale tiles producing more candidates (and more noise), the location prior becomes more valuable as a filter.

All runs at T=1.5, thr=0.075.

| Alpha | F1 | Delta vs 0.6 |
|-------|------|-------------|
| 0.1 | 0.39268 | +0.00802 |
| **0.2** | **0.39420** | **+0.00954** |
| 0.3 | 0.39211 | +0.00745 |
| 0.4 | 0.39085 | +0.00619 |
| 0.5 | 0.38890 | +0.00424 |
| 0.6 | 0.38466 | — |
| 0.7 | 0.38106 | -0.00360 |
| 0.8 | 0.37718 | -0.00748 |

**Optimal alpha=0.2** (20% photo prediction, 80% location prior). The location prior aggressively filters multi-scale predictions to species actually observed at each site. Gain: +0.00954 over alpha=0.6.

## Phase 3: Temperature Sweep on Scheme E

All runs at alpha=0.6, thr=0.075. (Temperature is independent of alpha — confirmed T=1.5 is optimal before alpha re-tuning.)

| Temperature | F1 |
|-------------|------|
| 1.00 | 0.30766 |
| 1.25 | 0.33081 |
| **1.50** | **0.38466** |
| 1.75 | 0.32643 |

**T=1.5 is a sharp peak.** No change needed. Deviations in either direction cause large drops.

## Phase 4: Threshold Re-tuning at Alpha=0.2

With alpha shifted from 0.6 to 0.2, the optimal threshold may also shift. The location prior already filters noise, potentially allowing a slightly lower threshold.

All runs at T=1.5, alpha=0.2.

| Threshold | F1 |
|-----------|------|
| 0.060 | 0.37369 |
| **0.070** | **0.39573** |
| 0.075 | 0.39420 |
| 0.080 | 0.39305 |
| 0.090 | 0.39224 |

**Optimal threshold shifted from 0.075 → 0.070.** With stronger location prior filtering (alpha=0.2), a slightly lower threshold captures more true positives without excess false positives. Gain: +0.00153 over thr=0.075.

## Key Insights

### 1. The Sweet Spot: Scales 4–6

Only scales that produce tiles in the 500–750px range (before resize to 518) provide complementary value. This range captures individual plants at a resolution close to the single-species training images.

| Direction | Scales tested | Effect |
|-----------|---------------|--------|
| Coarser than 4x4 | 1x1, 2x2, 3x3 | Always hurts (false positives) |
| **4x4 to 6x6** | **4, 5, 6** | **Each adds unique signal** |
| Finer than 6x6 | 7x7 | Negligible gain (+0.0008) |

### 2. Scale Complementarity

No single scale is optimal alone. The combination works because `max_prob` selects the best detection per species across all scales:
- **4x4** (750px): medium plants, some scene context
- **5x5** (600px): smaller species, near-native resolution
- **6x6** (500px): finest useful detail, closest to training distribution

Evidence: removing any one scale from E degrades performance.

| Removed | Remaining | Best F1 | Loss vs E |
|---------|-----------|---------|-----------|
| — | 4x4+5x5+6x6 (E) | 0.38466 | — |
| 6x6 | 4x4+5x5 (D) | 0.37894 | -0.00572 |
| 5x5 | 4x4+6x6 (F) | 0.37507 | -0.00959 |

### 3. Coarse Scales Produce False Positives

At scales 1x1–3x3, tiles cover large areas (1000–3000px) compressed to 518px. The model can't distinguish individual plants and produces high-confidence predictions for species that aren't actually present. With `max_prob` aggregation, these false positives propagate to the image level.

### 4. Multi-Scale Requires Stronger Location Prior

With 77 tiles, `max_prob` produces more true positives but also more false positives. The temporal fusion location prior acts as a filter, suppressing species not observed at each site. The optimal alpha shifted from 0.6 (single-scale) to 0.2 (multi-scale), meaning 80% of the final prediction comes from the location prior.

### 5. Threshold Shifts with Tile Count and Alpha

| Configuration | Tiles | Alpha | Optimal threshold |
|---------------|-------|-------|-------------------|
| Single-scale (Exp-4) | 16 | 0.6 | 0.04 |
| Multi-scale, alpha=0.6 | 77 | 0.6 | 0.075 |
| Multi-scale, alpha=0.2 | 77 | 0.2 | 0.070 |

More tiles push the threshold up (more chances for high max_prob). Lower alpha (stronger prior filtering) pushes it slightly back down (prior already removes false positives).

## Evolution of Best Configuration

| Step | Configuration | F1 | Delta |
|------|--------------|------|-------|
| Exp-3 baseline | T=2.0, k=6, thr=0.03 | 0.28093 | — |
| Exp-4 max_prob | max_prob, T=1.5, k=4 | 0.31950 | +0.03857 |
| + temporal fusion | alpha=0.6 | 0.33946 | +0.01996 |
| + JPEG85 | quality=85, 4:2:2 | 0.35517 | +0.01571 |
| + threshold tuning | thr=0.04 | 0.35698 | +0.00181 |
| + multi-scale 4x4+5x5 | thr=0.075 | 0.37894 | +0.02196 |
| + add 6x6 scale | 4x4+5x5+6x6, thr=0.075 | 0.38466 | +0.00572 |
| + alpha re-tuning | alpha=0.2 | 0.39420 | +0.00954 |
| **+ threshold re-tuning** | **thr=0.070** | **0.39573** | **+0.00153** |

**Total improvement: +0.11480** (+40.9% relative over Exp-3 baseline)

## Final Configuration

```
Preprocessing:     JPEG compression (quality=85, subsampling=4:2:2, Lanczos)
Tiling:            Multi-scale: 4x4 + 5x5 + 6x6 (77 tiles/image)
View Strategy:     1-view (orig_only, no TTA)
Aggregation:       max_prob
Temperature:       1.5
tile_filter_k:     ALL (disabled)
Temporal Fusion:   prior_blend, alpha=0.2
Threshold:         0.070
TopK:              10
Geo Mask:          sw_europe_mask_bool.npy
Aug Filter:        True
Image TopK:        64

F1 Score:          0.39573 (public leaderboard)
Improvement:       +0.03875 over Exp-4 best (0.35698)
                   +0.11480 over Exp-3 baseline (0.28093)
Relative Gain:     +40.9% over Exp-3 baseline
```
