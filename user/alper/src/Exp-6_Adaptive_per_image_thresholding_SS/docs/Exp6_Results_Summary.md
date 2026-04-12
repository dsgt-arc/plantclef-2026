# Exp-6: Adaptive Per-Image Thresholding — Results Summary

## Overview

Exp-6 investigated whether replacing the fixed global threshold + topk with per-image adaptive prediction counts could improve F1. The experiment tested 4 adaptive strategies and a joint threshold × topk sweep.

**Result: Simple threshold optimization beat all adaptive strategies.** Raising threshold from 0.03 to 0.045 improved F1 from 0.35517 to **0.35859** (+0.00342).

---

## Input

Best Exp-4 temporal fusion parquet:
- JPEG85 + 1v + max_prob + T=1.5 + k=3 + temporal fusion alpha=0.6
- Previous best: F1=0.35517 (thr=0.03, topk=10)

---

## Phase 1: Adaptive Strategy Sweep (34 configurations)

### Strategies Tested

| Strategy | How it works | Configs |
|----------|-------------|---------|
| max_gap | Cut at largest probability drop | 7 |
| relative_ratio | Keep species where p[i]/p[0] > ratio | 10 |
| elbow | Maximum curvature in probability curve (Kneedle-like) | 6 |
| fixed_threshold | Baseline: count above threshold (no topk cap) | 11 |

All strategies: geo mask applied first, min_k=1, max_k=25.

### Results

| Config | F1 | Avg K | Notes |
|--------|-----|-------|-------|
| **fixed_threshold thr=0.040** | **0.35698** | 3.6 | Best in Phase 1 |
| fixed_threshold thr=0.050 | 0.35531 | 2.8 | |
| fixed_threshold thr=0.030 | 0.35503 | 4.9 | ~same as old best (0.35517) |
| relative_ratio ratio=0.30 | 0.35380 | 4.3 | Best adaptive |
| relative_ratio ratio=0.20 | 0.34650 | 6.6 | |
| relative_ratio ratio=0.15 | 0.34066 | 8.8 | |
| fixed_threshold thr=0.020 | 0.33239 | 7.8 | |
| relative_ratio ratio=0.10 | 0.31585 | 12.3 | |
| fixed_threshold thr=0.015 | 0.30956 | 10.3 | |
| relative_ratio ratio=0.07 | 0.28746 | 15.2 | |
| fixed_threshold thr=0.010 | 0.27174 | 14.9 | |
| elbow smooth=3, N=50 | 0.25418 | 9.0 | |
| elbow smooth=5, N=50 | 0.23984 | 9.3 | |
| max_gap (all N values) | — | 1.6 | Broken: always cuts after 1st species |

### Key Finding

**F1 monotonically improves as prediction count decreases from 15 → 4:**

```
F1
0.357 |     *                                  ← K≈3.6 (thr=0.04)
0.355 |  *     *                               ← K≈2.8-4.9
0.354 |           *
0.347 |              *
0.341 |                 *
0.332 |                       *
0.316 |                          *
0.310 |                             *
0.287 |                                *
0.272 |                                   *
      +--+--+--+--+--+--+--+--+--+--+--+--
         2  4  6  8  10 12 14 16      Avg K
```

The old topk=10 was padding most images with 5-7 false positives.

### Why Adaptive Strategies Failed

1. **max_gap**: Probability distributions have a dominant first peak with a huge drop to second species. Max gap always fires after position 1 → avg K=1.6 (far too aggressive).

2. **elbow**: Despite producing reasonable avg K (7-9), the per-image cut points were noisy. The elbow detection doesn't distinguish true species from false positives — it just finds a geometric property of the curve.

3. **relative_ratio**: Performed reasonably (ratio=0.30 gave F1=0.35380, K=4.3) but couldn't beat a simple fixed threshold. The ratio normalization by p[0] introduces instability when p[0] varies across images.

4. **fixed_threshold wins** because the probability calibration from max_prob + temperature + temporal fusion is globally consistent. A single threshold cleanly separates true from false predictions across all images.

---

## Phase 2: Threshold × TopK Joint Sweep (35 configurations)

Sweep grid: topk ∈ {3,4,5,6,7} × threshold ∈ {0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050}

### Top Results

| Threshold | TopK | Avg K | F1 |
|-----------|------|-------|----|
| **0.045** | **7** | **3.1** | **0.35859** |
| 0.045 | 6 | 3.1 | 0.35841 |
| 0.045 | 5 | 3.0 | 0.35764 |
| 0.040 | 7 | 3.6 | 0.35695 |
| 0.035 | 7 | 4.0 | 0.35521 |
| 0.050 | 6 | 2.8 | 0.35531 |
| 0.050 | 5 | 2.7 | 0.35504 |
| 0.040 | 4 | 3.0 | 0.35174 |
| 0.035 | 5 | 3.6 | 0.35156 |
| 0.030 | 5 | 3.9 | 0.35073 |

### Key Finding: TopK Doesn't Matter When Threshold Is Optimal

At thr=0.045, top5/top6/top7 produce avg K=3.0-3.1. The topk cap rarely binds because few images have >5 species above 0.045. The F1 differences (0.35764 → 0.35859) come from a handful of images where the cap clips valid predictions.

**Higher threshold + more topk headroom > lower threshold + tight cap:**
- thr=0.045, top5 (K=3.0) → 0.35764 (higher bar, slight cap)
- thr=0.040, top4 (K=3.0) → 0.35174 (lower bar, tight cap)

Same avg K, vastly different F1 (+0.006). Prediction quality (threshold) matters far more than prediction count (topk).

---

## Final Configuration

```
Threshold:    0.045  (was 0.03)
TopK:         10     (not binding — only ~3 species/image exceed threshold)
All else:     unchanged from Exp-4 best

F1 Score:     0.35859 (public leaderboard)
```

### Evolution of Best Config (Full History)

| Step | Config | F1 | Key Change |
|------|--------|-----|-----------|
| Exp-3 Baseline | noisy_or + T=2.0 + k=6 + thr=0.03 + topk=10 | 0.28093 | — |
| Exp-4 Step 4 | per_view_or + max_prob + T=1.5 + k=3 | 0.31487 | max_prob + T=1.5 |
| Exp-4 Step 7 | orig_only + max_prob + T=1.5 + k=4 | 0.31950 | Remove TTA |
| Exp-4 Temporal | 1v + k=4 + alpha=0.1 | 0.33946 | Location prior blending |
| Exp-4 JPEG85 | JPEG85 + 1v + k=3 + alpha=0.6 | 0.35517 | JPEG compression |
| Exp-6 Thr Opt. | thr=0.04 | 0.35698 | Threshold 0.03→0.04 |
| **Exp-6 Final** | **thr=0.045** | **0.35859** | **Threshold 0.04→0.045** |

**Total improvement: +0.07766 (+27.6% relative)**

---

## Compute Cost

| Phase | Configs | Time | Resource |
|-------|---------|------|----------|
| Adaptive strategy sweep | 34 | ~45s | CPU, 16GB |
| Threshold × TopK sweep | 35 | ~40s | CPU, 16GB |
| **Total** | **69** | **~90s** | CPU-only |

---

## Conclusion

1. The optimal prediction count is **~3 species per image**, not 10.
2. A simple fixed threshold (0.045) outperforms all adaptive strategies.
3. The probability calibration from max_prob + temperature + temporal fusion is globally consistent — no per-image adaptation needed.
4. topk is irrelevant as long as it doesn't clip (≥7 is sufficient at thr=0.045).
