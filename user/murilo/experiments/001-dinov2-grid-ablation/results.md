# Results: DINOv2 Grid-Size Ablation

**Date Started**: 2026-03-10
**Author**: Murilo Gustineli

## Summary

Testing how NxN grid tiling affects species identification performance with a
frozen DINOv2 ViT-Base classifier on PlantCLEF vegetation plot images.

**Verdict**: in progress

## Kaggle Scores — Grid Sweep

**Metric**: Macro-averaged F1 score per sample

| Grid Size | Tiles/Image | Avg Species/Image | Kaggle F1 | Delta vs Baseline | Submission |
|-----------|-------------|--------------------|-----------|--------------------|------------|
| 0 (none)  | 1           | 1.0                | 0.14506   | —                  | 2026-03-10 |
| 2×2       | 4           |                    |           |                    |            |
| 3×3       | 9           |                    |           |                    |            |
| 4×4       | 16          |                    |           |                    |            |
| 5×5       | 25          |                    |           |                    |            |
| 6×6       | 36          |                    |           |                    |            |
| 7×7       | 49          |                    |           |                    |            |
| 8×8       | 64          |                    |           |                    |            |
| 9×9       | 81          |                    |           |                    |            |
| 10×10     | 100         |                    |           |                    |            |

## Kaggle Scores — Aggregation Methods (Grid 4x4, top-k=5 per tile)

| Method | max_species | Avg Species/Image | Kaggle F1 | Notes |
|--------|-------------|-------------------|-----------|-------|
| Union (cap on unique) | None | 37.2 | 0.21313 | Too many species per image |
| Union (cap on unique) | 20 | 19.8 | ~0.21 | Uniform 20 species per image |
| Union (cap on unique) | 5 | 5.0 | ~0.21 | |
| Union (cap on unique) | 8 | 8.0 | ~0.21 | |
| Union (cap on unique) | 10 | 10.0 | ~0.21 | |
| **Expanded-list** | **20** | **~10** | **0.23** | **Best so far**. Natural variation (2-18) |
| Quadrat agg (no cap) on union topk5 | None | 183.5 | worse | Way too many species |
| Quadrat agg (top-k=10) on expanded-list max20 | 10 | 10 | 0.17103 | Quadrat agg hurts |
| 2025 reference | — | 10.3 | 0.30810* | *With quadrat agg + unknown pipeline |

## Key Findings

### 1. Model predictions are consistent with 2025
- Per-image species overlap between 2025 and 2026 outputs is 85-92%
- The model is producing similar predictions; the difference is in post-processing

### 2. Per-image species count matters significantly
- Too many species (37 avg) → F1 ~0.21
- Capping on unique species count gives uniform lengths → F1 ~0.21
- **Expanded-list aggregation** (cap on frequency-expanded list, then dedup) → naturally varying lengths → **F1 = 0.23**

### 3. Quadrat-level aggregation hurts (unexpectedly)
- Grouping images by base quadrat (2105 images → 425 quadrats) and merging species made scores WORSE
- Possible reasons: our per-image species lists are still too noisy, so aggregating amplifies noise
- The 2025 score of 0.30810 used quadrat aggregation, but with a different (unknown) per-image pipeline

### 4. The 2025 reference CSV remains unexplained
- `dsgt_run_topk_20_species_grid_4x4.csv`: avg 10.3 species/image, varying 2-18
- The `SubmissionTask` was commented out in 2025 `workflow.py` — unknown code generated this CSV
- The 2025 `aggregation.py` uses `Counter.most_common(top_k)` (cap on unique), not expanded-list

## GPU Profiling

| Grid | GPU Util (avg) | Memory Used | Data Loading % | Throughput (avg) |
|------|----------------|-------------|----------------|------------------|
| 3×3  | 78.4%          | 0.77 GB / 25.2 GB | 99.6%     | 68 samples/s     |

## Artifacts

| File | Method | Avg Species | F1 |
|------|--------|-------------|-----|
| `no_grid_majority_vote.csv` | No grid, top-k=1 | 1.0 | 0.14506 |
| `grid_4x4_majority_vote_topk5_max5.csv` | Union, max_species=5 | 5.0 | ~0.21 |
| `grid_4x4_majority_vote_topk5_max8.csv` | Union, max_species=8 | 8.0 | ~0.21 |
| `grid_4x4_majority_vote_topk5_max10.csv` | Union, max_species=10 | 10.0 | ~0.21 |
| `grid_4x4_union_topk5.csv` | Union, no cap | 37.2 | 0.21313 |
| `grid_4x4_union_topk5_max20.csv` | **Expanded-list**, max=20 | ~10 | **0.23** |
| `agg_topk10_grid_4x4_union_topk5_max20.csv` | Quadrat agg top-k=10 | 10 | 0.17103 |
| `dsgt_run_topk_20_species_grid_4x4.csv` | 2025 reference | 10.3 | — |

GPU profiling: `./outputs/gpu_check/`
Code: Branch `mg/dinov2-grid-ablation`

## Next Steps

- [ ] Run grid sweep with expanded-list aggregation (best method so far)
- [ ] Try different max_species values (10, 15, 25, 30) with expanded-list
- [ ] Try organizers' max-probability aggregation (score-threshold approach)
- [ ] Verify checkpoint identity against 2025
- [ ] Investigate what generated the 2025 reference CSV
