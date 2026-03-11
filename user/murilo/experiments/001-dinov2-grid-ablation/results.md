# Results: DINOv2 Grid-Size Ablation

**Date Started**: 2026-03-10
**Author**: Murilo Gustineli

## Summary

Testing how NxN grid tiling affects species identification performance with a
frozen DINOv2 ViT-Base classifier on PlantCLEF vegetation plot images.

**Verdict**: in progress

## Kaggle Scores

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

**Aggregation**: majority vote, top-k=1 per tile

## Observations

### No-Grid Baseline (grid_size=0)

- F1 = 0.14506 — expected to be low
- With top-k=1 and no tiling, only 1 species is predicted per image
- Vegetation plots contain many species, so recall is near zero for most images
- This establishes a floor; any grid size should improve substantially by
  predicting more species per image

### GPU Profiling

| Grid | GPU Util (avg) | Memory Used | Data Loading % | Throughput (avg) |
|------|----------------|-------------|----------------|------------------|
| 3×3  | 78.4%          | 0.77 GB / 25.2 GB | 99.6%     | 68 samples/s     |

- GPU memory usage is very low (~3.1%), batch size could be increased further
- Data loading (NFS I/O) dominates step time — expected for a small model on
  network storage
- GPU utilization reaches 100% during compute bursts; the 78.4% average
  reflects I/O gaps between batches

## Analysis

The no-grid baseline confirms the core issue: with majority vote and top-k=1,
only one species is predicted per image. Since PlantCLEF images are multi-species
vegetation plots, this yields very low recall and consequently low F1.

Grid tiling should improve F1 by:
1. Different tiles capture different dominant species → more unique species predicted
2. More tiles = more "voters" = better coverage of the species present in the plot

The key question is whether larger grids (more tiles, more species detected) will
continue to improve or will degrade as tiles become too small to classify reliably.

### Confounders

- Model checkpoint originates from PlantCLEF 2025 and may not be optimal for 2026 data
- Majority vote with top-k=1 is the simplest aggregation; better methods may
  shift the optimal grid size
- Non-square images lose edge pixels when grid size doesn't divide evenly

## Artifacts

| Type | Location |
|------|----------|
| Submission CSVs | `./artifacts/submission_*.csv` |
| GPU profiling | `./outputs/gpu_check/` |
| Code | Branch `mg/dinov2-grid-ablation`, PR #7 |

## Next Steps

- [ ] Run all grid sizes via `sbatch/run.sh --all`
- [ ] Submit each CSV to Kaggle and fill in scores table
- [ ] Identify optimal grid size
- [ ] Test alternative aggregation methods (mean probability, weighted vote)
- [ ] Test top-k > 1 per tile with majority vote
- [ ] Plot F1 vs grid size curve

**Leads to**: Aggregation method experiments, ensemble across grid sizes
