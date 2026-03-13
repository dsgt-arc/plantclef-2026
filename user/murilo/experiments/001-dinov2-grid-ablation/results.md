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

> For aggregation method scores, see
> [Experiment 002: Results](../002-classification-methods/results.md).

## GPU Profiling

| Grid | GPU Util (avg) | Memory Used | Data Loading % | Throughput (avg) |
|------|----------------|-------------|----------------|------------------|
| 3×3  | 78.4%          | 0.77 GB / 25.2 GB | 99.6%     | 68 samples/s     |

## Artifacts

GPU profiling: `./outputs/gpu_check/`
Parquet outputs: `./logits/`
Code: Branch `mg/dinov2-grid-ablation`

## Next Steps

- [ ] Run all grid sizes (--all) and record Kaggle scores
- [ ] Verify checkpoint identity against 2025
- [ ] Compare inference time across grid sizes
