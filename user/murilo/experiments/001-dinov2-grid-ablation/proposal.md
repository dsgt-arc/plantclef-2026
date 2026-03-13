# Proposal: DINOv2 Grid-Size Inference

**Status**: proposed
**Created**: 2026-03-10
**Author**: Murilo Gustineli

## Hypothesis

Varying the grid tiling size when running inference with a fine-tuned DINOv2
classifier on vegetation plot images produces meaningfully different per-tile
prediction distributions. Larger grids isolate individual plants better but
produce noisier crops, while smaller grids preserve context but may miss
fine-grained species. A systematic sweep of grid sizes will provide the data
needed for downstream classification experiments.

## Background

In PlantCLEF 2025, our second-place solution used a fixed 4x4 tiling strategy
with the DINOv2 ViT-Base model (patch size 14, 518x518 receptive field). Tiles
were classified independently and aggregated via union of top-k species.
However, we never systematically evaluated how grid size impacts performance.

Vegetation plot images contain multiple plant species at varying scales. Smaller
grids preserve more context per tile but may miss small plants. Larger grids
isolate individual plants better but lose context and increase noise from
partial/ambiguous crops.

## Method

### Approach

Run frozen DINOv2 inference on all 2,106 PlantCLEF test images with 10
different grid configurations, saving full softmax probability vectors
to parquet for downstream classification by
[Experiment 002](../002-classification-methods/proposal.md).

### Setup

| Component | Details |
|-----------|---------|
| Model | Fine-tuned DINOv2 ViT-Base (vit_base_patch14_reg4_dinov2.lvd142m), 7806 classes |
| Checkpoint | `/shared/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth` |
| Data | PlantCLEF 2025 test set: 2,106 vegetation plot images |
| Class mapping | 7,806 species via `class_mapping.txt` |
| Compute | 1x RTX 6000 GPU, 8 CPUs, 64GB RAM, ~2h per grid size |
| Dependencies | timm, torch, PIL, pandas |

### Procedure

Run GPU inference once per grid size, saving full 7806-dim softmax vectors
to parquet.

1. For each grid size (0, 2, 3, 4, 5, 6, 7, 8, 9, 10):
   a. `python inference.py --grid-size N` → `logits/grid_NxN.parquet`
2. Validate parquet integrity (row counts, shape, dtype)
3. Profile GPU utilization and record wall-clock time per grid size

### Variables

- **Independent**: Grid size (0, 2, 3, 4, 5, 6, 7, 8, 9, 10)
- **Dependent**: Per-tile softmax vectors (parquet), inference time, GPU utilization
- **Controlled**: Model weights (frozen), test dataset, image preprocessing
  (timm defaults), batch size

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Inference time | Wall-clock time per grid configuration |
| GPU utilization | Average GPU compute and memory usage |
| Parquet size | Output file size per grid size |
| Tiles per image | N² tiles for grid size N |

### Baseline

- **No grid (grid_size=0)**: Full-image inference (1 tile per image)
- **4x4 grid**: Replication of our PlantCLEF 2025 approach (16 tiles per image)

### Success Criteria

- **Complete if**: All 10 grid sizes produce valid parquet files with correct
  dimensions (images × tiles × 7806 classes)
- **Reject if**: Inference fails or produces corrupt outputs for any grid size

> **Note**: Classification performance (Kaggle F1) is evaluated in
> [Experiment 002](../002-classification-methods/proposal.md).

## Limitations

- The model checkpoint needs verification (may differ from 2025 competition model)
- Non-square images lose edge pixels when grid size doesn't divide evenly
- Data-loading is the bottleneck (~99% of step time), not GPU compute
