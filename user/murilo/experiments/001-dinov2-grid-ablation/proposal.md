# Proposal: DINOv2 Grid-Size Ablation

**Status**: proposed
**Created**: 2026-03-10
**Author**: Murilo Gustineli

## Hypothesis

Varying the grid tiling size when running inference with a fine-tuned DINOv2
classifier on vegetation plot images will significantly affect species
identification performance. There exists an optimal grid size that balances
capturing fine-grained plant details (larger grids) against maintaining
sufficient context per tile (smaller grids).

## Background

In PlantCLEF 2025, our second-place solution used a fixed 4x4 tiling strategy
with the DINOv2 ViT-Base model (patch size 14, 518x518 receptive field). Tiles
were classified independently and aggregated via majority vote. However, we
never systematically evaluated how grid size impacts performance.

Vegetation plot images contain multiple plant species at varying scales. Smaller
grids preserve more context per tile but may miss small plants. Larger grids
isolate individual plants better but lose context and increase noise from
partial/ambiguous crops.

## Method

### Approach

Run frozen DINOv2 inference on all 2,106 PlantCLEF test images with 10
different grid configurations. Compare Kaggle leaderboard scores across
grid sizes to identify the optimal tiling strategy.

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

1. Load frozen DINOv2 model and class mappings
2. For each grid size (0, 2, 3, 4, 5, 6, 7, 8, 9, 10):
   a. Split each image into NxN non-overlapping tiles (0 = full image)
   b. Run each tile through the classifier (batch inference)
   c. Aggregate tile predictions via majority vote
   d. Generate submission CSV
3. Upload each CSV to Kaggle and record scores
4. Compare results across grid sizes

### Variables

- **Independent**: Grid size (0, 2, 3, 4, 5, 6, 7, 8, 9, 10)
- **Dependent**: Kaggle leaderboard score (macro-averaged F1)
- **Controlled**: Model weights (frozen), aggregation method (majority vote),
  top-k per tile (1), test dataset, image preprocessing (timm defaults)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Kaggle score | Official competition metric (macro-averaged F1) |
| Avg species per image | Number of unique species predicted per image |
| Inference time | Wall-clock time per grid configuration |

### Baseline

- **No grid (grid_size=0)**: Full-image inference as the simplest baseline
- **4x4 grid**: Replication of our PlantCLEF 2025 approach

### Success Criteria

- **Confirm if**: At least one grid size improves over no-grid baseline by >5% F1
- **Reject if**: No grid size improves over no-grid, or improvements are <2% F1

## Future Work: Aggregation Methods

This experiment uses majority vote as the baseline aggregation. Future runs
will explore:

1. **Mean probability**: Average softmax vectors across tiles, threshold top-k
2. **Max probability**: Take max probability per species across tiles
3. **Weighted vote**: Weight tile votes by confidence (max softmax probability)
4. **Threshold vote**: Only count votes from tiles with confidence above threshold
5. **Ensemble**: Combine predictions from multiple grid sizes

## Limitations

- No ground truth labels available locally; evaluation requires Kaggle upload
- The model checkpoint needs verification (may differ from 2025 competition model)
- Majority vote with top-k=1 per tile may undercount species in dense plots
- Non-square images lose edge pixels when grid size doesn't divide evenly
