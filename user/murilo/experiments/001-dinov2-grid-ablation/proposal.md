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
were classified independently and aggregated via union of top-k species.
However, we never systematically evaluated how grid size impacts performance.

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
   c. Aggregate tile predictions (union or majority vote)
   d. Generate submission CSV
3. Upload each CSV to Kaggle and record scores
4. Compare results across grid sizes

### Variables

- **Independent**: Grid size (0, 2, 3, 4, 5, 6, 7, 8, 9, 10)
- **Dependent**: Kaggle leaderboard score (macro-averaged F1)
- **Controlled**: Model weights (frozen), aggregation method, top-k per tile,
  test dataset, image preprocessing (timm defaults)

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

## Aggregation Methods

Each tile produces a softmax probability vector over 7,806 species. The top-k
species per tile are extracted, then aggregated across all tiles for an image.

### Per-image aggregation (tiles → image prediction)

#### 1. Union (default, `--aggregation union`, no `--max-species`)

Collect all unique species from every tile's top-k predictions. Return all of
them, sorted by frequency (how many tiles predicted each species). With 4x4
grid and top-k=5, produces avg ~37 unique species per image. **F1 = 0.21313**.

**CLI**: `python classify.py --grid-size 4 --top-k 5`

#### 2. Union with max-species cap (cap on unique species)

Same as union, but `Counter.most_common(max_species)` caps the number of
unique species returned. Produces uniform-length lists. **F1 ~0.21** regardless
of cap value (5, 8, 10, 20). This is also what the 2025 `aggregation.py`
functions (`union_agg_topk`) implement.

**CLI**: `python classify.py --grid-size 4 --top-k 5 --max-species 10`

#### 3. Expanded-list aggregation (current implementation with `--max-species`)

Build frequency-expanded list (species repeated by tile count), sort by
frequency, cap the expanded list at `--max-species`, then deduplicate. Species
that tiles strongly agree on consume more slots, naturally filtering rare
species. Produces varying-length lists (2-18 unique species with max=20).
**F1 = 0.23** — best so far.

Example: 3 species with counts [8, 5, 3], max_species=10:
  Expanded: [A]*8 + [B]*5 + [C]*3 → cap at 10 → [A]*8 + [B]*2 → dedup → [A, B]

Note: This approach is NOT in the 2025 codebase. The 2025 `union_agg_topk`
uses `Counter.most_common(top_k)` (cap on unique). However, expanded-list
produces output distributions similar to the 2025 reference CSV and scores
better empirically.

**CLI**: `python classify.py --grid-size 4 --top-k 5 --max-species 20`

### Quadrat-level aggregation (`quadrat_aggregate.py`)

Groups images by base quadrat ID (strips date suffixes) and unions species
across all images of the same quadrat. 2105 images → 425 unique quadrats
(avg 5 images/quadrat). The same aggregated species list is assigned to every
image in the group.

Pattern matching for quadrat grouping:
- `CBN-PdlC-A1-20130807` → `CBN-PdlC-A1`
- `RNNB-12-3-20210615` → `RNNB-12-3`
- `OPTMix-42-20200801` → `OPTMix-42`

**Result**: Quadrat aggregation made scores worse (F1 = 0.17). The 2025 pipeline
achieved 0.30810 with quadrat aggregation, but the per-image inputs were
different. Our per-image species lists may be too noisy for aggregation to help.

**CLI**: `python quadrat_aggregate.py artifacts/input.csv --top-k 10`

### Not yet implemented

- **Max probability**: Keep highest probability per species across tiles (organizers' notebook approach, scored 0.23)
- **Mean probability**: Average softmax vectors across tiles, threshold top-k
- **Weighted vote**: Weight tile votes by confidence
- **Ensemble**: Combine predictions from multiple grid sizes

## 2025 Pipeline Comparison

The 2025 PlantCLEF pipeline (in `../plantclef-2025/`) used a 3-stage process:

1. **Inference** (`classification/workflow.py`): Spark-based, tiles images via UDF,
   runs DINOv2 per tile, saves probability vectors to parquet
2. **Submission** (`classification/submission.py`): Extracts per-tile top-k species,
   generates per-image CSV. **Note**: `SubmissionTask` was commented out in
   `workflow.py` — the reference CSV was generated by unknown code.
3. **Aggregation** (`classification/aggregation.py`): Groups by base quadrat ID,
   unions species across images, optional top-k cap

Key 2025 scripts: `user/mgustineli/ondemand-grid-classification.sh` (inference
with `--grid-size 4 --top-k-proba 12`) and `ondemand-agg-submission.sh`
(aggregation with `top_k_species=10`).

The 2025 reference CSV (`dsgt_run_topk_20_species_grid_4x4.csv`) has avg 10.3
species/image with natural variation (2-18). How it was generated is unknown —
the SubmissionTask was commented out, and the "topk_20" naming doesn't match
any observed CLI parameters.

## Limitations

- No ground truth labels available locally; evaluation requires Kaggle upload
- The model checkpoint needs verification (may differ from 2025 competition model)
- Majority vote with top-k=1 per tile may undercount species in dense plots
- Non-square images lose edge pixels when grid size doesn't divide evenly
