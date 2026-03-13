# Proposal: Classification Methods for PlantCLEF 2026

**Status**: in progress
**Created**: 2026-03-12
**Author**: Murilo Gustineli

## Goal

Find the best way to combine per-tile DINOv2 softmax predictions into per-image
species lists. The 2025 inference pipeline has been fully reproduced (F1=0.30810).
The goal now is to improve beyond F1=0.31.

## Setup

Experiment 001 runs GPU inference and saves per-tile softmax vectors to parquet.
This experiment reads those parquet files and applies classification methods on
CPU — no GPU needed, fast iteration.

```
python classify.py --grid-size 4                          # sorted-prob (default)
python classify.py --grid-size 4 --method union           # union by tile frequency
python classify.py --grid-size 4 --method max-prob        # organizers' approach
```

Data: `/shared/inference/dinov2-vit-base/grid_NxN.parquet`
Code: `classify.py` (self-contained, no `plantclef` imports)

## Baseline (2025 reproduction)

**Sorted-prob with top-k=9 on 4x4 grid: F1 = 0.30810** (Kaggle-confirmed).

This exactly matches the 2025 reference CSV (`dsgt_run_topk_9_grid_4x4.csv`).
99.9% exact match at the per-image level (2103/2105 images identical).

## Classification Methods

Each tile produces a softmax vector over 7,806 species. The classification
method combines these into a per-image species list.

### 1. Sorted-prob (default) — F1 = 0.30810

Pool all per-tile predictions, sort by probability descending, take top-k
entries, deduplicate. Produces variable-length species lists: confident species
consume multiple top-k slots, so images with strong agreement get fewer unique
species.

```
4 tiles, top-k=6:
  (0.55, A)  (0.28, B)  (0.13, A)  (0.13, C)  (0.06, B)  (0.05, D)
  → dedup → [A, B, C, D]  (4 species, not 6 — A and B consumed extra slots)
```

Key insight: **probability magnitude > tile frequency**. A species at 0.55 in
one tile is more informative than a species at 0.02 in six tiles.

### 2. Union — F1 = 0.19–0.21

Count how many tiles predicted each species, return most common. Ranks by tile
frequency, ignoring probability magnitude. Always returns exactly `--max-species`
species. F1 insensitive to cap value.

### 3. Expanded-list — F1 = 0.23

Frequency-expanded list (species repeated by tile count), cap, dedup. Better
than union because high-frequency species consume more slots, but still
ignores probability magnitude.

### 4. Max-prob (organizers' approach)

Per-species max probability across tiles, threshold filter. Scored 0.23 in
organizers' notebook. Not yet submitted.

### 5. Quadrat-level aggregation (post-processing)

Group images by vegetation plot, union species across images of the same plot.
Made scores worse (F1=0.17) — per-image lists are too noisy for aggregation
to help.

## To explore

- Sorted-prob with different top-k values (6, 12, 15, 20)
- Sorted-prob across different grid sizes (2x2, 3x3, 5x5, etc.)
- Mean-probability averaging across tiles
- Weighted voting (tile confidence as weight)
- Ensembling predictions from multiple grid sizes
- Quadrat-agg on top of sorted-prob (may work now that per-image lists are cleaner)

## Evaluation

- **Metric**: Kaggle macro-averaged F1 (no local ground truth)
- **Baseline**: F1 = 0.30810 (sorted-prob, top-k=9, grid 4x4)
- **Target**: F1 > 0.31
