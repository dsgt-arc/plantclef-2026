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

The input is a single image split into an NxN grid of tiles. Each tile is
passed through the DINOv2 classifier independently, producing a softmax
probability vector over 7,806 species. For a 4x4 grid, this gives 16 vectors
of 7,806 probabilities each.

**Step 1 — Pool.** Flatten all tile probability vectors into a single array.
For a 4x4 grid: 16 x 7,806 = 124,896 (probability, class_index) pairs. Every
tile contributes every species — most with near-zero probability.

**Step 2 — Sort and take top-k.** Pick the k highest probabilities from this
pool of 124,896 entries. With top-k=10, you get the 10 highest-confidence
individual predictions across all tiles, regardless of which tile or species
they came from.

**Step 3 — Map and deduplicate.** Convert each class index to its species ID
via `class_mapping.txt`, then deduplicate keeping first occurrence (= highest
probability). The same species can appear multiple times in the top-k if
multiple tiles predicted it with high confidence — each duplicate consumes a
slot.

**Why this produces variable-length species lists:**

```
Image A — tiles strongly agree (one dominant species):
  top-10: [X, X, X, X, Y, X, Y, Z, X, Y]
  after dedup: [X, Y, Z] → 3 species

Image B — tiles disagree (diverse scene):
  top-10: [A, B, C, D, E, F, G, A, H, I]
  after dedup: [A, B, C, D, E, F, G, H, I] → 9 species
```

The top-k acts as a **budget**. Confident repeated predictions "spend" budget
on species the model is sure about, leaving less room for uncertain ones.

Key insight: **probability magnitude > tile frequency**. A species predicted
at 0.55 in one tile is more informative than a species predicted at 0.02 in
six tiles. This is why sorted-prob (F1=0.31) vastly outperforms union
(F1=0.21), which ranks by tile count and discards probability magnitude.

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
