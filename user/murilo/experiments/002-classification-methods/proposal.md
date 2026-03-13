# Proposal: Classification Methods for PlantCLEF 2026

**Status**: proposed
**Created**: 2026-03-12
**Author**: Murilo Gustineli

## Hypothesis

The classification method — how per-tile softmax predictions are combined into
per-image species lists — is the primary driver of F1 variation given fixed
inference outputs. Finding the right approach can close the gap from our current
best (F1=0.23) toward the 2025 reference score (F1=0.31).

## Background

Experiment 001 (DINOv2 grid inference) produces per-tile softmax vectors saved
to parquet. This experiment takes those outputs and explores different
classification strategies: how to aggregate tile predictions, threshold
confidence, and select species per image.

Initial experiments showed that method choice matters significantly — union
aggregation scored F1=0.21 regardless of species cap, while expanded-list
aggregation reached F1=0.23.

The 2025 pipeline achieved F1=0.30810, but how its reference CSV was generated
remains unknown (the `SubmissionTask` was commented out in `workflow.py`). The
gap between 0.23 and 0.31 likely comes from classification/post-processing
differences rather than model differences (per-image species overlap is 85-92%).

## Method

### Approach

Read parquet files produced by Experiment 001 (GPU inference), apply different
classification strategies (aggregation, thresholding, species selection), and
upload submission CSVs to Kaggle. Iterate quickly on CPU without re-running
expensive GPU inference.

### Setup

| Component | Details |
|-----------|---------|
| Data | Parquet files from `../001-dinov2-grid-ablation/logits/` |
| Compute | CPU only (no GPU needed) |
| Code | `classify.py` (self-contained, no `plantclef` imports) |
| Dependencies | pandas, pyarrow, numpy |
| Class mapping | 7,806 species via `class_mapping.txt` |

### Procedure

1. Read parquet from exp 001: `../001-dinov2-grid-ablation/logits/grid_NxN.parquet`
2. Apply aggregation method via `classify.py`
3. Upload submission CSV to Kaggle
4. Record score in results.md

### Baseline

- **Union no cap** (F1=0.21313): All unique species from top-k per tile, no filtering
- **Expanded-list max20** (F1=0.23): Frequency-expanded list, cap at 20, dedup — best so far

## Classification Methods

Each tile produces a softmax probability vector over 7,806 species. The
classification task is to combine these per-tile predictions into a final
per-image species list.

### Per-image classification (tiles → species list)

#### 1. Union (default, `--method union`, no `--max-species`)

Collect all unique species from every tile's top-k predictions. Return all of
them, sorted by frequency (how many tiles predicted each species). With 4x4
grid and top-k=5, produces avg ~37 unique species per image. **F1 = 0.21313**.

**CLI**: `python classify.py --input logits/grid_4x4.parquet --method union --top-k 5`

#### 2. Union with max-species cap (cap on unique species)

Same as union, but `Counter.most_common(max_species)` caps the number of
unique species returned. Produces uniform-length lists. **F1 ~0.21** regardless
of cap value (5, 8, 10, 20). This is also what the 2025 `aggregation.py`
functions (`union_agg_topk`) implement.

**CLI**: `python classify.py --input logits/grid_4x4.parquet --method union --top-k 5 --max-species 10`

#### 3. Expanded-list aggregation (best method, F1=0.23)

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

**CLI**: `python classify.py --input logits/grid_4x4.parquet --method expanded-list --top-k 5 --max-species 20`

#### 4. Max probability (organizers' approach)

Keep highest probability per species across all tiles, filter by threshold.
Scored 0.23 in organizers' notebook.

**CLI**: `python classify.py --input logits/grid_4x4.parquet --method max-prob --min-score 0.01`

### Quadrat-level post-processing (in `classify.py`)

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

**CLI**: `python classify.py --input logits/grid_4x4.parquet --method expanded-list --top-k 5 --max-species 20 --quadrat-agg --quadrat-top-k 10`

### Not yet implemented

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

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Kaggle score | Official competition metric (macro-averaged F1) |
| Avg species per image | Number of unique species predicted per image |

### Success Criteria

- **Confirm if**: A classification method closes the gap to F1 ≥ 0.28
- **Reject if**: No method improves beyond expanded-list's F1=0.23

## Limitations

- No ground truth labels available locally; evaluation requires Kaggle upload
- Quality of classification is bounded by quality of per-tile predictions from exp 001
- The 2025 reference pipeline remains partially unknown
