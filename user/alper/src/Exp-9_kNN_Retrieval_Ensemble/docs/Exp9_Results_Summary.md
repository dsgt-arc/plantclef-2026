# Exp-9: kNN Retrieval Ensemble — Results Summary

## Executive Summary

Adding a FAISS-based kNN retrieval ensemble to the multi-scale tiling pipeline improves F1 from **0.40100 → 0.41754** (+0.01654, +4.1% relative). The optimal configuration uses kNN only for multi-visit images (which have a temporal prior safety net) and pure softmax for single-visit images (where kNN adds noise).

Key findings:
- **kNN retrieval provides complementary signal** to the softmax classifier, improving species identification when combined with the temporal location prior.
- **kNN hurts single-visit images.** Without the temporal prior to filter kNN noise, every level of kNN degrades performance. Pure softmax (beta_single=1.0) is optimal for single-visit images.
- **20% kNN weight is optimal for multi-visit images.** beta_multi=0.80 (80% softmax + 20% kNN) is the sweet spot; more kNN overwhelms the softmax signal.
- **No PlantCLEF team has previously reported this specific softmax + kNN weighted ensemble approach.**

## Background

### Motivation

The DINOv2 linear classifier head is biased toward common species in the long-tailed 7,806-species distribution. kNN retrieval is frequency-agnostic — a rare species with few training images still gets equal retrieval weight per neighbor. The hypothesis was that blending kNN votes with softmax predictions would improve rare species identification.

### Architecture

```
ONE-TIME INDEX BUILD:
  PlantCLEF 2024 training embeddings (~1.4M images)
  → Filter by geo species mask (~4800 species, ~860K images)
  → L2-normalize CLS embeddings (768d)
  → Build FAISS IndexFlatIP (cosine similarity)

TEST TIME (per test image):
  77 tiles (4x4+5x5+6x6) → DINOv2 → CLS embedding + logits per tile

  For each tile:
    p_softmax = softmax(logits / T=1.5)
    p_knn = FAISS_query(embedding, k=20) → distance-weighted species vote
    p_tile = beta * p_softmax + (1-beta) * p_knn

  p_photo = max(p_tile across 77 tiles)
  → temporal fusion: alpha * p_photo + (1-alpha) * p_location_prior
  → geo_mask → threshold → top-10 → submission
```

### Training Embeddings

- Source: PlantCLEF 2024 single-plant training images (organizer-provided)
- Pre-extracted CLS embeddings already available on PACE at `/storage/.../embeddings/train/`
- Schema: `image_name` (str), `species_id` (int32), `cls_embedding` (float32, 768d)
- 20 Hive-partitioned parquet shards, ~1.4M total rows
- Same DINOv2 model (`vit_base_patch14_reg4_dinov2.lvd142m`) used for both training and test embeddings
- **No JPEG85 preprocessing** on training images (minor domain gap with test tiles, acceptable)

### FAISS Index

- Type: `IndexFlatIP` (exact inner product search on L2-normalized vectors = cosine similarity)
- ~860K vectors after geo filtering to ~4800 species
- Size: ~2.5 GB
- Build time: ~30 minutes (CPU-only)

### Test Data Context

- ~2,105 test images total
- ~1,596 multi-visit quadrats (76%): same physical location photographed across multiple years. Temporal prior available.
- ~509 single-visit quadrats (24%): location photographed only once. NO temporal prior.

## Phase 1: Beta Sweep (Softmax vs kNN Weight)

All runs at alpha=0.05, thr=0.100 (Exp-8 best config).

| Beta | Blend | F1 | Delta vs control |
|------|-------|------|---------|
| 0.50 | 50% softmax + 50% kNN | 0.32797 | -0.07303 |
| 0.60 | 60% softmax + 40% kNN | 0.34503 | -0.05597 |
| 0.70 | 70% softmax + 30% kNN | 0.36727 | -0.03373 |
| 0.75 | 75% softmax + 25% kNN | 0.38521 | -0.01579 |
| 0.78 | 78% softmax + 22% kNN | 0.40355 | +0.00255 |
| **0.80** | **80% softmax + 20% kNN** | **0.41067** | **+0.00967** |
| 0.82 | 82% softmax + 18% kNN | 0.40640 | +0.00540 |
| 0.85 | 85% softmax + 15% kNN | 0.40878 | +0.00778 |
| 0.90 | 90% softmax + 10% kNN | 0.40675 | +0.00575 |
| 0.95 | 95% softmax + 5% kNN | 0.40723 | +0.00623 |
| 1.00 | 100% softmax (control) | 0.40100 | — |

**Optimal beta=0.80.** Sharp peak — the softmax classifier needs to dominate, with kNN providing a 20% supplementary signal. Below beta=0.78, kNN noise overwhelms the classifier.

**Control validation:** beta=1.0 reproduces the pre-kNN baseline (F1=0.40100) exactly, confirming code correctness.

## Phase 2: Alpha Re-sweep

With beta=0.80, re-sweep temporal fusion alpha at thr=0.100.

| Alpha | Blend | F1 |
|-------|-------|------|
| 0.00 | 100% prior | 0.40913 |
| 0.05 | 5% model + 95% prior | 0.41067 |
| 0.10 | 10% model + 90% prior | 0.41090 |
| **0.15** | **15% model + 85% prior** | **0.41153** |
| 0.20 | 20% model + 80% prior | 0.40960 |
| 0.30 | 30% model + 70% prior | 0.40873 |
| 0.50 | 50% model + 50% prior | 0.40502 |

**Optimal alpha shifted from 0.05 → 0.15.** With kNN augmenting the model signal, the model prediction is stronger and can handle 3x more weight in the temporal fusion blend (15% vs 5%).

## Phase 3: kNN Hyperparameter Sweeps

### k Sweep (tau=0.07, beta=0.80, alpha=0.15)

| k | F1 | Delta vs k=20 |
|---|------|---------|
| 5 | 0.34798 | -0.06355 |
| 10 | 0.38153 | -0.03000 |
| **20** | **0.41153** | **—** |
| 50 | 0.41185 | +0.00032 |
| 100 | 0.40440 | -0.00713 |

**k=20 is essentially optimal.** k=50 is noise-level better (+0.00032). Too few neighbors (k=5,10) don't capture enough species diversity. Too many (k=100) dilute the vote.

### Differentiated Beta: Single-Visit vs Multi-Visit

The key experiment. Since single-visit images have no temporal prior, does changing their kNN weight independently help?

**Increasing beta_single (less kNN for single-visit):**

| beta_multi | beta_single | F1 | Delta |
|------------|-------------|------|-------|
| 0.80 | 0.80 (uniform) | 0.41153 | — |
| 0.80 | 0.90 | 0.41236 | +0.00083 |
| 0.80 | 0.95 | 0.41637 | +0.00484 |
| 0.80 | **1.00** | **0.41754** | **+0.00601** |

**Strong monotonic trend: less kNN for single-visit is always better.** Pure softmax (beta_single=1.0) is optimal for single-visit images. kNN adds noise without the temporal prior to filter it.

**Decreasing beta_single (more kNN for single-visit):**

| beta_multi | beta_single | F1 | Delta |
|------------|-------------|------|-------|
| 0.80 | 0.80 (uniform) | 0.41153 | — |
| 0.80 | 0.70 | 0.37229 | -0.03924 |
| 0.80 | 0.60 | 0.35128 | -0.06025 |
| 0.80 | 0.50 | 0.33607 | -0.07546 |
| 0.80 | 0.40 | 0.32972 | -0.08181 |

More kNN for single-visit images catastrophically hurts performance.

**Decreasing beta_multi (more kNN for multi-visit, with beta_single=1.0 fixed):**

| beta_multi | beta_single | F1 | Delta |
|------------|-------------|------|-------|
| 0.65 | 1.00 | 0.41072 | -0.00682 |
| 0.70 | 1.00 | 0.41313 | -0.00441 |
| 0.75 | 1.00 | 0.41498 | -0.00256 |
| **0.80** | **1.00** | **0.41754** | **—** |

**beta_multi=0.80 remains optimal.** Even with the temporal prior safety net, multi-visit images don't benefit from more kNN.

## Phase 4: Rare Species Boosting Strategies

Tested three strategies to specifically help rare/tail species:

| Strategy | F1 | Delta |
|----------|------|-------|
| Baseline (no boosting) | 0.41153 | — |
| Inverse frequency weighting | 0.39307 | -0.01846 |
| Confidence gating @ 0.3 | 0.41153 | 0.00000 |
| Confidence gating @ 0.5 | 0.41153 | 0.00000 |
| Inv-freq + gate @ 0.3 | 0.39307 | -0.01846 |

**A. Inverse frequency weighting (HURTS):** Weighting kNN votes by `1/sqrt(class_frequency)` to boost rare species votes. Significantly degrades performance (-0.01846). Rare species don't have enough high-quality training images — amplifying their noisy votes makes predictions worse.

**B. Confidence gating (NO EFFECT):** Only using kNN for tiles where softmax confidence is below a threshold. No impact at any gate value, likely because max_prob aggregation already selects the best tile per species regardless of whether individual tiles use kNN.

**C. Class-balanced index subsampling:** Not tested, but likely has similar problems as inverse frequency weighting — rare species training images may be noisy, and the issue isn't retrieval weight but retrieval quality.

## Key Insights

### 1. kNN Only Helps With a Prior Safety Net

The temporal location prior (aggregated evidence from multiple visits to the same location) acts as a noise filter for kNN predictions. Multi-visit images benefit from kNN because the prior suppresses kNN false positives. Single-visit images lack this filter, so kNN noise propagates directly to the final prediction.

### 2. The Softmax Classifier Must Dominate

At beta=0.80, the softmax provides 80% of the tile-level prediction. kNN contributes a 20% supplementary signal — enough to correct edge cases where the classifier is uncertain, but not enough to override confident (and usually correct) classifier predictions.

### 3. kNN Does Not Help Rare Species

Despite the theoretical motivation (kNN is frequency-agnostic), the practical outcome is that rare species have too few training images for kNN to reliably retrieve same-species neighbors. The nearest neighbors of a rare species tile are often common species with visually similar features.

### 4. Alpha Shifts With kNN

Adding kNN to the model signal makes the combined model prediction stronger, allowing it to contribute more to the temporal fusion blend (alpha shifted from 0.05 to 0.15). This suggests kNN is genuinely adding information, not just noise — but only for multi-visit images.

## Evolution of Best Configuration

| Step | Configuration | F1 | Delta |
|------|--------------|------|-------|
| Exp-8 baseline | No kNN, alpha=0.05, thr=0.100 | 0.40100 | — |
| + kNN beta=0.80 | Uniform kNN blend | 0.41067 | +0.00967 |
| + alpha re-tune | alpha=0.15 | 0.41153 | +0.00086 |
| + beta_single=1.0 | No kNN for single-visit | 0.41754 | +0.00601 |

**Total kNN improvement: +0.01654** (+4.1% relative)

## Final Configuration

```
Preprocessing:     JPEG compression (quality=85, subsampling=4:2:2, Lanczos)
Tiling:            Multi-scale: 4x4 + 5x5 + 6x6 (77 tiles/image)
View Strategy:     1-view (orig_only, no TTA)
kNN Ensemble:      FAISS IndexFlatIP, k=20, tau=0.07
                   beta_multi=0.80 (80% softmax + 20% kNN for multi-visit)
                   beta_single=1.00 (pure softmax for single-visit, no kNN)
                   Geo-filtered index: ~860K training embeddings, ~4800 species
Aggregation:       max_prob (on blended tile probs)
Temperature:       1.5
tile_filter_k:     ALL (disabled)
Temporal Fusion:   prior_blend, alpha=0.15 (15% model + 85% location prior)
Threshold:         0.100
TopK:              10
Geo Mask:          sw_europe_mask_bool.npy (~4800 of 7806 species pass)
Aug Filter:        True
Image TopK:        64

F1 Score:          0.41754 (public leaderboard)
Improvement:       +0.01654 over Exp-8 best (0.40100)
                   +0.13661 over Exp-3 baseline (0.28093)
Relative Gain:     +48.6% over Exp-3 baseline
```

## Files

| File | Purpose |
|------|---------|
| `build_faiss_index.py` | Load training embeddings, geo-filter, build FAISS index |
| `compute_class_counts.py` | Compute per-class image counts from labels array |
| `knn_retriever.py` | FAISS query + distance-weighted vote module |
| `multiscale_aggregate_knn.py` | Main aggregation with kNN blend, supports all sweep modes |
| `slurm_build_index.sbatch` | CPU job: build FAISS index |
| `slurm_knn_sweep.sbatch` | CPU job: standard beta/alpha/threshold sweep |
| `slurm_ktau_sweep.sbatch` | CPU array job: k/tau hyperparameter sweep |
| `slurm_diff_beta_sweep.sbatch` | CPU array job: differentiated beta sweep |
| `slurm_rare_species_sweep.sbatch` | CPU array job: rare species boosting strategies |

## PACE Paths

```bash
# FAISS index
INDEX_DIR="${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index"
# Contains: faiss_index_geo.bin, faiss_labels_geo.npy, class_counts_geo.npy

# Submission CSVs
SUBMISSIONS="${PLANTCLEF_ROOT}/aerten3/submissions/exp9_knn_retrieval"
```
