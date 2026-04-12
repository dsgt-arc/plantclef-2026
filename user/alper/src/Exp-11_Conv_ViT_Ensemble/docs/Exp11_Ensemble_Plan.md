# Exp-11: ConvNeXt + ViT Ensemble

## Context

Current best: ViT-B DINOv2 pipeline (Exp-9) achieves F1=0.41754 on Kaggle.
ConvNeXt-L DINOv3 Phase 2 is projected to reach ~74-75% top-1 accuracy.
Before investing time in full ConvNeXt pipeline integration (multi-scale tiling, kNN, temporal fusion), we need to validate that ensembling actually improves F1.

## Goal

Quick validation: Does blending ConvNeXt + ViT predictions improve F1 over ViT alone?

---

## Quick Test Strategy: 4x4 Single-Scale Ensemble

### Why 4x4 only?

- 16 tiles per image vs 77 (full scheme) = ~5x faster extraction
- We're testing IF ensembling helps, not optimizing it
- If it helps at 4x4, it will help more with full multi-scale
- 4x4 tiles are 750x750 from 3000x3000 source — good coverage for medium-sized plants

### Evaluation Order

1. **ViT 4x4 alone** → F1 baseline (already known from Exp-4: ~0.357)
2. **ConvNeXt 4x4 alone** → F1 (same postprocessing: max_prob, T=1.5, temporal fusion, geo_mask, threshold)
3. **Ensemble 4x4** → F1 (sweep weight `w`)

Comparing steps 1 and 2 tells us standalone ConvNeXt quality.
The gap between step 3 and the better of 1-2 is the pure ensemble gain.

### Step-by-Step

#### Step 1: Extract ConvNeXt 4x4 embeddings + logits

Run `extract_convnext_embeddings.py` on 2105 test quadrat images at scale=4:

```
Input: 2105 test images (3000x3000)
Process: center crop → resize to 2072x2072 (4*518) → tile into 16 tiles of 518x518
Output: scale_4x4.parquet with columns:
  - image_name, tile_index, embedding_orig_f16, teacher_logits_f16, max_prob
```

**Key difference from ViT**: ConvNeXt uses `forward_head(features, pre_logits=True)` for embeddings, not CLS token. Embedding dim = 1536 for Large model.

**Estimated time**: ~30 min on 1x H200 (2105 images × 16 tiles = 33,680 forward passes)

**Checkpoint**: `convnext_large_phase2/model_best.pth.tar` (or `last.pth.tar` if still training)

#### Step 2: Aggregate ConvNeXt tiles → image-level probabilities

Reuse `Exp-8/aggregate_tiles.py` with:
- temperature=1.5
- aggregation=max_prob
- Single scale (4x4 only)

```
Input: scale_4x4.parquet (ConvNeXt)
Output: convnext_aggregated.parquet with columns:
  - image_name, image_probs_f16 [7806]
```

#### Step 3: Load existing ViT aggregated probabilities

Two options for ViT baseline:

**Option A — Fair comparison (ViT 4x4 only)**:
Re-aggregate ViT tiles from existing 4x4 parquet only (no 5x5, 6x6).
This isolates the ensemble effect from multi-scale advantage.

**Option B — Best ViT baseline (full pipeline)**:
Load the Exp-9 best aggregated output (4x4+5x5+6x6 + kNN + temporal fusion).
This tests whether ConvNeXt adds value ON TOP of the fully optimized ViT pipeline.

**Recommendation**: Do both. Start with Option A for clean comparison, then test Option B.

#### Step 4: Ensemble at probability level

```python
P_ensemble = w * P_vit + (1 - w) * P_convnext
```

Sweep `w` from 0.1 to 0.9 in steps of 0.1.

#### Step 5: Generate submission + evaluate

For each weight `w`:
1. Apply geo_mask (zero out species outside SW Europe)
2. Apply threshold (start with 0.100)
3. Generate submission CSV
4. Submit to Kaggle

Also sweep threshold: [0.05, 0.075, 0.100, 0.125, 0.150]

---

## What We Learn From Each Outcome

| Result | Interpretation | Next Step |
|--------|---------------|-----------|
| Option A ensemble > ViT 4x4 alone | Architecture diversity helps, even at same scale | Proceed with full multi-scale ConvNeXt |
| Option B ensemble > ViT full pipeline | ConvNeXt adds value on top of optimized ViT | Full integration worth the effort |
| No improvement in either | ConvNeXt errors too correlated with ViT | Reconsider ensemble strategy |

---

## Files Needed

### Existing (reuse):
| File | Source | Purpose |
|------|--------|---------|
| `extract_multiscale_embeddings.py` | Exp-7 | Template for ConvNeXt extraction |
| `aggregate_tiles.py` | Exp-8 | Tile → image aggregation |
| `generate_submission.py` | Exp-8 | Submission CSV generation |
| `temporal_fusion.py` | Exp-8 | Location prior blending |
| `utils/io_bytes.py` | Exp-8 | float16 encode/decode |
| `class_to_species_id.json` | PACE data | Class index → species ID |
| `sw_europe_mask_bool.npy` | PACE data | Geo filter mask |

### New (to create):
| File | Purpose |
|------|---------|
| `extract_convnext_embeddings.py` | Already exists in Exp-10 — adapt tile_size for 518 |
| `ensemble_probabilities.py` | Load 2 parquets, blend probs, sweep w, generate submissions |
| `slurm/extract_4x4.sbatch` | SLURM job for ConvNeXt 4x4 extraction |
| `slurm/ensemble_sweep.sbatch` | SLURM job for weight + threshold sweep |

### Data on PACE:
| Path | Content |
|------|---------|
| `${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale/` | ViT tile parquets (4x4, 5x5, 6x6) |
| `${PLANTCLEF_ROOT}/aerten3/checkpoints/exp10_convnext/convnext_large_phase2/` | ConvNeXt-L checkpoint |
| `${PLANTCLEF_ROOT}/aerten3/submissions/exp9_knn_diff_beta/` | ViT best submission outputs |
| Test images | 2105 quadrat images (3000x3000) |

---

## Pipeline Flow

```
QUICK TEST (4x4 only):

  ConvNeXt-L DINOv3 checkpoint
         |
         v
  Test images (2105 × 3000x3000)
         |
    [4x4 tiling: 16 tiles/image]
         |
         v
  ConvNeXt forward pass (518x518 tiles)
    → teacher_logits_f16 [7806]
    → embedding_orig_f16 [1536]
         |
         v
  scale_4x4_convnext.parquet
         |
    [max_prob aggregation, T=1.5]
         |
         v
  convnext_image_probs.parquet ----+
                                   |
                                   v
  vit_image_probs.parquet ---------> weighted average
  (from Exp-8/9)                   P = w·P_vit + (1-w)·P_cnxt
                                   |
                                   v
                            [geo_mask + threshold]
                                   |
                                   v
                            submission.csv → Kaggle
```

---

## Estimated Compute Cost

| Step | GPUs | Time | SUs |
|------|------|------|-----|
| ConvNeXt 4x4 extraction | 1x H200 | ~30 min | ~0.3 |
| Aggregation + ensemble sweep | CPU only | ~10 min | ~0 |
| **Total** | | **~40 min** | **~0.3 SUs** |

Negligible SU cost — this is a very cheap experiment.

---

## Success Criteria

- **Minimum**: Option A ensemble beats ViT 4x4 alone by ≥0.005 F1
- **Target**: Option B ensemble beats ViT full pipeline (F1 > 0.418)
- **Stretch**: Ensemble F1 > 0.43

---

## After Quick Test (if successful)

1. Full multi-scale ConvNeXt extraction (4x4 + 5x5 + 6x6, 77 tiles)
2. Build ConvNeXt FAISS index for kNN
3. Independent ConvNeXt pipeline (aggregate → kNN → temporal fusion)
4. Ensemble after full post-processing
5. Joint kNN with concatenated embeddings [768 + 1536 = 2304 dim]
6. Re-tune all hyperparameters (alpha, beta, threshold, w)
