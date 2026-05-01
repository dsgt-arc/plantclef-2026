# Exp-11C: ViT-L Ensemble (parallel to Exp-11B but with the new DINOv2 ViT-L)

## What this is

Exp-11B is the current production ensemble: ConvNeXt-L (w=0.23) + Exp-9B ViT-B + LoRA + ArcFace + kNN (w=0.77) → **F1 = 0.44428**.

Exp-11C swaps the **ViT branch backbone** from ViT-Base (Exp-9B) to the freshly fine-tuned **DINOv2 ViT-L** from Exp-13 (val EMA top-1 = 77.53, vs ViT-B's ~76.16 on the same val) and re-runs the ensemble pattern.

## Why

- ViT-L is +1.37 EMA top-1 above ViT-B's 100-epoch peak. Standalone gain expected on Kaggle public LB.
- Embedding dim is 1024 (vs 768 for ViT-B), which requires re-indexing the kNN FAISS retriever — see `kNN re-indexing` below.

## Plan (in order)

### Step 0 — Multiscale extraction at 5 scales (3, 4, 5, 6, 7)
**Lives in Exp-13** (`user/alper/src/Exp-13_ViT_finetuning/slurm/extract_multiscale_3to7.sbatch`).

```bash
cd user/alper/src/Exp-13_ViT_finetuning
sbatch --export=MODEL=dinov2_large,EXP_SUFFIX=phase2v2 slurm/extract_multiscale_3to7.sbatch
```

Output: `/storage/.../aerten3/parquet/exp13_vit_dinov2_large_phase2v2/scale_{3..7}x{3..7}.parquet`

### Step 1 — Standalone ViT-L F1 confirmation (no kNN, no ConvNeXt) — **matched ViT-B comparison**

Run two parallel aggregations using identical parameters (alpha=0.15, T=1.5, image_topk=64) so the only difference between ViT-L and ViT-B is the model:

| Scheme | Tiles | sbatch |
|---|---|---|
| **A (4)** = 4 only (single-scale baseline) | 16 | both |
| **E (456)** = 4+5+6 | 77 | both |
| **J (3456)** = 3+4+5+6 | 86 | both |
| **K (4567)** = 4+5+6+7 | 126 | both |
| **L (34567)** = 3+4+5+6+7 (NEW) | 135 | both |

```bash
# After ViT-L extraction completes (5 scales 3..7 from Exp-13/extract_multiscale_3to7.sbatch):
sbatch --export=MODEL=dinov2_large,OUTPUT_TAG=phase2v2 slurm/aggregate_vitL_standalone.sbatch
sbatch                                                slurm/aggregate_vitB_standalone.sbatch
```

Each generates 16 CSVs (4 schemes × 4 thresholds). Submit matching pairs to Kaggle for direct ViT-L vs ViT-B comparison.

**Existing ViT-B published numbers (from `Exp-7/docs/Exp7_Results_Summary.md`, NOT alpha-matched to this run):**
| Scheme | Tiles | Published F1 (ViT-B) | Config |
|---|---|---|---|
| A (4 only) | 16 | 0.35698 | Exp-3/4 baseline |
| E (456) | 77 | **0.40100** | thr=0.100, **alpha=0.05** |
| J (3456) | 86 | 0.38227 | thr=0.075, **alpha=0.60** |
| K (4567) | 126 | 0.38542 | thr=0.080, **alpha=0.60** |
| L (34567) | 135 | (no reference) | — |

These are NOT directly comparable to the new ViT-L runs because each scheme used a different alpha. The matched ViT-B aggregation in `aggregate_vitB_standalone.sbatch` uses the same alpha=0.15 across all schemes — that gives the apples-to-apples comparison.

> **Note:** the 0.42267 number that appears in `Exp-11B/docs/Exp11B_Plan.md` is **NOT** standalone ViT-B — it is the Exp-9B ViT-B + LoRA + ArcFace + kNN retriever output, which combines the classifier softmax with a kNN vote. The fair comparison for standalone ViT-L (no kNN) is the Exp-7 number above (~0.40 for scheme E).

**Expected outcome**: ViT-L beats ViT-B by 0.01–0.03 F1 on each matched scheme. The winner among E/J/K/L for ViT-L tells us which tile scheme to use in the ensemble (Step 3).

### Step 2 — kNN re-index on ViT-L embeddings (TBD; depends on Step 1 result)
The Exp-9B ArcFace+kNN FAISS index was built from **ViT-B embeddings (768-d)** and **cannot be reused** for ViT-L (1024-d). To use kNN with ViT-L, we'd need to:
- Re-train the LoRA + ArcFace head on top of fine-tuned ViT-L (Exp-9B's recipe), OR
- Skip ArcFace and just build a plain FAISS index over the new 1024-d ViT-L embeddings (faster, lower quality).

We will only do this if Step 1 shows ViT-L's standalone F1 is high enough to justify wiring it through the kNN path.

### Step 3 — Ensemble blend with ConvNext (matches Exp-11B pattern)
Once ViT-L standalone F1 is known:
- If ViT-L > ViT-B: increase ViT weight `w` in the blend (current Exp-11B uses w=0.77 for ViT branch).
- Sweep `w in {0.70, 0.75, 0.80, 0.85}` against fixed ConvNext branch.

`slurm/ensemble_vitL_with_convnext.sbatch` (TBD — created after Step 1 confirms ViT-L is worth ensembling).

## What's reused vs new

| Component | Source | Status |
|---|---|---|
| Aggregation pipeline (max_prob, temporal fusion) | Copied from Exp-7 | Same |
| Submission generator + threshold sweep | Copied from Exp-7 | Same |
| Ensemble blending logic | Copied from Exp-11 | Same |
| ViT-L embedding extraction | Exp-13 | Already trained |
| ConvNext-L tile parquets | Exp-10 / Exp-11 | Reused |
| ArcFace + kNN retriever | Exp-9B (ViT-B based) | **Cannot reuse — needs re-index for 1024-d** |
| `VIT_EMB_DIM` constant in scripts | `768 → 1024` | Patched in Exp-11C copies |

## Files in this experiment

| File | Origin | Modified? |
|---|---|---|
| `aggregate_tiles.py` | Exp-7 | No |
| `multiscale_aggregate.py` | Exp-7 | No |
| `temporal_fusion.py` | Exp-7 | No |
| `generate_submission.py` | Exp-7 | No |
| `threshold_sweep.py` | Exp-7 | No |
| `utils/` | Exp-7 | No |
| `ensemble_with_knn.py` | Exp-11 | No (logits only, dim-agnostic) |
| `ensemble_tile_knn.py` | Exp-11 | **Yes** — `VIT_EMB_DIM = 1024` |
| `run_knn_on_merged.py` | Exp-11 | **Yes** — `VIT_EMB_DIM = 1024` |
| `merge_tiles.py` | Exp-11 | No |
| `ensemble_multiscale.py`, `ensemble_optimized.py`, `ensemble_sweep.py` | Exp-11 | No |
| `slurm/aggregate_vitL_standalone.sbatch` | New | — |

## Verification

1. After Step 0 completes: confirm 5 parquets exist with expected tile counts (9, 16, 25, 36, 49 tiles/image).
2. After Step 1: compare CSV submissions for schemes E and K vs ViT-B baseline F1 of 0.42267.
3. Decision tree (vs ViT-B scheme-E baseline 0.39997):
   - ViT-L scheme E ≥ 0.42 → strong win, proceed to ensemble integration with ConvNext.
   - 0.40 ≤ ViT-L scheme E < 0.42 → modest win, ensemble worth trying but expectations lower.
   - ViT-L scheme E < 0.39997 → regression; investigate before proceeding (likely tile-config or threshold issue, not the model).
   - Scheme L (3+4+5+6+7) > scheme E → use L; otherwise use E (whichever wins also informs which ConvNext tile scheme to match for the ensemble blend).
