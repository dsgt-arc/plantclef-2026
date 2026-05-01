# Exp-9B Session Context Summary

Full working context from the session that created Exp-9B. Captures the
reasoning, decisions, and results so future sessions can pick up cold.

## 1. Starting point

Exp-9's kNN retrieval ensemble achieved **F1=0.41754** on the PlantCLEF 2026
leaderboard with this pipeline:

```
image -> tiles (4x4 + 5x5 + 6x6 = 77 tiles/image, JPEG85 preprocessed)
      -> frozen DINOv2 ViT-Base -> 768-d CLS + 7806 logits per tile
                                     |
            +------------------------+------------------------+
            |                                                 |
   softmax(logits / T=1.5)               FAISS IndexFlatIP (768-d) -> kNN vote
            |                                                 |
            +----- beta blend (b_multi=0.80, b_single=1.00) --+
                                  |
                        max_prob across tiles
                                  |
                     alpha=0.15 temporal fusion
                                  |
                          thr=0.100, topk=10
```

The retriever branch was using **frozen off-the-shelf DINOv2** (not
contrastively trained). The classifier branch was also using frozen DINOv2
with the timm classification head.

## 2. Question that started the experiment

**"Would a contrastively-trained retriever improve the ensemble?"**

Key observation motivating the investigation: Exp-9's optimum has
`beta_single=1.00`, meaning **single-visit images get zero kNN contribution**.
kNN is currently only useful on multi-visit images where the temporal prior
acts as a noise filter. A sharper, species-discriminative retriever could
potentially earn `beta_single < 1.00` and unlock a population of test images
that currently gets no benefit from the retriever.

Target: **final ensemble F1 >= 0.42254** (0.41754 + 0.005).

## 3. Approach chosen: ArcFace metric learning on DINOv2

ArcFace adds an angular margin to the softmax cross-entropy classifier such
that the training objective (cosine discrimination between classes) matches
what FAISS actually queries (cosine similarity on L2-normalized vectors).
This is the standard recipe for fine-grained retrieval (face recognition,
iNat, Google Landmarks).

Two tiers considered:

- **Tier 1 — head-only ArcFace on cached CLS embeddings.** Trains only a
  projection head on top of the frozen DINOv2 CLS. Hours of training on one
  GPU. Low cost, quick go/no-go signal.

- **Tier 2 — LoRA adapters on DINOv2 attention layers + ArcFace head.**
  Trains low-rank adapters that modify how the ViT computes CLS itself.
  Overnight training on one GPU. Higher ceiling.

Chosen strategy: run Tier 1 first. If retriever shows pure-kNN signal,
proceed to Step 2 (ensemble beta sweep) and/or Tier 2.

## 4. Tier 1 results

### Training (v1: 768 -> 512, epochs=5)

First run had a schedule bug (`--warmup-steps=500` but total=425). Rerun with
`epochs=15, warmup-steps=100` converged cleanly:

```
Final metrics (step 1260/1275):
  loss:           4.47  (from 9.57 at step 0)
  in-batch r1:    0.495 (from 0.351)
  intra cosine:   0.457 (same-species)
  inter cosine:   0.011 (different-species)
  gap:            0.446 (doubled from initial 0.21)
```

Loss flattened over the last 4 epochs, rank-1 plateaued. LR decayed cleanly.
Training took ~4 minutes wallclock on RTX 6000.

### Step 1 leaderboard result (512-d ArcFace vs 768-d frozen DINOv2)

Pure kNN, beta=0, alpha=0.15, thr=0.100:

| Retriever | Dim | Leaderboard F1 |
|---|---|---|
| Old (frozen DINOv2) | 768 | 0.31328 |
| New (ArcFace Tier 1) | 512 | 0.31634 |
| Delta | | **+0.00306** |

The new retriever wins — ArcFace training extracts real signal even with
768->512 compression. Clear go-signal for Step 2 beta sweep.

### Dim-reduction confound

Step 1's comparison mixed two variables: retriever geometry (ArcFace vs
frozen) AND dimensionality (512 vs 768). To isolate: user requested
retraining at `out_dim=768` so the retriever has the same dim as the original
frozen DINOv2 baseline. Retraining at 768-d is scaffolded but not yet run.

## 5. Tier 2 decision

User opted to proceed with Tier 2 directly rather than run the 768-d Tier 1
redo first. Keeping output dim at 768 for Tier 2.

## 6. File inventory

### Tier 1 training (already run, ready for re-use)

| File | Purpose |
|---|---|
| `train_arcface_head.py` | Tier 1: MLP projection head + sub-center ArcFace on cached CLS |
| `slurm_train_arcface.sbatch` | RTX 6000, 3h budget, currently configured for `--out-dim 768` |
| `project_embeddings.py` | Forward cached 768-d CLS through Tier 1 head; --mode train / --mode test |
| `slurm_project_embeddings.sbatch` | GPU, runs both train and test projection |
| `build_faiss_index_512.py` | Dim-agnostic FAISS IndexFlatIP builder (misnamed; works at any dim) |
| `slurm_build_faiss_512.sbatch` | CPU, ~1 min |

### Tier 2 training (scaffolded, not yet run)

| File | Purpose |
|---|---|
| `lora.py` | LoRALinear wrapper, `inject_lora_into_attention`, `freeze_non_lora` helpers |
| `train_arcface_lora.py` | Tier 2: DINOv2 frozen + LoRA on attn.qkv (all 12 blocks) + ArcFace head 768-d + sub-center classifier |
| `slurm_train_lora.sbatch` | H200, 16h budget, batch=64, 3 epochs, LoRA r=8 alpha=16 |
| `extract_embeddings_lora_train.py` | Forward 1.4M train images through LoRA backbone + head |
| `slurm_extract_lora_train.sbatch` | H200, 4h budget |
| `extract_embeddings_lora_test.py` | Re-tile test images, forward through LoRA, merge onto existing Exp-7 parquets |
| `slurm_extract_lora_test.sbatch` | RTX 6000, 4h budget, batch=64 |

### Inference / ensemble (copied + adapted from Exp-9)

| File | Purpose |
|---|---|
| `knn_retriever.py` | FAISS query + vote (dim-agnostic, imports from Exp-7 modules) |
| `compute_class_counts.py` | Compute class frequency from labels array |
| `multiscale_aggregate_knn.py` | Main sweep runner — added `--embedding-col` and `--embedding-dim` args |
| `slurm_step1_knn_only.sbatch` | 2-task array: pure kNN old vs new retriever |
| `slurm_knn_sweep.sbatch` | Standard beta/alpha/threshold sweep |
| `slurm_diff_beta_sweep.sbatch` | Array: beta_multi sweep with beta_single=1.00 |
| `slurm_ktau_sweep.sbatch` | Array: k sweep |
| `slurm_rare_species_sweep.sbatch` | Array: 6-way rare-species boosting test |

### Docs

| File | Purpose |
|---|---|
| `docs/Exp9B_Experiment_Plan.md` | Original experiment plan (Tier 1 focus) |
| `docs/Exp9B_Session_Context_Summary.md` | This file |

## 7. Critical design decisions (with reasoning)

### Architectural

- **FAISS type: `IndexFlatIP`** — exact brute-force on L2-normalized vectors
  = exact cosine similarity. At 1.2M vectors this is fast enough (~100 ms per
  batch query). No ANN (HNSW, IVF) because it introduces a confounding
  variable (recall vs speed tradeoff) and isn't the bottleneck.
- **Dim = 768** at user's request (was 512 in initial Tier 1 run) to avoid
  the dim-reduction confound in retriever-vs-retriever comparisons.
- **Sub-center ArcFace, K=3** as a cheap hedge against noisy labels in
  PlantCLEF training set.
- **ArcFace hyperparameters**: m=0.30, s=30.0, margin warmup over ~1 epoch,
  label smoothing=0.10. Standard face-recognition defaults.

### Training

- **Tier 1 head architecture**: Linear(768->out_dim) -> BN -> ReLU ->
  Linear(out_dim->out_dim) -> L2-normalize. Simple 2-layer MLP.
- **Tier 2 LoRA targets**: `attn.qkv` only (not `attn.proj`, not MLP). Rank
  8, alpha 16. Standard cheap-LoRA config.
- **Backbone fully frozen in Tier 2** via `freeze_non_lora()`. Only LoRA A/B
  matrices + head + ArcFace classifier are trainable (~20M params vs 86M
  frozen).
- **Gradient checkpointing ON by default** in Tier 2 training to fit batch=64
  at 518x518 on 24GB GPUs. Has a known gotcha with reentrant checkpointing +
  frozen-input tensors + LoRA-only gradients; defensive `requires_grad_()`
  on input is applied inside `forward_embeddings`.

### Data pipeline

- **Train images: no JPEG85 compression** (user instruction: train/test image
  types are different enough that matching doesn't help).
- **Test tiles: JPEG85 kept** because the existing `teacher_logits_f16`
  column in Exp-7 parquets was computed from JPEG85-preprocessed tiles, and
  we want per-row alignment between retriever embeddings and classifier
  logits within the same tile_index.
- **Training image resize**: `transforms.Resize(518)` (shorter-side) +
  `CenterCrop(518)`. Differs from test-time tiling which does
  `center_crop_square -> resize(N*518) -> tile`. Accepted mismatch because
  train vs test distributions are already different by nature of the task.

### Operational

- **Single GPU everywhere** — no DDP. ~8 h on H200 is short enough that
  multi-GPU scaffolding isn't worth the risk.
- **SLURM `cpus-per-task=6`** everywhere to respect the 6:1 CPU:GPU ratio
  cap seen on `gpu-rtx6000` (may also apply to H200).
- **`PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True`** in all training and
  extraction scripts — defensive against corrupt files in 1.4M-image
  training set.

## 8. PACE paths

```bash
PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
EXP9B_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn"

# Train-side data
TRAIN_IMAGES="${PLANTCLEF_ROOT}/train/PlantCLEF2024"                      # ImageFolder (species_id subdirs)
TRAIN_EMB_CACHED="${PLANTCLEF_ROOT}/embeddings/train"                     # 1.4M cached 768-d CLS (Tier 1 input)

# Test-side data
TEST_IMAGES="${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images"        # ~2105 raw test images
TEST_PARQUETS="${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp7_multiscale"  # scale_{4x4,5x5,6x6}.parquet

# Tier 1 outputs
TIER1_CKPT="${EXP9B_DIR}/arcface_head/arcface_head_final.pt"
TIER1_TRAIN_EMB="${EXP9B_DIR}/embeddings_arcface/train"
TIER1_TEST_EMB="${EXP9B_DIR}/embeddings_arcface/test"

# Tier 2 outputs
TIER2_CKPT="${EXP9B_DIR}/arcface_lora/lora_head_final.pt"
TIER2_TRAIN_EMB="${EXP9B_DIR}/embeddings_arcface_lora/train"
TIER2_TEST_EMB="${EXP9B_DIR}/embeddings_arcface_lora/test"

# FAISS indexes (same filename reused for new builds)
FAISS_DIR="${EXP9B_DIR}/faiss_index"
FAISS_INDEX="${FAISS_DIR}/faiss_index_arcface_geo.bin"
FAISS_LABELS="${FAISS_DIR}/faiss_labels_arcface_geo.npy"
FAISS_COUNTS="${FAISS_DIR}/class_counts_arcface_geo.npy"

# Shared
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
BACKBONE="${PLANTCLEF_ROOT}/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
```

## 9. Run sequences

### Tier 1 (already executed)

```bash
sbatch slurm_train_arcface.sbatch         # 5 min, trained at 512-d (done)
sbatch slurm_project_embeddings.sbatch    # 15-30 min
sbatch slurm_build_faiss_512.sbatch       # 1 min
sbatch slurm_step1_knn_only.sbatch        # Step 1 comparison (done: +0.003)
```

Current sbatches are configured for 768-d; rerunning the training step would
produce a Tier-1 768-d retriever.

### Tier 2 (scaffolded, not run)

```bash
sbatch slurm_train_lora.sbatch                    # H200 ~4-8h
sbatch slurm_extract_lora_train.sbatch            # H200 ~1-2h
sbatch slurm_extract_lora_test.sbatch             # RTX 6000 ~30-60 min
# Then patch build_faiss_index_512.sbatch to point at TIER2_TRAIN_EMB
# before running, or invoke build_faiss_index_512.py manually.
sbatch slurm_build_faiss_512.sbatch               # 1 min (after path patch)
sbatch slurm_step1_knn_only.sbatch                # Rerun Step 1 vs Tier 2 index
sbatch slurm_diff_beta_sweep.sbatch               # Step 2 beta re-sweep
```

## 10. Submission budget (100/day cap)

| Phase | Submissions | Status |
|---|---|---|
| Tier 1 Step 1 (old vs new kNN) | 2 | Done |
| Tier 1 Step 2 beta sweep (4x4 coarse + fine) | ~25 | Not yet run |
| Tier 2 Step 1 (old vs Tier-2 kNN) | 2 | Not yet run |
| Tier 2 Step 2 beta sweep | ~25 | Not yet run |
| **Total** | **~54** | Well under 100/day cap |

## 11. Calibrated expectations

Based on Tier 1's +0.003 pure-kNN delta and ensemble-math compression:

| Scenario | Pure-kNN F1 delta | Ensemble F1 delta after beta sweep |
|---|---|---|
| Tier 1 beta re-sweep (current retriever) | +0.003 (measured) | +0.001 to +0.003 |
| Tier 2 (expected) | +0.008 to +0.020 | +0.003 to +0.008 |
| Tier 2 + beta_single drops below 1.00 | upper bound | +0.005 to +0.010 |

Tier 1 alone is plausibly insufficient for the +0.005 ensemble target.
Tier 2 has a realistic chance.

## 12. Open items / things to verify

- **Training image folder layout**: `${PLANTCLEF_ROOT}/train/PlantCLEF2024`
  is assumed to be an ImageFolder with numeric species_id subdirs. If
  subdirs are scientific names instead, `ImageFolderWithSpeciesId` needs a
  different mapping path.
- **H200 CPU:GPU ratio**: set to 6:1 everywhere to respect the cap seen on
  gpu-rtx6000. Confirm H200 allows >=6 before running.
- **H200 queue availability**: 16h budget on gpu-h200 may queue slowly. If
  queue is long, fallback to gpu-rtx6000 with batch=32 (wallclock ~2x).
- **The `build_faiss_index_512.py` is dim-agnostic despite the name** — it
  reads dim from parquet column width. Rename to `build_faiss_index.py` at
  some point for clarity.
- **No resume-from-checkpoint** in Tier 2 training. If the 16h job times out
  mid-run, rerun from scratch (epoch checkpoints exist but aren't wired to a
  --resume flag).

## 13. What to watch in Tier 2 training logs

Healthy signals (based on Tier 1's final state):
- Loss should drop from ~9.5 at step 0 to ~3.0-4.0 by end of training.
- In-batch rank-1 should climb from ~0.35 (frozen DINOv2 baseline) to
  >=0.60 (higher than Tier 1's 0.495 — LoRA can rewrite CLS, not just
  project it).
- Intra-class cosine should climb above Tier 1's 0.457 plateau (indicating
  the backbone is actually pulling same-species embeddings tighter, not
  just spreading different-species apart).
- Inter-class cosine should stay low (~0.01 or lower).

Warning signals:
- `[model] lora_trainable=0.00M` at init -> LoRA injection failed silently,
  kill job.
- Loss oscillates without trending down -> LR too high (try 1e-4).
- Loss NaN -> check gradient scaler, possibly reduce LR, check batch data.
- In-batch rank-1 stuck near chance for many steps -> gradient-checkpoint
  gotcha may have bitten (gradients not flowing through LoRA). Kill, try
  `--no-grad-checkpoint`.

## 14. Pipeline decoupling that matters for future experiments

Key architectural property established in this experiment: **the classifier
and retriever are completely independent branches that only meet at the
beta blend**. Both must produce `p[7806]` distributions; everything else is
replaceable.

Practical consequences:
- Swap DINOv2 classifier for DINOv3 / Exp-13-fine-tuned / ConvNext at any
  time. Does not require re-indexing the retriever.
- Swap the retriever encoder (Tier 1, Tier 2, or something else entirely)
  without touching the classifier.
- Each classifier swap requires a fresh beta re-sweep. Each retriever swap
  requires re-indexing and fresh beta re-sweep.
- Ensemble decorrelation is maximized when classifier and retriever use
  different backbones and different training objectives — already the case
  here (frozen classifier, ArcFace retriever).

## 15. Key references back to Exp-9

- `Exp-9_kNN_Retrieval_Ensemble/docs/Exp9_Results_Summary.md` — complete
  results summary for the baseline. Beta sweep data, k/tau sweep, rare
  species strategies.
- Exp-9 best config at 0.41754 = reference point for all Exp-9B comparisons.
- Exp-9's rare-species work showed inv-freq weighting HURTS on the 768-d
  DINOv2 retriever. Worth retesting on the Tier-2 retriever since sharper
  neighborhoods could change that.
