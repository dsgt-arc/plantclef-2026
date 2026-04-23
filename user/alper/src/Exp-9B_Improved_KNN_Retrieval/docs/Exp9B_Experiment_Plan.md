# Exp-9B: Improved kNN Retrieval via ArcFace-Tuned Embeddings

## Executive Summary

Replace the frozen DINOv2 retriever in the Exp-9 pipeline with an ArcFace-tuned
embedding head. The DINOv2 backbone stays frozen; only a small projection head is
trained with ArcFace (angular margin) loss over the 7806 species labels on the
cached training embeddings. Goal: a sharper, more species-discriminative
retrieval geometry that allows the kNN branch to contribute more signal to the
ensemble.

Target: **final ensemble F1 ≥ 0.42254** (current best 0.41754 + 0.005).

## Why This Experiment

Exp-9 established that the classifier + kNN ensemble yields F1=0.41754 at
β_multi=0.80, β_single=1.00. The single-visit branch currently gets *zero* kNN
contribution because the retriever's neighborhoods are too noisy — kNN adds more
error than signal when there is no temporal prior safety net.

The retriever is frozen off-the-shelf DINOv2. It was not trained with any
objective aligned to cosine-similarity retrieval. ArcFace makes the training
objective (angular margin in normalized embedding space) match what FAISS
actually queries (inner product on L2-normalized vectors = cosine). Same-species
images are pulled together, different-species images pushed apart by a
guaranteed angular margin.

If this works, two things happen:
1. kNN-only F1 on the leaderboard goes up meaningfully.
2. β_single can drop below 1.00, unlocking kNN contribution on a population it
   currently cannot touch.

## Scope and Non-Goals

**In scope (Tier 1 only):**
- Freeze DINOv2 backbone entirely.
- Train only a projection head (Linear 768→512 → BN → ReLU → Linear 512→512)
  and a sub-center ArcFace classifier on cached 768-d CLS embeddings.
- Re-project train and test embeddings through the trained head to 512-d.
- Rebuild FAISS index at 512-d, re-run the existing kNN retrieval + ensemble
  pipeline with the new retriever.

**Out of scope (explicitly):**
- No changes to the main softmax classifier. It keeps using the original
  frozen-DINOv2 + classification head outputs (cached teacher logits from Exp-7).
- No backbone unfreezing, no LoRA, no full finetune. Those are Tier 2+ and only
  considered if Tier 1 shows signal.
- No changes to the tile generation, geo mask, temporal fusion, or thresholding
  logic. The only thing that changes is what embedding is fed to FAISS.

## Pipeline Delta

Before (Exp-9):
```
image -> tiles -> DINOv2 -> 768-d CLS
                              |
                    +---------+----------+
                    |                    |
            classifier head       FAISS IndexFlatIP (768-d)
            -> 7806 logits        -> kNN votes (p_knn)
            -> p_softmax
                    \                    /
                     +-- beta blend -----+
                              |
                         p_final
```

After (Exp-9B):
```
image -> tiles -> DINOv2 -> 768-d CLS
                              |
                    +---------+---------------------------+
                    |                                     |
            classifier head       NEW ArcFace projection head (768 -> 512)
            -> 7806 logits        -> 512-d L2-normalized
            -> p_softmax          -> FAISS IndexFlatIP (512-d)
                                  -> kNN votes (p_knn)
                    \                                     /
                     +------------ beta blend ------------+
                              |
                         p_final
```

Everything left of "beta blend" in the classifier branch is byte-identical to
Exp-9. Everything in the retriever branch is replaced.

## Training Design (Tier 1)

### Model

```python
class ArcFaceHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=512):
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn  = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return F.normalize(self.fc2(self.act(self.bn(self.fc1(x)))), dim=1)
```

Sub-center ArcFace classifier (K=3 sub-centers per class) on top of the head
output. Sub-centers act as a small hedge against noisy labels by letting each
class have multiple prototypes; the closest sub-center is treated as the
effective class center.

### Loss

Sub-center ArcFace with:
- `s = 30` (logit scale, standard for face-recognition-style retrieval)
- `m = 0.3` rad angular margin on the true class
- Margin warmup: `m` linearly ramps 0 -> 0.3 over epoch 1, then stays at 0.3
- `K = 3` sub-centers per class

Label smoothing: `eps = 0.1` to keep the softmax from collapsing on known-noisy
PlantCLEF labels.

### Data

- Source: pre-extracted 768-d CLS embeddings at
  `${PLANTCLEF_ROOT}/embeddings/train/sample_id={0..19}/*.parquet`
  (same parquets used by Exp-9 `build_faiss_index.py`).
- Schema: `image_name` (str), `species_id` (int32), `cls_embedding` (float32 768d).
- All ~1.4M rows used (no geo filter at training time — the ArcFace head should
  learn discriminative structure over all species, filtering happens at
  index-build time as in Exp-9).
- Species-to-class index via `class_to_species_id.json` (same file Exp-9 uses).

### Optimization

- Optimizer: AdamW, `lr = 1e-3`, `weight_decay = 1e-4`.
- Scheduler: cosine LR with 500-step linear warmup, decays to `lr * 0.01` at end.
- Batch size: 16384 (embeddings are 768-d floats; a batch fits easily on one GPU).
- Epochs: 5.
- Mixed precision (`torch.cuda.amp.autocast` + `GradScaler`).
- Seed: 42. Deterministic DataLoader shuffle.

### Expected runtime

Cached embeddings, so there is no image decoding and no ViT forward pass. One
epoch over 1.4M rows at batch 16k = ~85 steps. Each step is a matmul on a
batch_size x 768 tensor plus an ArcFace classifier matmul against `7806 x 3 x 512`
weights. On one RTX 6000 / H200, 5 epochs should finish in **~30-90 minutes**.

If convergence is clean at 3 epochs, early stopping cuts it further.

### Sanity-check metrics during training

Logged every N steps:
- Training loss.
- Within-batch same-class mean cosine vs different-class mean cosine (spread).
- Rank-1 retrieval accuracy on an in-batch query (query = first half, gallery =
  second half of the batch). A simple train-side proxy — not val.

No held-out validation set; see note in Validation section.

## Re-Projection

Script: `project_embeddings.py`.

Two passes:
1. Train embeddings: for each `sample_id=*` parquet, load 768-d cls_embedding,
   forward through the trained ArcFace head, write a parallel parquet with a new
   `emb512` column (float32 512-d, L2-normalized). Same partition layout.
2. Test tile embeddings: the multi-scale test tile parquets (produced by Exp-7,
   consumed by Exp-9 `multiscale_aggregate_knn.py`) carry `embedding_orig_f16`
   in their schema. Read those, forward through the head, write parallel
   parquets with `emb512_f16`.

Implementation notes:
- Batch size 32768 on GPU, ~minutes end-to-end for 1.4M train + all test tiles.
- Head is in `.eval()` mode during projection.
- Output stored under `${PLANTCLEF_ROOT}/aerten3/exp9b_knn/embeddings_arcface/`.

## FAISS Index Build

Script: `build_faiss_index_512.py` (adapted from Exp-9 `build_faiss_index.py`).

Changes from Exp-9:
- Reads the projected 512-d parquets instead of the 768-d originals.
- Hardcodes `dim = 512` in `IndexFlatIP` construction.
- Otherwise identical: same geo-mask filtering (SW Europe, ~4800 species),
  same L2-normalize (redundant but harmless — already normalized by the head),
  same `max_per_class` support, same sanity check.

Output:
- `${PLANTCLEF_ROOT}/aerten3/exp9b_knn/faiss_index/faiss_index_arcface_geo.bin`
- `${PLANTCLEF_ROOT}/aerten3/exp9b_knn/faiss_index/faiss_labels_arcface_geo.npy`
- `${PLANTCLEF_ROOT}/aerten3/exp9b_knn/faiss_index/class_counts_arcface_geo.npy`

## Execution Plan — Two Steps

### Step 1: kNN-only comparison (no ensemble)

Purpose: isolate the retriever change. Decide go/no-go before any ensemble
sweeps.

Procedure:
1. Submit old retriever standalone: run `multiscale_aggregate_knn.py` with
   `beta_multi=0.0` and `beta_single=0.0` (pure kNN) using the existing Exp-9
   768-d index. This produces a pure-kNN submission on the original retriever.
   Score it: **F1_old_knn**.
2. Submit new retriever standalone: same pipeline, `beta = 0.0`, but pointed at
   the new 512-d ArcFace index. Score it: **F1_new_knn**.
3. Compare.

Decision rule:
- `F1_new_knn > F1_old_knn` by more than leaderboard noise: proceed to Step 2.
- `F1_new_knn <= F1_old_knn`: retriever is not better. Stop. Do not proceed to β
  re-sweep. Re-examine training hyperparameters (margin, scale, epochs) or abort.

Submissions: 2.

### Step 2: Ensemble with β re-sweep

Only if Step 1 passed.

Procedure:
1. Sanity-check submission: new retriever with β_multi=0.80, β_single=1.00
   (Exp-9 optimal). Should not catastrophically regress. Call result F1_sanity.
2. Coarse β grid (one leaderboard submission each):
   - β_multi in {0.50, 0.65, 0.80, 0.95}
   - β_single in {0.75, 0.85, 0.95, 1.00}
   - 16 submissions total.
3. Identify peak. Fine-grid ±0.05 around it in both dimensions (up to 9 more
   submissions).
4. Final: best-β submission compared against Exp-9's F1=0.41754 baseline
   submission (re-submit Exp-9 baseline fresh on same day to neutralize any
   leaderboard-side changes).

Submissions: ~26 across two days, well under the 100/day cap.

Success criteria:
- Primary: best Exp-9B ensemble F1 ≥ Exp-9 baseline + 0.005.
- Secondary: β_single drops below 1.00 at the optimum — evidence the retriever
  earned single-visit contribution.

## Validation Philosophy

No labeled held-out validation set exists. Leaderboard submissions (100/day cap)
are the evaluation oracle. Every F1 number in this plan is a leaderboard F1.
No proxy val set is built; the distribution mismatch cost is higher than the
submission cost at this cap.

Training-side diagnostics (loss curves, in-batch rank-1, intra- vs inter-class
cosine separation) are used only to decide whether the trained head looks
reasonable enough to bother submitting. They are not F1 estimates.

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| ArcFace head overfits to clean train distribution, hurts retrieval on messy test tiles | Medium | Step 1 pure-kNN comparison catches this before any β sweep |
| 512-d is too narrow and compresses useful information | Low | Projection head output dim is a hyperparameter; if Step 1 fails, try out_dim=768 with same recipe (no dim reduction) before abandoning |
| Sub-center K=3 destabilizes training | Low | Drop to K=1 (standard ArcFace) if training loss diverges |
| Optimal β doesn't change relative to Exp-9 | Medium | This still succeeds if the F1 at Exp-9's β is higher — the sharper retriever is carrying the same weight with more signal |
| Leaderboard F1 variance swamps the +0.005 target | Medium | Baseline re-submission on the same day as final; only compare same-day submissions |
| Test tile parquets do not carry raw 768-d embeddings, only float16 | Low | Schema check up front; if only f16 present, load as f32 before head forward pass |

## Directory Layout (this experiment)

```
Exp-9B_Improved_KNN_Retrieval/
├── docs/
│   └── Exp9B_Experiment_Plan.md            (this file)
├── train_arcface_head.py                   (Tier 1 training)
├── project_embeddings.py                   (apply head to train + test)
├── build_faiss_index_512.py                (build 512-d FAISS index)
├── slurm_train_arcface.sbatch              (GPU job, 1-2 hours)
├── slurm_project_embeddings.sbatch         (GPU job, ~minutes)
└── slurm_build_faiss_512.sbatch            (CPU job, ~30 min)
```

Reused from Exp-9 (no duplication):
- `knn_retriever.py` — works at any embedding dim as long as the FAISS index
  matches.
- `multiscale_aggregate_knn.py` — feed it the 512-d index path and the
  projected test embeddings; code doesn't care about dim.
- β sweep sbatch files — same sweeps, different index path argument.

## PACE Paths

```bash
PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
EXP9B_DIR="${PLANTCLEF_ROOT}/aerten3/exp9b_knn"

# Trained ArcFace head checkpoint
CHECKPOINT_DIR="${EXP9B_DIR}/arcface_head"

# Projected embeddings
TRAIN_EMB_ARCFACE="${EXP9B_DIR}/embeddings_arcface/train"
TEST_EMB_ARCFACE="${EXP9B_DIR}/embeddings_arcface/test"

# FAISS index (512-d)
FAISS_DIR="${EXP9B_DIR}/faiss_index"
```

## Submission Budget Summary

| Phase | Submissions | Purpose |
|-------|-------------|---------|
| Exp-9 baseline re-submit | 1 | Anchor for comparison |
| Step 1 pure-kNN old | 1 | Old retriever standalone |
| Step 1 pure-kNN new | 1 | New retriever standalone |
| Step 2 sanity-check | 1 | New retriever at Exp-9 optimal β |
| Step 2 coarse grid | 16 | β_multi x β_single 4x4 |
| Step 2 fine grid | ~9 | ±0.05 around peak |
| Buffer | ~4 | Debug / retries |
| **Total** | **~33** | Fits single-day cap of 100 |

## Exit Criteria

**Ship the new retriever if:**
- Final Exp-9B best β submission > Exp-9 baseline + 0.005 on the same day.
- Step 1 confirmed pure-kNN improvement (defensive: ensures we're not just β-tuning away a regression).

**Keep Exp-9 retriever otherwise:**
- Step 1 fails → abort before any sweep.
- Step 2 best β < Exp-9 + 0.005 → use Exp-9 retriever in the final ensemble.
  Optionally escalate to Tier 2 (LoRA) only if the Tier 1 best was at least
  +0.002 (indicates signal exists but is capped by the head-only bottleneck).
