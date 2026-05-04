# Proposal: TileQ-Decoder — Label-as-Query Multi-Label Decoding from Single-Label Plant Embeddings

**Status**: proposed
**Created**: 2026-05-02
**Author**: Murilo Gustineli

## Hypothesis

A small label-as-query transformer decoder, trained on synthetic multi-label
"pseudo-quadrats" assembled in CLS-embedding space, can replace the standard
per-tile max-probability aggregation used by every PlantCLEF working note since
2022. The headline ablation is a controlled comparison between two
pseudo-quadrat sampling strategies:

- **H1.** A label-as-query decoder over per-tile DINOv2 CLS embeddings, trained
  on synthetic pseudo-quadrats with Asymmetric Loss, produces a competitive
  multi-label classifier without modifying the frozen backbone.
- **H2.** Anchoring synthetic pseudo-quadrats to real test-quadrat tile CLSs
  (Sampler B), with FAISS-based species detection on those backgrounds, produces
  a more useful training distribution than random-background sampling
  (Sampler A), measured by macro-F1 on the test set after identical decoder
  training.

The contribution is methodological, not leaderboard-driven. A null result on
**H2** is publishable.

## Background

PlantCLEF 2026 is multi-label species identification on ~3000×3000 vegetation
quadrats over 7,806 species, scored by macro-averaged F1. Training data is
single-label single-plant (1.4 M Pl@ntNet images). The two open problems are:

1. **Single→multi-label domain gap.** The fine-tuned ViT-B DINOv2 classifier
   (organizers' checkpoint) was trained with single-label cross-entropy on
   isolated plants; test images contain ~5–10 species each in a cluttered scene.
2. **Global-vs-local feature representation at extreme resolution.** A test
   image is ~3000×3000 px; a single 518×518 ViT input cannot see the full image
   without downsampling, and small-but-discriminative species often live in a
   small fraction of the image.

The team's existing pipeline (cached) tiles each test image into a 4×4 grid of
518×518 tiles, runs the frozen DINOv2 ViT-B forward pass per tile, and
aggregates by per-class max across tiles (`max_prob`). My exp 002 reproduced the
2025 sorted-prob top-k=9 baseline at F1=0.30810. This experiment asks whether a
small attention-based decoder, trained on CLS-domain pseudo-quadrats, can do
better — and whether anchoring those pseudo-quadrats to the *test* domain is
the leverage point.

This is an **independent** experiment: it consumes only my own cached
parquets (exp 001, `shared/inference/dinov2-vit-base/`), the team-shared train
CLS embeddings (`shared/embeddings/train/`), the public ViT-B checkpoint, and
the raw test/train images. It does **not** consume any of Alper's exp 7/9/11
artefacts (multiscale parquets, FAISS index, ConvNeXt weights, geo mask).

## Method

### Approach

A **TileQ-Decoder** is a 3–5 M-parameter transformer head over a frozen ViT-B
DINOv2 backbone:

```
test image → 4×4 grid → 16 tiles × 518² → frozen ViT-B → 16 × 768 CLS tokens
                                                                   │
                                            ┌──────────────────────┘
                                            ▼
                          linear proj → 16 × 384 (keys/values)
                                            ▲
       64 learnable group queries (384) ────┘  cross-attn (8 heads)
                                            │
                                            ▼
                                    LayerNorm + FFN
                                            │
                                            ▼
                            group-FFN: einsum (B,G,D)·(G,D,C') → (B,G,C')
                                            │
                                            ▼
                                  reshape → 7806 logits
                                            │
                                            ▼
                                       Asymmetric Loss
```

Pseudo-quadrats for training are assembled **directly from cached CLS
embeddings**, not from pixel mosaics. Because each ViT tile is forwarded
independently in the existing pipeline (no cross-tile attention), a CLS computed
from a tile in a synthetic pixel mosaic is identical to the cached CLS of that
tile photographed alone. This removes any need for image compositing,
attention-rollout masking, or DINOv2 forward passes during training.

### Setup

| Component | Details |
|-----------|---------|
| Backbone | `vit_base_patch14_reg4_dinov2.lvd142m`, ~86 M params, **frozen** at all times. Public checkpoint at `shared/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth`. |
| Decoder head | TileQ-Decoder, ~4 M params; 64 group queries × 384 dim, 1 cross-attn layer (8 heads), 2-layer FFN, group-FFN classifier head |
| Train data (synthetic) | Pseudo-quadrats assembled from cached train CLS embeddings (`shared/embeddings/train/`, 1,408,033 × 768-d) |
| Test data | Cached test-tile CLS at `shared/inference/dinov2-vit-base/cls_grid_4x4.parquet` (to be produced; ~2 GPU-h via extended exp-001 inference) |
| Compute | 1× RTX 6000 (24 GB) or 1× H100 (80 GB) for ~4–8 GPU-h decoder training; 1× RTX 6000 for ~2 GPU-h test-CLS extraction; CPU node (≤8 h) for FAISS background map + train-lookup build |
| Dependencies | `torch>=2.1`, `timm>=1.0`, `faiss-cpu>=1.7`, `pyarrow`, `pandas`, `numpy`, `tqdm` |
| Code | `user/murilo/src/plantclef/` (shared library) + `experiments/003-tileq-decoder/` (experiment-specific scripts) |

### Procedure

Logical-dependency order (sections in `spec.md` are numbered to match):

1. **Test-CLS extraction.** Extend `experiments/001-dinov2-grid-ablation/inference.py` with a `--save-cls` flag; run for grid 4×4 → `shared/inference/dinov2-vit-base/cls_grid_4x4.parquet` (33,680 rows × 768-d float16).
2. **Train-CLS lookup.** Build `dict[species_id → np.ndarray (n_i, 768)]` once at training start (4.3 GB RAM) from the hive-partitioned train embeddings.
3. **Class-frequency weights.** Compute `w_s ∝ 1/√count[s]` per species; serialise to `artifacts/species_weights.npy`.
4. **A0 baseline submissions** (no training required). Reuse `experiments/002-classification-methods/classify.py`:
   - **A0a**: `max_prob` top-k=10 on `grid_4x4.parquet`.
   - **A0b**: sorted-prob top-k=9 (reproducible 2025 baseline, F1=0.30810).
5. **FAISS background map** (Sampler B prerequisite). L2-normalise all train + test CLS, build `IndexFlatIP`, query top-1 per test tile, threshold at τ chosen from the similarity-histogram shoulder. Output: `artifacts/tile_to_species.pkl`.
6. **Sanity checks.** Overfit-one-batch (loss → 0 in 1000 steps), shape audit, sampler unit tests for both A and B.
7. **Train Run 1 (Sampler A).** 80 k steps, AdamW, ASL. Save `artifacts/checkpoints/decoder_samplerA.pt`.
8. **Train Run 2 (Sampler B).** Same config, different sampler. Save `artifacts/checkpoints/decoder_samplerB.pt`.
9. **Inference.** Forward each test quadrat through both decoders → top-k decoding at **both k=9 (H1 anchor, matches A0b) and k=10 (organizers' convention)** from one forward pass → submission CSVs `submissions/A1_samplerA_topk{9,10}_seed42.csv` and `submissions/A2_samplerB_topk{9,10}_seed42.csv`.
10. **A3 (bonus).** α-blend decoder probs with cached `max_prob` baseline for α ∈ {0, 0.25, 0.5, 0.75, 1.0}; produces 5 CSVs.

### Variables

- **Independent**: pseudo-quadrat sampler (A vs B); optional α-blend weight.
- **Dependent**: Kaggle macro-F1 per submission, ablation deltas vs A0a/A0b.
- **Controlled**: backbone (frozen public ViT-B), tile schema (4×4 grid, 518² tiles, no JPEG round-trip, no Lanczos), decoder architecture, optimizer schedule, total training steps, seed.

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Kaggle macro-F1 | Public-leaderboard score per submission |
| Avg species/image | Distribution sanity (target ≈ 5–10) |
| Decoder val loss | Multi-label cross-entropy on 1 % synthetic held-out (training-loss sanity only, **not** generalisation) |

### Baseline

- **A0a** (`max_prob` top-10 on grid_4x4): expected ≈ 0.23 from organizers' notebook.
- **A0b** (sorted-prob top-9 on grid_4x4): F1=0.30810 (validated, exp 002).

### Success Criteria

H1 anchor is **A0b = 0.30810** (validated 2025 sorted-prob top-9 baseline). A0a
(`max_prob` top-10 = 0.22986) is the organizers'-approach reference, not the
anchor — beating A0a is necessary but not sufficient.

- **Confirm H1 if**: A1 (Sampler A) ≥ **0.30810** (A0b). Decoder beats the best-known aggregator on the same per-tile signal.
- **Confirm H2 if**: A2 (Sampler B) − A1 (Sampler A) ≥ +0.01 absolute F1.
- **Reject H1 if**: A1 < **0.288** (A0b − 0.02). Decoder fails to capture useful multi-label signal beyond the aggregator baseline.
- **Reject H2 if**: |A2 − A1| ≤ 0.005. Test-anchored sampling does not produce a more useful training distribution than naive sampling. **Publishable null result.**

## Limitations

- No real validation set with ground-truth labels — Kaggle is the only test signal. Synthetic-only val loss only confirms the model is fitting the training distribution.
- `n_positives = 5` is fixed across all pseudo-quadrats; the real test distribution has variance not modelled here.
- Single grid scale (4×4) — multiscale (Alper's 4+5+6) requires fresh extraction at additional scales and is out of scope.
- Sampler B's FAISS retrieval only assigns species to *test* tiles whose top-1 train-neighbour exceeds threshold τ; tiles below τ are treated as "true background" but may contain real species that lacked a near match in the train embedding space. This is acknowledged as a noise source the ASL hard-margin is meant to absorb.
- The frozen backbone is the organizers' single-label-trained ViT-B; any benefit from end-to-end fine-tuning is outside this experiment.
