---
document: "TileQ-Decoder — Paper Briefing"
target: "CLEF 2026 working notes paper"
experiment: "003-tileq-decoder"
author: Murilo Gustineli
date_drafted: 2026-05-17
purpose: Self-contained narrative briefing for Claude Research to draft the working-notes paper. Synthesizes proposal.md / spec.md / tasks.md / results.md into a single document. All numbers reproduced inline; no internal repo links to follow.
status: experiment complete (2026-05-06); paper unwritten
contribution: publishable null result on label-as-query decoding from CLS-domain pseudo-quadrats, plus a newly characterized failure mode (FAISS-undersampling under-confidence) not enumerated in the design spec.
---

# TileQ-Decoder — Paper Briefing

## TL;DR

We designed a small label-as-query transformer decoder (TileQ-Decoder, ~4.5 M
parameters) that runs over per-tile DINOv2 ViT-B CLS embeddings from a 4×4
grid of 518×518 tiles per ~3000×3000 vegetation quadrat. The decoder is
trained on synthetic multi-label "pseudo-quadrats" assembled directly in
CLS embedding space, using Asymmetric Loss. The headline experiment is a
controlled comparison between two pseudo-quadrat sampling strategies for
PlantCLEF 2026:

- **Sampler A (naive)** — 5 species CLSs from the training set, padded
  with 11 random-background train CLSs.
- **Sampler B (test-anchored)** — 5 species CLSs injected into 11 real
  test-quadrat tile CLSs, with FAISS top-1 retrieval used to label
  surviving backgrounds before they enter the loss.

Two falsifiable hypotheses, evaluated on Kaggle macro-F1:

- **H1.** The decoder beats the team's best aggregator (sorted-prob top-9
  on the same 4×4 tile grid, F1 = 0.30810) on the same backbone features.
- **H2.** Test-anchored Sampler B beats random-background Sampler A by
  at least +0.01 macro-F1.

**Outcome.** Both rejected. The decoder underperforms every aggregator
baseline computed from the same per-tile probabilities:

| Submission | Kaggle macro-F1 | vs anchor |
|---|---:|---:|
| A0a: cached `max_prob` top-10, grid 4×4 | 0.22986 | reference |
| **A0b: sorted-prob top-9, grid 4×4 (H1 anchor)** | **0.30810** | — |
| Murilo no-grid baseline (exp 001, full-image inference) | 0.14506 | reference |
| **A1@k9: TileQ-Decoder Sampler A, top-9 (H1 result)** | **0.12180** | **−0.186 vs A0b** |
| A1@k10: TileQ-Decoder Sampler A, top-10 | 0.11640 | |
| **A2@k9: TileQ-Decoder Sampler B (τ=0.65), top-9 (H2 contrast)** | **0.09060** | **−0.031 vs A1@k9** |
| A2@k10: TileQ-Decoder Sampler B (τ=0.65), top-10 | 0.08445 | |

**H1 rejected decisively** — A1=0.12180 is 0.186 F1 below A0b and 0.166
below the proposal's reject threshold of 0.288. The decoder is even
below the no-grid full-image baseline (0.14506), meaning it destroys
signal that already exists in the cached per-tile probabilities.

**H2 rejected, in the opposite direction the proposal anticipated** —
A2 − A1 = −0.031 macro-F1 (six times outside the noise band), with
Sampler B significantly *worse* than Sampler A.

**A3 α-blend** (`s = α · decoder + (1−α) · max_prob`, swept α ∈ {0, 0.1,
…, 1.0} at k=10) gives a **monotonically decreasing** Kaggle curve from
0.22144 (α=0, pure max_prob) to 0.11640 (α=1, pure decoder). No convex
combination of decoder and max_prob beats α=0. The decoder is not even
partially redundant; it actively destroys F1 at any non-zero weight.

**Two paper-worthy contributions:**

1. **A clean null result** on label-as-query multi-label decoding from
   CLS-domain pseudo-quadrats — the proposal explicitly framed this as
   publishable, and the result is the cleanest possible version (decoder
   below every baseline, both samplers fail, blend monotone-decreasing).
2. **A newly characterized failure mode** — *FAISS-undersampling
   under-confidence* — not in the spec's enumerated failure modes:
   when test-anchored sampling labels only a small fraction of test
   tiles (5.46 % at τ=0.65 here), the decoder learns "test-shaped tile
   → no species in label" → outputs uniformly low logits at inference →
   forced top-k decoding picks essentially noise. This is distinct from
   the synthetic-distribution overfit mode the spec did enumerate for
   Sampler A.

---

## 1. Context

### 1.1 PlantCLEF 2026

PlantCLEF 2026 is a multi-label species identification challenge on
~3000×3000 vegetation-quadrat photographs over 7,806 plant species,
scored by macro-averaged F1. The provided training data is
single-label single-plant: 1,408,033 Pl@ntNet images of isolated
specimens with one species label each. The test set is 2,105
quadrats, each containing roughly 5–10 species in cluttered scenes.

The challenge has two open problems:

1. **Single→multi-label domain gap.** The provided ViT-B DINOv2
   classifier was trained with single-label cross-entropy on isolated
   plants; the test images contain many species each, mostly small
   relative to image area, often partially occluded.
2. **Global/local representation at extreme resolution.** A
   ~3000×3000 quadrat cannot be passed through a 518×518 ViT input
   without losing the small-but-discriminative plants that drive
   F1. Tiling addresses this but introduces a per-tile aggregation
   problem.

### 1.2 Existing pipeline (team-shared)

The team's existing pipeline tiles each test image into a 4×4 grid of
518×518 tiles, runs the frozen DINOv2 ViT-B/14-reg4 (organizers'
2024-fine-tuned checkpoint, ~86 M params) on each tile, and aggregates
the 16 per-tile softmax outputs by per-class max across tiles
(`max_prob`). The cached per-tile probability tensor for the 2,105
test images at grid 4×4 is the input most of the PlantCLEF working
notes work on top of:

- `shared/inference/dinov2-vit-base/grid_4x4.parquet`,
  schema `(image_name: str, tile_idx: int32, probs: binary float16[7806])`,
  33,680 rows = 2,105 × 16.

The team-level best on this pipeline is F1 = 0.4379 (ConvNeXt-L DINOv3 +
ViT-B DINOv2 probability-level ensemble with multiscale 4+5+6 tiles,
ViT-only FAISS kNN, temporal `prior_blend`, geo mask, threshold 0.100,
topK 10).

### 1.3 Validated baseline reproduced (Murilo's exp 002)

Exp 002 (`user/murilo/experiments/002-classification-methods/`)
reproduced the 2025 working-notes sorted-prob top-9 baseline on
`grid_4x4.parquet` at F1 = 0.30810 — a 99.9 % exact match (2,103 / 2,105
rows identical) to the reference CSV `dsgt_run_topk_9_grid_4x4.csv`.
This identifies the 2025 method as **sorted-prob top-k = 9**
(rank species by softmax probability summed across tiles, pick top
9 per image) and gives a clean per-image anchor for any new method on
the same input.

Key prior from exp 002: **probability magnitude beats tile frequency.**
Sorted-prob top-9 at F1 = 0.30810 vs union (rank by count of tiles
nominating a species) at F1 0.19–0.21 on identical per-tile predictions.
Confidence matters more than agreement.

### 1.4 Independence boundary

This experiment was deliberately scoped to **not** consume any of the
team's exp 7 / 9 / 11 artefacts (Alper Ertenç) — specifically the
multiscale parquets, the ViT-only FAISS index, the ConvNeXt-L DINOv3
weights, the geo mask, the temporal-fusion code. The rationale is
paper-driven, not technical: a single working-notes paper is cleanest
when the method's contribution is decoupled from another author's
pipeline. The fair-game inputs are:

- Murilo's own cached per-tile probability parquets
  (`shared/inference/dinov2-vit-base/{no_grid, grid_2x2, …, grid_10x10}.parquet`).
- Team-shared train CLS embeddings
  (`shared/embeddings/train/sample_id={0..19}/`, 1.4 M × 768-d, raw).
- The public DINOv2 ViT-B/14-reg4 PlantCLEF 2024 checkpoint at
  `shared/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth`.
- Raw test/train images, plus public metadata
  (`PlantCLEF2024singleplanttrainingdata.csv`, etc.).
- Per-tile *test* CLS embeddings extracted fresh for this experiment
  (~2 GPU-h, written to `cls_grid_4x4.parquet`).

The independence boundary should be revisited only after the working-notes
paper is drafted; for an ensemble-track follow-up it is not load-bearing.

---

## 2. Hypotheses

Both stated as falsifiable, with explicit Kaggle macro-F1 acceptance
and rejection thresholds set *in advance*:

- **H1.** A label-as-query decoder over per-tile DINOv2 CLS embeddings,
  trained on synthetic pseudo-quadrats with Asymmetric Loss, produces a
  competitive multi-label classifier without modifying the frozen
  backbone.
  - **Confirm** if A1 ≥ **0.30810** (the A0b anchor — sorted-prob top-9
    on grid_4x4, the best-known aggregator on the same per-tile signal).
  - **Reject** if A1 < **0.288** (A0b − 0.02).

- **H2.** Anchoring synthetic pseudo-quadrats to real test-quadrat tile
  CLSs (Sampler B) — with FAISS-based species detection on the
  surviving backgrounds — produces a more useful training distribution
  than random-background sampling (Sampler A), measured by macro-F1 on
  the test set after identical decoder training.
  - **Confirm** if A2 − A1 ≥ **+0.01** macro-F1.
  - **Reject** if |A2 − A1| ≤ **0.005**.

The proposal explicitly stated that a null result on H2 is publishable:
the value of the experiment is the controlled comparison, not the
absolute leaderboard delta. The proposal did not enumerate a third
state ("B is significantly worse than A") — this is one of the findings.

The paper's value is methodological: an independent ablation on top of
the cached per-tile features, evaluated cleanly against the
existing best aggregator from the 2025 working notes.

---

## 3. Method

### 3.1 Architecture: TileQ-Decoder

A small (~4.5 M parameter) label-as-query transformer decoder runs on
top of the frozen ViT-B backbone:

```
test image (~3000×3000)
   ↓ tile into 4×4 grid of 518×518
16 tiles × 518²
   ↓ frozen DINOv2 ViT-B forward (per tile)
Z ∈ ℝ^(16 × 768)             (per-quadrat tile CLS tensor)
   ↓ linear projection (768 → 384)
K, V ∈ ℝ^(16 × 384)
   ↑                          ← 64 learnable group queries Q ∈ ℝ^(64 × 384)
   ↓ multi-head cross-attention (8 heads)
H ∈ ℝ^(64 × 384)
   ↓ LayerNorm + residual + 2-layer FFN (GELU, hidden 2·d_model) + LayerNorm
H' ∈ ℝ^(64 × 384)
   ↓ group-FFN classifier: W ∈ ℝ^(64 × 384 × 122), einsum(bgd, gdc → bgc)
logits ∈ ℝ^(64 × 122) → reshape and trim → ℝ^(7806)
   ↓ sigmoid + top-k decode
species set
```

Concrete decoder dimensions: `d_model = 384`, `n_groups = 64`, `n_heads
= 8`, `classes_per_group = ⌈7806 / 64⌉ = 122`, total trim from 7,808 to
7,806 outputs. Exact parameter count: **4,509,824**.

**Why group-FFN, not per-class queries.** Per-class queries (7806
individual learnable queries) would gradient-starve the long tail —
in our class-frequency stats, 609 species (7.8 %) have fewer than 5
training images, and 1,062 (13.6 %) have fewer than 10. Group-FFN
(ML-Decoder; Ridnik et al., WACV 2023) keeps the query count tractable
(every batch updates all 64 queries) and uses a per-group linear row
to disambiguate species within a group.

**Backbone is fully frozen at all times.** The decoder is the only
trained component (~5 % the size of the backbone). The backbone is
loaded once from `vit_base_patch14_reg4_dinov2.pth` and used through
`forward_features` followed by `forward_head(..., pre_logits=True)` to
extract the 768-d CLS token per tile.

### 3.2 Loss: Asymmetric Loss

Asymmetric Loss (Ridnik et al., ICCV 2021) with parameters γ⁻ = 4,
γ⁺ = 0, m = 0.05. The hard-margin shift on negatives (`p_- = max(p − m,
0)`) discards confident-but-wrong negatives — important for Sampler B
where unknown background tiles may secretly contain positive species.

For ground-truth label `y ∈ {0,1}` and prediction `p = σ(logit)`:

```
p_+ = p
p_- = max(p − m, 0)
L_+ = (1 − p_+)^{γ⁺} · log(p_+)
L_- = (p_-)^{γ⁻}      · log(1 − p_-)
loss = − ( y · L_+ + (1 − y) · L_- )
```

ASL reduces to BCE when γ⁻ = γ⁺ = m = 0, which is a unit test we
included.

### 3.3 Pseudo-quadrat construction (CLS-domain)

A pseudo-quadrat is a `(16, 768)` tensor + a 7806-dim multi-hot label.
Both samplers share these parameters:

- `n_positives = 5` per pseudo-quadrat, fixed across the experiment.
- Species sampling weights `w_s ∝ 1 / √count[s]`, where `count[s]` is
  the number of training images of species `s`. Common species still
  have many CLSs to draw from; rare species get an oversampling boost
  without each rare CLS appearing too many times per epoch.
- Per chosen positive species, sample one CLS uniformly from that
  species' cached embeddings.
- CLS vectors are read **as-is** (not L2-normalised) into the decoder;
  L2 normalisation is only applied for the FAISS retrieval step in §3.4.

**Why CLS-domain not pixel-domain.** Because each ViT tile is forwarded
independently in the pipeline (no cross-tile attention in the
backbone), the CLS computed from a tile in a synthetic pixel mosaic is
identical to the cached CLS of that tile photographed alone. This
eliminates any need for image compositing, attention-rollout masking,
or backbone forward passes during training — the synthetic supervision
runs entirely on cached features at ~150 steps/second on a single
RTX 6000.

#### Sampler A — naive (random-background)

For each pseudo-quadrat:

1. Sample `n_positives = 5` species without replacement, weighted by
   `w_s ∝ 1/√count[s]`.
2. For each chosen species, draw one CLS uniformly from that species.
3. Sample 11 additional "background" species with replacement (same
   weighting), draw one CLS each.
4. Shuffle the 16 CLSs to a random order.
5. The multi-hot label is exactly the 5 injected species — the
   backgrounds are labelled as negatives even though they are themselves
   real plant CLSs from other species.

The ASL hard-margin (`m = 0.05`) is the principled way to absorb the
resulting label noise: backgrounds whose probability prediction is
within `m` of zero are treated as correctly classified.

#### Sampler B — test-anchored, FAISS-relabelled

For each pseudo-quadrat:

1. Pick a random test quadrat `q ∈ {0, …, 2104}`.
2. Initialise the 16 tile CLSs to `test_cls_grid4[q]` — the real
   cached test CLSs for that quadrat.
3. Build the background label set from FAISS-discovered species on the
   real test tiles (see §3.4 for the FAISS map).
4. Sample 5 injected species the same way as Sampler A.
5. Pick 5 random slots to overwrite with train-domain CLSs of the
   injected species.
6. Drop from the background label set any species whose tile got
   overwritten (we no longer have evidence of that species in the
   pseudo-quadrat).
7. Multi-hot label = injected species set ∪ surviving background labels.

So in Sampler B, 5 of 16 tiles are train-domain (the injected species),
and 11 are real test-domain. Of those 11, at τ = 0.65 only about 0.9
on average have a FAISS-discovered species label (see §3.4 for the
threshold rationale and the new-failure-mode consequences).

### 3.4 FAISS background-cleaning map (Sampler B prerequisite)

Built once before Sampler B training. Procedure:

1. L2-normalise all 1,408,033 train CLS embeddings (mean L2 ≈ 18.6
   in raw form; normalised in place to unit sphere).
2. Build `faiss.IndexFlatIP` over the normalised train vectors
   (~4.3 GB CPU RAM).
3. L2-normalise all 33,680 test tile CLS embeddings.
4. For each test tile, query top-1 nearest train neighbour
   → record (similarity, train index, train species).
5. Plot the histogram of top-1 cosine similarities across all 33,680
   test tiles.
6. Pick threshold τ at the visible "shoulder" of the distribution.
7. Tiles with top-1 sim ≥ τ get their train neighbour's species as a
   pseudo-label; tiles below τ are treated as true background (no
   label).

The spec's prior was τ = 0.85 based on the *expected* bimodal shape
(same-species cluster around ~0.85, true-background cluster around
~0.3–0.5).

**Empirical reality** (from the histogram, n = 33,680):

```
min  0.279
mean 0.513
p50  0.508
p90  0.617
p95  0.655
p99  0.751
max  0.885
```

The distribution is **unimodal** centred at 0.51 with a thin right tail
to 0.89. There is *no shoulder* at the spec's 0.85 default — labelling
at τ = 0.85 would have applied to only 18 of 33,680 tiles (0.05 %).
This is the empirical signature of the train↔test domain gap (see
§3.5): even genuine same-species matches max out around 0.89 cosine
because test tiles are vegetation-scene crops while training images are
isolated single plants.

| τ | tiles labelled | % of 33,680 |
|---|---:|---:|
| 0.85 (spec default) | 18 | 0.05 % |
| 0.80 | 114 | 0.34 % |
| 0.75 | 342 | 1.02 % |
| **0.65 (chosen)** | **1,838** | **5.46 %** |
| 0.60 | 4,509 | 13.4 % |

**τ = 0.65 chosen** as the empirical override of the spec default.
Rationale:

1. **Asymmetric cost.** False-positive labels are toxic (they teach
   the decoder a wrong test-tile ↔ species mapping); false-negatives
   are absorbed by ASL γ⁻ = 4 margin. Err high (toward fewer labels).
2. **Spot-check at the high-sim tail confirms biological coherence.**
   The top-10 highest-similarity tiles all match plausible Mediterranean
   species — e.g., 4 tiles in `GUARDEN-CBNMed-29-*-20240428` quadrats
   match *Eryngium maritimum* (sim 0.87–0.89); 5 tiles in
   `GUARDEN-CBNMed-{8,9}-*-20240429` consecutive sites match *Quercus
   coccifera* (sim 0.86–0.87).
3. **0.65 is past p95 of the distribution but still in a regime where
   retrieval looks sensible.** Roughly 0.9 surviving FAISS-relabelled
   tiles per pseudo-quadrat on average — a meaningful Sampler B
   contrast vs Sampler A's zero structural information about test
   distribution.

### 3.5 Domain-gap diagnostics (an in-passing finding worth surfacing)

Per-tile CLS L2 norms diverge between train and test:

| | Train CLS | Test CLS |
|---|---:|---:|
| L2 norm mean | 18.6 | 25.4 |
| L2 norm std | 3.2 | 2.8 |
| Source | 1.4 M Pl@ntNet single-plant photos | 33,680 vegetation-quadrat patches |

Test L2 is ~37 % higher than train L2. This is the empirical signature
of the train↔test domain shift: the backbone produces higher-magnitude
CLS responses on cluttered vegetation scenes than on isolated plants.

The decoder's `proj_kv` linear is expected to absorb the scale shift
during training. We verified this post-hoc by comparing the
overfit-one-batch convergence rate for the two samplers on real
artefacts — Sampler B (tile L2 ≈ 24) and Sampler A (tile L2 ≈ 19)
converge at near-identical rates (both reach near-zero loss by step 80
of 200; see §5.2). Scale absorption is not the failure mode.

### 3.6 Inference

At test time:

1. Load `shared/inference/dinov2-vit-base/cls_grid_4x4.parquet`
   (33,680 rows = 2,105 images × 16 tiles, 768-d float16).
2. Reconstruct per-image `(16, 768)` tensors keyed on `image_name`.
3. Forward each through the trained decoder → 7,806 logits.
4. Sigmoid → per-class probabilities.
5. Top-k decoding at both `k = 9` (apples-to-apples vs A0b's
   2025-baseline anchor) and `k = 10` (organizers' notebook convention)
   from the *same* forward pass — two CSVs per decoder.
6. Map class indices → Pl@ntNet species IDs via
   `class_to_species_id.json`.
7. Write CSV in the Kaggle submission format
   (`pandas.to_csv(sep=",", quoting=csv.QUOTE_ALL)` matching exp 002).
8. Persist the per-image probability vector to a `.npy` for later
   α-blending (§3.7) — no decoder forward needed at blend time.

No per-class thresholding, no geo mask. Top-k decoding alone, by
design, to keep the contrast vs A0b as clean as possible.

One forward pass over 2,105 quadrats takes 7–8 s on a login-node CPU.

### 3.7 Score-blend (A3 bonus ablation)

For α ∈ {0.0, 0.1, …, 1.0}:

```
s_c = α · ŷ_c^{decoder} + (1 − α) · max_t p_{t, c}^{cached}
```

then top-k decode at fixed k = 10. The α-blend is a pure NumPy
combination of the decoder's per-image probability vector with the
cached `max_prob` baseline; no retraining.

The proposal expanded the spec's 5-point α grid to 11 points after
verifying the Kaggle daily submission cap is 100/day, not the ~5/day
the spec earlier cited.

α = 0 and α = 1 are sanity duplicates of the source baselines (max_prob
top-10 = A0a; decoder top-10 = A1@k10) and we use them to verify the
blend math byte-for-byte.

---

## 4. Experimental setup

| Component | Value |
|---|---|
| Backbone | DINOv2 ViT-B/14 reg4, PlantCLEF-2024-fine-tuned, public checkpoint, fully frozen |
| Decoder params | 4,509,824 (~4.5 M) |
| Training data | 1.4 M train CLS embeddings, 7,806 classes (median 176 / species, max 823, min 1) |
| Pseudo-quadrat construction | CLS-domain (no pixel compositing, no backbone forwards at training time) |
| n_positives per pseudo-quadrat | 5, fixed |
| Tiles per pseudo-quadrat | 16 (matching test grid 4×4) |
| Species sampling weights | `w_s ∝ 1 / √count[s]`, normalised to sum 1 |
| Loss | Asymmetric Loss (γ⁻ = 4, γ⁺ = 0, m = 0.05) |
| Optimiser | AdamW, weight_decay = 0.05 |
| Learning rate | 3e-4 peak, 1,000-step linear warmup, cosine decay to 1e-5 |
| Batch size | 32 (RTX 6000, 24 GB) |
| Total steps | 80,000 (2.56 M pseudo-quadrats per run) |
| Gradient clipping | 1.0 |
| Precision | Mixed: bf16 attempted, fp32 fallback on RTX 6000 path |
| Seed | 42 (logged per run; numpy / torch / python) |
| Validation | 1 % of training images held out by `image_name` hash; ~2,000 synthetic val pseudo-quadrats; multi-label cross-entropy + acc@10 logged every 2,000 steps |
| Inference grid | 4×4 only (single-scale) |
| Inference decoding | sigmoid → top-k (k ∈ {5, 7, 9, 10, 12, 15}) — k = 9 (vs A0b) and k = 10 (organizers' convention) are the headline cells |
| Submission format | CSV, `pandas.to_csv(sep=",", quoting=csv.QUOTE_ALL)`, columns `quadrat_id, species_ids` |
| Kaggle cap | 100 submissions/day (verified) |
| W&B project | `plantclef-2026-exp003` (run IDs: A = `u5ma36yz`, B = `hdcx1p23`) |

**Sanity gates before any long training**, all met:

- **Overfit-one-batch.** Sample 32 pseudo-quadrats once, train on this
  single batch for 200 steps. Loss must drop by ≥ 5×. Result: Sampler
  A 306.66 → 0.0157 (**19,561×** reduction in 44 s); Sampler B
  305.10 → 0.0233 (**13,075×** reduction in 46 s). Both reach
  near-zero loss by step 80.
- **Shape audit.** All tensor shapes printed through one
  forward + backward; 17 / 17 parameters receive gradient; output
  shape `(B, 7806)` confirmed.
- **Sampler unit tests.** 12 tests per sampler covering: shapes,
  label cardinality = `n_positives`, positives distinct, tiles drawn
  from correct per-class pool, tile-order shuffling, weight respect over
  many draws (Sampler A); shapes, label includes injections + any
  surviving FAISS-discovered backgrounds, overwritten background labels
  strictly dropped, label dedupes when injection coincides with bg label,
  untouched slots match `test_cls[q, t]` exactly (Sampler B).
- **Pytest.** 80 / 80 unit tests passing.

**What was deliberately omitted in v1** (each chosen for ablation
cleanliness, all in spec §8 / paper Discussion as future work):

- Geo mask (LUCAS SW-Europe). The team's existing pipeline uses
  Alper's `sw_europe_mask_bool.npy`; under the independence boundary
  we would rebuild it from `metadata_lucas/LUCAS_metadata_SWE_Wide_77k.csv`.
  Recommendation was to omit it from v1 to keep the decoder→submission
  path the cleanest possible single-variable comparison.
- Per-class F1-optimal thresholding.
- Multiscale tile inputs (3×3, 5×5, 6×6).
- Variation of `n_positives` per pseudo-quadrat.
- End-to-end fine-tuning (decoder + backbone).
- Abstain query (K + 1 logit absorbing "no species" tiles) — this is
  a particularly relevant future direction given the failure mode we
  characterise in §6.

---

## 5. Training results (synthetic / offline)

Both Sampler A and Sampler B trained at the same hyperparameters on
RTX 6000 partitions of PACE Phoenix.

### 5.1 Wall-clock and step rate

| | Sampler A | Sampler B |
|---|---:|---:|
| SLURM job | 8038028 | 8038496 |
| GPU | RTX 6000 (atl1-1-03-006-25-0) | RTX 6000 (atl1-1-03-006-3-0) |
| Wallclock | 12:05 (9.4 min training + venv install) | 11:46 (9.1 min training + venv install) |
| Step rate | 142.5 step/s | 146.7 step/s |
| Peak RAM | 24.0 GB | 24.0 GB |

Each run trained on 2.56 M pseudo-quadrats (80 k steps × batch 32),
hour-scale of compute on the RTX 6000 path. The 4.5 M-param decoder is
GPU-cheap per step; the spec's earlier estimate of ~6 h training was
loose.

### 5.2 Loss / val-acc trajectory (synthetic)

Values pulled from `loss_sampler{A,B}.csv` (full per-step curves in
`loss_sampler{A,B}.png`):

| Step | A train loss | A val loss | A acc@10 | B train loss | B val loss | B acc@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 306.72 | — | — | 305.02 | — | — |
| 2,000 | 8.36 | 8.39 | 0.9251 | 5.99 | 6.30 | 0.9931 |
| 4,000 | 7.30 | 7.23 | 0.9831 | 3.95 | 3.70 | 1.0000 |
| 8,000 | 6.45 | 6.33 | 0.9916 | 2.49 | 2.59 | 1.0000 |
| 18,000 | 5.59 | 5.66 | 0.9965 | 1.67 | 1.76 | 1.0000 |
| **80,000** | **4.91** | **4.85** | **0.9965** | **0.93** | **0.92** | **1.0000** |

Both runs train cleanly: train ↘ val tightly tracked (no overfitting
on the synthetic-vs-synthetic comparison), LR warmup ends at step 1,000
at peak 3e-4, cosine floor of 1e-5 reached exactly at step 80,000.

**Sampler B's final loss is ~5.3× lower than Sampler A's** (0.93 vs
4.85). At training time we read this as encouraging for H2. **It was
not** — Sampler B's *synthetic* val loss is easier to fit because the
sampler's synthetic distribution is different (and easier) than
Sampler A's:

- Sampler A's 11 background tiles are real plant CLSs of "wrong"
  species — they sit confidently in train-CLS regions of the embedding
  space, and ASL has to drive the decoder to label them as negatives
  anyway.
- Sampler B's 11 untouched tiles are real *test-quadrat* CLSs which
  live in a region of CLS space distinct from any training cluster (see
  §3.4 — top-1 cosine sim is unimodal at 0.51 cosine, not bimodal).
  The decoder finds it trivially easy to output near-zero logits
  everywhere on those.

**This is the key insight the synthetic val loss obscured:** the lower
B loss reflects an easier *training distribution*, not a generalisation
gain. The H2 question — whether B's training distribution transfers
better to the real test set than A's — is, by construction, decidable
only by Kaggle.

---

## 6. Kaggle results (the only test signal)

### 6.1 Headline ablation table (Murilo's independent runs only)

| # | Method | k | Avg species/img | Public LB F1 | Δ vs A0b |
|---|---|---:|---:|---:|---:|
| ref | Murilo no-grid baseline (exp 001, full-image inference) | — | — | 0.14506 | −0.16304 |
| A0a | cached `max_prob` top-10, grid 4×4 | 10 | 10.0 | 0.22986 | −0.07824 |
| **A0b** | cached sorted-prob top-9, grid 4×4 (H1 anchor) | 9 | 5.0 | **0.30810** | — |
| **A1** | TileQ-Decoder Sampler A | 9 | 9.0 | **0.12180** | **−0.18630** |
| A1 | TileQ-Decoder Sampler A | 10 | 10.0 | 0.11640 | −0.19170 |
| **A2** | TileQ-Decoder Sampler B (τ = 0.65) | 9 | 9.0 | **0.09060** | **−0.21750** |
| A2 | TileQ-Decoder Sampler B (τ = 0.65) | 10 | 10.0 | 0.08445 | −0.22365 |

For reference (out of scope for the independence-respecting paper run,
but shown for context):

| | F1 |
|---|---:|
| Team best (Alper exp 11): ConvNeXt-L DINOv3 + ViT-B DINOv2, multiscale 4+5+6, FAISS kNN, prior_blend, geo mask | 0.4379 |

### 6.2 H1 / H2 verdicts (per pre-registered criteria)

| Hypothesis | Pre-registered criterion | Outcome |
|---|---|---|
| **H1**: A1 ≥ 0.30810 (A0b anchor) | reject if A1 < 0.288 | **REJECTED** (A1 = 0.12180; 0.166 below reject threshold) |
| **H2**: A2 − A1 ≥ +0.01 | reject if \|A2 − A1\| ≤ 0.005 | **REJECTED, opposite direction** (A2 − A1 = −0.0312; 6× outside the noise band, but **negative**) |

**H2 falls into a third state the proposal didn't enumerate.** The
proposal binarised the H2 outcome to confirm (B ≥ A + 0.01) vs reject
(|A − B| ≤ 0.005). The actual outcome — B significantly *worse* than
A — is decisively rejected by the "≤ 0.005" criterion, but the *sign*
is the interesting thing for the paper. Worth flagging in the
Discussion as a sub-finding.

### 6.3 The decoder is below the no-grid baseline

A1 = 0.12180 (best decoder result) is **below the no-grid baseline of
0.14506** (Murilo's exp 001 — running the same frozen ViT-B on the
whole un-tiled image, then sorted-prob top-9). In other words, the
decoder's per-image probability vector is *less useful for top-k F1
than the unconditioned full-image classifier*. The decoder is
destroying signal that's already present in the cached per-tile probs.

This is a stronger negative result than "decoder ≈ baseline" or
"decoder underperforms aggregator" would have been. The decoder is not
neutral; it is actively worse than ignoring the tile structure
entirely.

### 6.4 k-sensitivity (within sampler)

For both samplers, k = 10 underperforms k = 9 by a small but consistent
margin:

| | k = 9 | k = 10 | Δ |
|---|---:|---:|---:|
| A1 (Sampler A) | 0.12180 | 0.11640 | −0.005 |
| A2 (Sampler B) | 0.09060 | 0.08445 | −0.006 |

The 10th class adds more false positives than true positives,
consistent with the model already over-predicting at k = 9. CSVs at
k ∈ {5, 7, 12, 15} per sampler are written but un-submitted; they
would not change the headline picture (A1 is below A0b at every
realistic k by an order of magnitude bigger than the
k-sensitivity gap).

### 6.5 A3 α-blend (the strongest version of the null)

Source: A1 (Sampler A) — the better-scoring decoder at 0.12180 LB. All
A3 submissions at fixed top-k = 10 to match A0a's k.

```
s_c = α · decoder_prob_c + (1 − α) · max_prob_c     →     top-10 decode
```

| α | LB F1 | Δ vs α=0 | Δ vs A0a (0.22986) |
|---:|---:|---:|---:|
| **0.00** (pure max_prob) | **0.22144** | — | −0.00842 |
| 0.10 | 0.21952 | −0.00192 | −0.01034 |
| 0.20 | 0.21827 | −0.00317 | −0.01159 |
| 0.30 | 0.21001 | −0.01143 | −0.01985 |
| 0.40 | 0.19954 | −0.02190 | −0.03032 |
| 0.50 | 0.18931 | −0.03213 | −0.04055 |
| 0.60 | 0.17292 | −0.04852 | −0.05694 |
| 0.70 | 0.15277 | −0.06867 | −0.07709 |
| 0.80 | 0.14562 | −0.07582 | −0.08424 |
| 0.90 | 0.13154 | −0.08990 | −0.09832 |
| **1.00** (pure decoder) | **0.11640** | −0.10504 | −0.11346 |

**Key observations:**

1. **The curve is monotonically decreasing in α.** Every increase of
   α reduces LB F1. No convex combination of decoder + max_prob beats
   α = 0.
2. **α = 1.00 = 0.11640 exactly matches A1@k10.** Sanity check
   confirms the blend math (and the file IO) is byte-correct at the
   boundary.
3. **α = 0.00 = 0.22144 vs A0a = 0.22986** has a 0.008 gap. Source:
   tie-breaking differences (numpy `argpartition + argsort` vs pandas
   `sort` on species with near-equal softmax probabilities). About 80
   of 2,105 rows have positions 8/9 or 9/10 swapped, occasionally a
   different boundary species at rank 10 in the strictest cases. Same
   top-10 sets in 96 % of rows. The *relative* A3 curve is internally
   consistent (all 11 use the same tie-breaking).
4. **The smallest hit at α = 0.10 (Δ = −0.002 vs α = 0) is inside
   Kaggle's noise floor** — but the curve is monotone from α = 0.10
   onward, so there's no narrow "sweet spot" hidden in finer α
   resolution. We did not run finer resolution; it would not change
   the conclusion.

**A3 verdict.** The spec §2.5 framed the α-blend question as
"complementary vs redundant" — the implicit prior was that some
intermediate α might lift performance (if decoder probabilities had any
useful disagreement with max_prob), or that the best α would be near
the boundaries (purely complementary vs purely redundant). We get a
**third outcome**: best α is *exactly* at the boundary (α = 0, pure
max_prob), with monotone-worse blends on every step toward α = 1. The
decoder is not even partially redundant; it is purely additive noise
that destroys F1 at any non-zero weight.

---

## 7. Mechanism analysis

### 7.1 Why did Sampler A underperform every baseline?

The proposal's prior failure mode for Sampler A was **synthetic-
distribution overfit** (spec §7.3): the decoder learns the
synthetic-pseudo-quadrat distribution well but the synthetic
distribution doesn't transfer to the real test distribution.

This is consistent with the observed pattern:

- A's synthetic val loss plateaus at ~4.85 (acc@10 = 99.65 %) — the
  decoder fits its training distribution cleanly.
- Most-predicted top-5 species across the 2,105 test images: *Carex
  pyrenaica* (703 / 2,105), *Pinguicula nevadensis* (633), *Primula
  intricata* (630), *Oreochloa elegans* (608), *Viola pyrenaica*
  (572). Biologically plausible Mediterranean / Pyrenean species
  — i.e., the decoder collapsed onto a *prior-like* output, mode-
  collapsing onto plausible vegetation but not actually discriminating
  per-image content.
- Avg per-image probability statistics: classes per image with prob >
  0.1 = 6,988 / 7,806 (90 %); classes per image with prob > 0.5 =
  median 10; unique classes appearing in any top-9 across all 2,105
  images = 1,874 / 7,806 (24 %).

So Sampler A's decoder is overconfident on out-of-distribution
inputs (test tiles look unlike train tiles, but A's synthetic
distribution was made of train-like tiles, so A produces train-like
predictions on test). Its predictions are concentrated on ~1,800 plausibly-
Mediterranean species — better than uniform guessing, but well below the
aggregator on the same per-tile probabilities.

### 7.2 Why did Sampler B underperform Sampler A?

This is the failure mode worth a Discussion paragraph. The proposal's
prior was that Sampler B would *help* via two mechanisms:

1. `proj_kv` exposure to test-scale CLSs at training time would
   eliminate the train↔test domain gap from the decoder's perspective.
2. FAISS-discovered species labels on surviving (non-overwritten)
   backgrounds would teach the decoder a coherent map from
   test-distribution tiles to species labels.

Mechanism (1) **did work.** Both samplers converge cleanly without
scale collapse (§5.2), and Sampler B's per-image prediction
distribution is sharper and more diverse than Sampler A's — see §7.4
below for the prediction-distribution table.

But the Kaggle leaderboard contradicts the prior anyway.

**The mechanism we believe explains the failure** (this is the new
failure mode worth surfacing in the paper):

At τ = 0.65, only 5.46 % of test tiles get a FAISS label (1,838 of
33,680). The remaining 94.5 % of test tiles enter Sampler B's
pseudo-quadrats as 11 untouched slots that contribute *zero* to the
multi-hot label. Training-time, the decoder consistently sees:

> *"test-shaped tile → no species in the multi-hot label."*

Over 80,000 training steps × batch 32 with 11/16 test-shaped
unlabelled tiles per pseudo-quadrat, that supervision signal
dominates. The decoder learns to output low logits everywhere on
test-shaped inputs. At inference time, *all 16* tiles are
test-shaped, so the decoder's per-class probability vector is
uniformly low — and forced top-k decoding picks essentially noise from
the bottom of the probability distribution.

Sampler A's overconfident-on-OOD output happens to have **more useful
structure** for top-k decoding than Sampler B's flat-on-OOD output —
because at least the predicted classes are confidently chosen
biologically-plausible Mediterranean plants, some of which are
correct by the prior.

This is **not** the failure mode spec §7.3 enumerated for Sampler A
(synthetic-distribution overfit, which was about Sampler A's
confidently-wrong negative labels, and *did* hold for Sampler A
empirically). It's a *new* mode specific to Sampler B's mechanism of
test-anchoring + sparse FAISS labels.

### 7.3 Why this isn't a problem of FAISS τ choice

A natural question: would relaxing τ (more FAISS labels per tile)
fix it? Answer: probably not, in this regime. Two arguments:

1. **The histogram has no shoulder.** The distribution is unimodal at
   0.51 cosine with a thin right tail. Dropping τ to 0.50 would
   label ~50 % of tiles with their top-1 train neighbour's species,
   but the median similarity is 0.51, which means most of those
   "labels" would be near-orthogonal — random species assignments
   leaking into the multi-hot supervision. We argued in §3.4 that
   false-positive labels are toxic (they teach a wrong
   test-tile↔species map); a low τ trades the under-confidence
   failure mode for a high-noise mode that ASL would only partially
   absorb.
2. **The asymmetry is the contrast.** If the H2 contrast required
   tuning τ post-hoc to match A, the experimental design would be
   compromised. The right framing for the paper is: at the most
   defensible τ choice from the empirical histogram (τ = 0.65, just
   past p95 of the distribution), Sampler B underperforms — and
   the under-confidence mechanism explains why.

What would actually fix this failure mode (Discussion section): an
*abstain query* (K + 1 logit), or a denser test-tile labelling source
(attention-rollout, land-cover prior). See §10.

### 7.4 Pre-Kaggle prediction-distribution analysis — foreshadowed the failure

Before submitting, we computed per-image probability statistics
across all 2,105 test images:

| Metric | A1 (Sampler A) | A2 (Sampler B) |
|---|---:|---:|
| classes per img with prob > 0.1 | 6,988 (90 %) | 312 (4 %) |
| classes per img with prob > 0.5 (median) | 10 | **0** |
| unique classes in any top-9 | 1,874 (24 %) | 3,737 (48 %) |
| max prob per image (mean) | 0.65 | 0.57 |

At the time we read this as "B is sharp + diverse" — encouraging for
H2. **The LB now lets us reinterpret: B's median 0 classes > 0.5 was
the warning sign.** Forced top-9 from a near-flat probability vector
is essentially noise on the low end of the distribution; A's
overconfident vector at least has *informative* mode-collapse onto
plausible species.

Most-predicted top-5 species (any top-9):

- A1: *Carex pyrenaica* (703), *Pinguicula nevadensis* (633), *Primula
  intricata* (630), *Oreochloa elegans* (608), *Viola pyrenaica* (572).
- A2: *Helictochloa versicolor* (353), *Carex distans* (259), *Agrostis
  canina* (245), *Ulmus pumila* (210), *Festuca filiformis* (199).

Both pick biologically plausible SW-Europe vegetation, just with
different mode-collapse patterns. A1 collapses harder onto fewer
species (1,874 unique top-9 classes total); A2 spreads predictions
across more species (3,737 unique) but with much lower individual
confidence.

### 7.5 Why the spec's pre-registered failure modes did and didn't bind

Spec §7 enumerated seven expected failure modes with detection signals.
For the record:

| Spec §7 mode | Did it activate? | Notes |
|---|---|---|
| Mode collapse on common species | Partially | A1's 1,874 unique top-9 classes; clear collapse onto Mediterranean prior |
| Background bleed / over-prediction | No | Avg species / image at k = 9 is exactly 9 / 10 by construction; no excess |
| Synthetic-distribution overfit (Sampler A) | **Yes** | A1's training val loss plateaus at 4.85 while LB collapses to 0.12180. Detection signal as predicted (val ↘ but LB flat-or-worse). |
| Group-FFN class-assignment artefacts | Not visible | Did not investigate per-group F1; no period-by-class-index inspection done |
| FAISS τ misset | **Partially — different direction.** | Detection signal "B's effective label count much higher than A" did not occur (only 5.46 % of tiles labelled, vs A's effective ~5 / 16 = 31 % label rate). The misset was in the *other* direction, and it led to a different failure mode (§7.2) rather than the noisy-relabelling one §7 anticipated. |
| Submission CSV format mismatch | Caught and fixed | A0a/A0b derisk + infer.py CSV-format bug → fixed before headlines |
| Embedding-scale collapse | **No** | `proj_kv` absorbed the test L2 ≈ 25.4 vs train L2 ≈ 18.6 cleanly; both samplers converged at near-identical rates in overfit-one-batch |

Net: 1 of 7 spec failure modes was the binding mode for the LB result
(synthetic-distribution overfit for Sampler A). The dominant failure
mode for Sampler B was **not** in the enumerated list — the
FAISS-undersampling under-confidence mode. This is the paper's
secondary contribution.

---

## 8. Synthesizing the headline contributions

For the paper, the three things worth surfacing are:

1. **A pre-registered null result on label-as-query decoding from
   CLS-domain pseudo-quadrats** (the publishable null the proposal
   anticipated). The decoder underperforms every aggregator baseline
   on the same backbone features, on the same per-tile CLS, with the
   same compute envelope. This is the cleanest possible null in this
   class of methods: the decoder doesn't tie the baseline, doesn't
   weakly help, doesn't tie at one k and lose at another — it loses
   decisively at every k, both samplers, every α.

2. **A newly characterized failure mode**: FAISS-undersampling
   under-confidence. When test-anchored sampling labels only a small
   fraction of test tiles (5.46 % at τ = 0.65), the decoder learns
   "test-shaped tile → no species in the label" → outputs uniformly
   low logits at inference → top-k picks noise. Mechanistically distinct
   from the synthetic-distribution overfit mode the spec anticipated
   for Sampler A. Detection signals: median classes per image with
   prob > 0.5 = 0, decoder LB *worse* than test-naive Sampler A.

3. **A3 monotone-decreasing α-curve as a strong version of the null.**
   The proposal framed the blend question as "complementary vs
   redundant"; the actual outcome is a third state — best α is at
   the boundary (α = 0, pure max_prob), with monotone-worse blends on
   every step toward α = 1. The decoder probabilities are not even
   partially redundant; they are purely additive noise.

Each of these is novel-enough by itself to motivate a paper section.
The first is the headline; the second is the mechanism contribution;
the third is the strength-of-null argument.

---

## 9. Auxiliary findings worth recording

Cumulative findings established during this experiment that don't fit
in the main contribution but should appear in the paper's
discussion or related-work positioning:

- **Probability magnitude beats tile frequency.** Sorted-prob top-k = 9
  (rank species by per-tile probability) at F1 = 0.30810 vs union (rank
  by count of tiles nominating a species) at F1 0.19–0.21 on identical
  per-tile predictions. Established in exp 002, reconfirmed by A0a vs
  A0b in this experiment (Δ = +0.078 F1 on the same per-tile probs).
- **Quadrat aggregation hurts at low aggregation strength.** Grouping
  per-image lists by physical plot and unioning made F1 drop 0.21 → 0.17
  in exp 002. The per-image lists are too noisy to benefit from
  cross-visit union; would only help on top of a stronger per-image
  classifier.
- **Data-loading is the inference bottleneck, not GPU compute.** Exp 001
  measured ~99 % of step time on NFS reads at grid 3×3 with GPU util
  ~78 %. Any inference-side experiment on this data should consider
  local-scratch staging or webdataset.
- **Train CLS embeddings are NOT L2-normalised.** Mean L2 ≈ 18.6,
  std 3.2, range [11.8, 29.5]. Must be normalised before any cosine
  retrieval / `IndexFlatIP` query.
- **Train data is hash-partitioned by `image_name`.** Every species is
  spread across all 20 `sample_id={0..19}` partitions. Any
  species-level aggregation (centroids, prototypes, FAISS) requires
  reading all 20 partitions.
- **Long-tail species are 7.8 %.** 609 / 7,806 species have < 5
  images; 1,062 (13.6 %) have < 10. This is the prior that motivates
  group-FFN over per-class queries in TileQ-Decoder; per-class queries
  would gradient-starve.
- **Synthetic val loss is uncorrelated with LB F1.** Sampler B trained
  to ~5.3× lower synthetic val loss than Sampler A (0.93 vs 4.85), with
  sharper and more diverse top-k predictions on the test set, but on
  Kaggle B underperformed A by 0.031 F1 at k = 9. The synthetic
  distribution is different enough from the real test distribution that
  fitting it better doesn't help; train monitoring is a pipeline-health
  check, not a generalisation proxy.
- **Kaggle is the only test signal.** No public ground-truth labels for
  the 2,105 test images. Each Kaggle submission is the only evaluation
  channel; the daily cap (100/day) is generous enough to run the full
  ablation grid + sweeps inside one calendar day.

---

## 10. Discussion / Future-work directions

These are out of scope for v1 but should appear in the Discussion
section as alternatives motivated by the failure mode we characterized.

1. **DINOv2 attention-rollout for background tile selection.** The
   binding failure mode for Sampler B was sparse FAISS labelling (5.46 %
   of test tiles). Attention-rollout aggregates self-attention across
   layers to produce a *foreground mask per tile* that doesn't depend
   on cosine similarity to single-plant train CLSs. Cleaner anchor
   identification might let Sampler B label most or all test tiles
   without false-positive risk.
2. **LUCAS land-cover Bayesian prior.** The 77 K SW-Europe LUCAS records
   in `metadata_lucas/LUCAS_metadata_SWE_Wide_77k.csv` give habitat
   labels that could provide habitat-conditional priors over species —
   reducing or eliminating the need for FAISS-based label discovery at
   all. This would also be a clean independent experiment for a
   working-notes paper.
3. **Abstain query (K + 1 logit).** Add an explicit "no species" logit
   that absorbs unlabelled tiles. This would directly address the
   FAISS-undersampling under-confidence failure mode — instead of
   forcing the model to output uniformly low logits when no species
   matches, it could predict the abstain logit and the top-k decoder
   could ignore unlabelled tiles.
4. **Per-class F1-optimal thresholding** (Lipton et al., ECML-PKDD 2014).
   The current top-k decoding picks a fixed-cardinality output set
   regardless of confidence. Per-class threshold calibration on a held-
   out validation set could allow variable-length predictions.
5. **Multiscale tile inputs.** The team's best pipeline (Alper exp 11)
   uses 4 + 5 + 6 tile grids; the decoder used 4 × 4 only. Fresh CLS
   extraction at 3 × 3, 5 × 5, 6 × 6 is in budget (~2 GPU-h per scale).
6. **Variation of `n_positives` per pseudo-quadrat.** Currently fixed at
   5; the real test distribution has variance not modelled here.
7. **End-to-end fine-tuning (decoder + backbone).** The backbone is
   frozen throughout this experiment. Any benefit from joint training
   is outside the scope of v1.
8. **Geo mask reconstruction.** Independent rebuild from
   `metadata_lucas/LUCAS_metadata_SWE_Wide_77k.csv` (35°–50° N,
   −10°–10° E box) for v2 — same effect as Alper's `sw_europe_mask_bool.npy`
   without depending on the artefact. Likely +0.02 to +0.04 F1
   regardless of decoder quality.

---

## 11. Reproducibility

### 11.1 Wall-clock per stage

| Stage | Compute | Wallclock |
|---|---|---:|
| Test CLS extraction | V100 (1 GPU) | 16 min |
| Build train CLS lookup (sorted-by-class, 1.4 M rows) | cpu-small | 2:14 (build alone 21 s + venv install) |
| Build FAISS background map (1.4 M index, 33,680 queries) | cpu-small | 11:15 (FAISS query 9.8 min) |
| Sanity check (overfit-one-batch, both samplers, 200 steps each) | login-node CPU | 1:30 |
| Train Sampler A (80 k steps × batch 32) | RTX 6000 | 12:05 (9.4 min training) |
| Train Sampler B (80 k steps × batch 32) | RTX 6000 | 11:46 (9.1 min training) |
| Inference per sampler (forward + top-k decode, k ∈ {5,7,9,10,12,15}) | login-node CPU | 7–8 s |
| α-blend (11 CSVs at k = 10) | login-node CPU | < 1 min total |

Total experiment compute envelope: roughly **1 GPU-hour on RTX 6000**
plus **~30 min CPU** for build pipelines plus **~20 min V100** for test
CLS extraction. The decoder is GPU-cheap; the binding compute cost
was the one-time FAISS map build and the test-CLS extraction.

### 11.2 Paths (absolute, on PACE Phoenix)

```
# Inputs (read-only, team-shared)
backbone checkpoint        /storage/project/ps-clef2026_plantclef-0/shared/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth
class_mapping.txt          /storage/project/ps-clef2026_plantclef-0/shared/data/raw/class_id_to_species_mapping/class_mapping.txt
class_to_species.json      /storage/project/ps-clef2026_plantclef-0/shared/data/raw/class_id_to_species_mapping/class_to_species_id.json
species_id_to_name.txt     /storage/project/ps-clef2026_plantclef-0/shared/data/raw/class_id_to_species_mapping/species_id_to_name.txt
test images                /storage/project/ps-clef2026_plantclef-0/shared/data/raw/PlanCLEF_test_data/images/*.jpg
test metadata              /storage/project/ps-clef2026_plantclef-0/shared/data/raw/PlanCLEF_test_data/PlantCLEF2025_test.csv
train metadata             /storage/project/ps-clef2026_plantclef-0/shared/data/raw/metadata_PlantCLEF_training_set/PlantCLEF2024singleplanttrainingdata.csv
train CLS (1.4 M)          /storage/project/ps-clef2026_plantclef-0/shared/embeddings/train/sample_id={0..19}/part-*.parquet
test probs (grid 4×4)      /storage/project/ps-clef2026_plantclef-0/shared/inference/dinov2-vit-base/grid_4x4.parquet
LUCAS SW-Europe            /storage/project/ps-clef2026_plantclef-0/shared/data/raw/metadata_lucas/LUCAS_metadata_SWE_Wide_lat35_50_lon-10_10_77k.csv

# Outputs (this experiment)
test CLS (grid 4×4)        /storage/project/ps-clef2026_plantclef-0/shared/inference/dinov2-vit-base/cls_grid_4x4.parquet                                 # 49.6 MB, 33,680 rows
artifacts                  user/murilo/experiments/003-tileq-decoder/artifacts/
  train_cls_by_class_idx.npz   sorted-by-class CLS lookup + offsets, 4.1 GB
  species_weights.npy          1/√count weights, 31 KB
  sampler_b_lookup.npz         image_names, test_cls, top1_sim, top1_class_idx, tile_to_class_idx, threshold (τ=0.65), 100 MB
  tile_top1_sim_histogram.png  FAISS τ decision artefact
  faiss_threshold.txt          percentile + coverage table
  checkpoints/
    decoder_samplerA.pt        52 MB, step 80 k
    decoder_samplerB.pt        52 MB, step 80 k
  loss_sampler{A,B}.csv        full per-step training curves
  loss_sampler{A,B}.png        plotted training curves
  config_sampler{A,B}.yaml     run hyperparameters
submissions                user/murilo/experiments/003-tileq-decoder/submissions/    # 25 CSVs + SUBMISSIONS.md log
```

### 11.3 Schemas

- `cls_grid_4x4.parquet`: `(image_name: str, tile_idx: int32,
  cls_embedding: binary)` where `cls_embedding` is float16 bytes of
  shape `(768,)`. Decode with `np.frombuffer(b, np.float16)`.
  Joinable to `grid_4x4.parquet` on `(image_name, tile_idx)`.
- `grid_4x4.parquet`: `(image_name: str, tile_idx: int32, probs:
  binary)` where `probs` is float16 bytes of shape `(7806,)`. Decode
  the same way.
- Train CLS partitions: `(image_name: str, species_id: int32,
  cls_embedding: list<float32>[768])`. Hive-partitioned by
  `sample_id ∈ {0..19}` (hash on `image_name`).
- Train metadata: semicolon-delimited CSV; columns include `species_id`,
  `latitude`, `longitude`, others.

### 11.4 Reproducibility checklist

- [x] All random seeds logged at run start (numpy, torch, python; seed
      = 42 for both training runs).
- [x] Hyperparameters serialised to YAML alongside each checkpoint
      (`config_sampler{A,B}.yaml`).
- [x] Build scripts deterministic; both `build_train_lookup.py` and
      `build_faiss_background_map.py` are pure functions of their
      named input files. Input file modification times are printed to
      the log for audit.
- [x] Submission CSVs versioned by run (filename includes sampler +
      seed + top-k).
- [x] Loss curves saved (CSV log + matplotlib plot in `artifacts/`).
- [x] Wall-clock per stage recorded (§11.1).
- [x] All file paths in scripts are absolute or anchored to the
      experiment directory; no `cd` dependencies.

### 11.5 Test coverage

80 unit tests pass in 12.7 s, covering:

- ASL loss (reduces to BCE when γ⁻ = γ⁺ = m = 0; gradient sanity).
- Decoder shape and gradient flow (17/17 params receive gradient).
- Sampler A: 12 tests covering shapes, label cardinality, positive
  distinctness, per-class draw correctness, shuffle effect, weight
  respect.
- Sampler B: 12 tests covering shapes, label set composition,
  overwritten-background-label dropping, dedup against injected
  species, untouched-slot identity to `test_cls[q, t]`.
- Build scripts (train lookup, FAISS map).
- Dataset wrapper (per-worker RNG reseeding).
- LR schedule (warmup + cosine bounds).
- Submission CSV format.

---

## 12. References

For the paper's bibliography.

- Liu, S. et al. *Query2Label: A Simple Transformer Way to Multi-Label
  Classification.* arXiv 2107.10834, 2021.
- Ridnik, T. et al. *ML-Decoder: Scalable and Versatile Classification
  Head.* WACV 2023.
- Ridnik, T. et al. *Asymmetric Loss for Multi-Label Classification.*
  ICCV 2021.
- Cole, E. et al. *Multi-Label Learning from Single Positive Labels.*
  CVPR 2021.
- Zhou, D. et al. *Acknowledging the Unknown for Multi-label Learning
  with Single Positive Labels.* ECCV 2022.
- Bochkovskiy, A. et al. *YOLOv4: Optimal Speed and Accuracy of Object
  Detection.* arXiv 2004.10934, 2020 (Mosaic augmentation).
- Oquab, M. et al. *DINOv2: Learning Robust Visual Features without
  Supervision.* arXiv 2304.07193, 2023.
- Goëau, H. et al. *Overview of PlantCLEF 2024 / 2025.* Working notes,
  CEUR-WS.
- Lipton, Z. et al. *Optimal Thresholding of Classifiers to Maximize F1
  Measure.* ECML-PKDD 2014.

---

## 13. Suggested paper structure (one possible arrangement)

This is the arrangement that would let a research tool draft the paper
in one pass; it follows the narrative of this briefing.

1. **Abstract** — pre-registered null on label-as-query decoding from
   CLS-domain pseudo-quadrats; H1 rejected (A1 0.122 vs anchor 0.308);
   H2 rejected in the opposite direction (B 0.031 below A);
   A3 α-curve monotone decreasing; new failure mode (FAISS-undersampling
   under-confidence) characterized.
2. **§1 Introduction** — PlantCLEF 2026 single→multi-label gap +
   global/local representation, team's existing pipeline as the
   incumbent baseline, A0b as the pre-registered anchor.
3. **§2 Related work** — Query2Label, ML-Decoder, ASL; CLEF working
   notes 2022–2025 on aggregation strategies; positioning of
   label-as-query against per-class linear heads for very-long-tail
   (7,806 classes) settings.
4. **§3 Method** — TileQ-Decoder architecture, ASL, Sampler A vs B,
   FAISS background-cleaning, inference. Pre-register H1 and H2 with
   thresholds. (Material from §3 of this briefing.)
5. **§4 Experimental setup** — backbone, hyperparameters, compute,
   what was deliberately omitted under the independence boundary.
   (Material from §4.)
6. **§5 Results** — full ablation table, k-sensitivity, A3 α-curve.
   (Material from §6.)
7. **§6 Mechanism analysis** — why A underperforms baselines (synthetic-
   distribution overfit per spec §7.3); why B underperforms A
   (FAISS-undersampling under-confidence — *new failure mode*); why A3
   is monotone-decreasing. (Material from §7.)
8. **§7 Discussion / future work** — attention-rollout, LUCAS prior,
   abstain query, per-class thresholding, multiscale, end-to-end.
   (Material from §10.)
9. **§8 Conclusion** — what we learned about label-as-query decoding
   from CLS-domain pseudo-quadrats in the absence of a real-test val
   set; what would have to be true for the method to work; the
   secondary contribution of the new failure-mode characterization.
10. **Appendix** — wall-clock per stage, FAISS τ-tuning histogram and
    coverage table, training-loss trajectories, reproducibility
    checklist. (Material from §11.)
