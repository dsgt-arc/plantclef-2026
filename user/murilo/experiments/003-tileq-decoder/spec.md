# TileQ-Decoder — Technical Specification

> Comprehensive ground-truth implementation reference for experiment 003.
> Consult `proposal.md` for the hypothesis/eval framing and `tasks.md` for the
> incremental implementation checklist. This document is dependency-ordered, not
> day-ordered: any section can be picked up independently once its prerequisites
> are met.

---

## 1. Problem and Hypothesis

PlantCLEF 2026 is multi-label species identification on ~3000×3000 vegetation
quadrats over 7,806 species, scored by macro-F1. Training is single-label
single-plant (1.4 M Pl@ntNet images). The two open problems are
(i) the single→multi-label domain gap and (ii) global-vs-local feature
representation at extreme resolution.

The team's existing pipeline tiles each test image into a 4×4 grid of 518×518
tiles, runs a frozen DINOv2 ViT-B/14 forward pass per tile, and aggregates by
per-class max across tiles (`max_prob`). Murilo's exp 002 reproduced the 2025
sorted-prob top-k=9 baseline at F1=0.30810.

Falsifiable hypotheses (also in `proposal.md`):

- **H1.** A label-as-query decoder over per-tile CLS embeddings, trained on
  synthetic pseudo-quadrats with Asymmetric Loss, produces a competitive
  multi-label classifier without modifying the frozen backbone.
- **H2.** Anchoring synthetic pseudo-quadrats to real test-quadrat tile CLSs
  (Sampler B) produces a more useful training distribution than random-background
  sampling (Sampler A), measured by macro-F1 on the test set after identical
  decoder training.

The paper's value is the controlled comparison; a null result on H2 is
publishable.

---

## 2. Method

### 2.1 Architecture: TileQ-Decoder

Notation. Let `f_θ` be the frozen public DINOv2 ViT-B/14 with 4 register tokens,
fine-tuned on PlantCLEF 2024 single-label data, available at
`/storage/project/ps-clef2026_plantclef-0/shared/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth`.
For each test quadrat, tile on a 4×4 grid of 518×518 → 16 tiles. Run each tile
through `f_θ` and collect the CLS token (`forward_features` followed by
`forward_head(..., pre_logits=True)`). Result per quadrat: `Z ∈ ℝ^{16 × 768}`.

Decoder head:

- Linear projection from 768 to `d_model = 384`.
- `G = 64` learnable group queries `Q ∈ ℝ^{G × 384}`, init `N(0, 0.02²)`.
- One multi-head cross-attention layer (8 heads), `Q` attending to projected
  tile embeddings.
- LayerNorm + residual + 2-layer FFN (GELU, hidden `2 · d_model`).
- Group-FFN head: packed weight `W ∈ ℝ^{G × d_model × C'}` with
  `C' = ⌈7806 / G⌉ = 122`. Each group query produces 122 logits via
  `einsum("bgd,gdc->bgc", h, W) + bias`. Reshape to `(B, G·C')` and trim to
  7806.

PyTorch sketch (place at `user/murilo/src/plantclef/tileq_decoder.py`):

```python
import math
import torch
import torch.nn as nn


class TileQDecoder(nn.Module):
    """Label-as-query multi-label decoder over per-tile CLS embeddings."""

    def __init__(
        self,
        n_classes: int = 7806,
        n_groups: int = 64,
        d_in: int = 768,
        d_model: int = 384,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_groups = n_groups
        self.classes_per_group = math.ceil(n_classes / n_groups)

        self.proj_kv = nn.Linear(d_in, d_model)
        self.queries = nn.Parameter(torch.randn(n_groups, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.group_heads = nn.Parameter(
            torch.randn(n_groups, d_model, self.classes_per_group) * 0.02
        )
        self.group_bias = nn.Parameter(torch.zeros(n_groups, self.classes_per_group))

    def forward(self, tile_cls: torch.Tensor) -> torch.Tensor:
        # tile_cls: (B, T=16, 768)
        B = tile_cls.size(0)
        kv = self.proj_kv(tile_cls)                          # (B, T, D)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)      # (B, G, D)
        attn_out, _ = self.cross_attn(q, kv, kv)             # (B, G, D)
        h = self.norm1(q + attn_out)
        h = self.norm2(h + self.ffn(h))                       # (B, G, D)
        logits = torch.einsum("bgd,gdc->bgc", h, self.group_heads) + self.group_bias
        return logits.reshape(B, -1)[:, : self.n_classes]    # (B, 7806)
```

Total decoder parameters ≈ 4 M. Backbone (~86 M) is never trained.

**Why group-FFN.** Training 7806 individual queries causes gradient starvation
on rare classes (median 176 images/species, but 7.8 % of species have <5
images). Group-FFN keeps the query count tractable (every batch updates all 64
queries) while a per-group linear row handles within-group species
disambiguation. Reference: ML-Decoder (Ridnik et al., WACV 2023).

**Class-to-group assignment.** Default: contiguous (`group = class_idx // 122`).
Optional mitigation for failure mode §7.4: shuffle the assignment per epoch
with a fixed seed schedule.

### 2.2 Loss: Asymmetric Loss (ASL)

Use ASL from Ridnik et al. ICCV 2021 with `γ⁻ = 4`, `γ⁺ = 0`, `m = 0.05`. The
hard-margin shift on negatives discards confident-but-wrong negatives —
important for Sampler B where unknown background tiles may secretly contain
positive species.

For ground-truth label `y ∈ {0, 1}` and prediction `p = σ(x)`:

```
p_+  = p
p_-  = max(p − m, 0)
L_+  = (1 − p_+)^{γ⁺} · log(p_+)            # for positives (y=1)
L_-  = (p_-)^{γ⁻}  · log(1 − p_-)            # for negatives (y=0)
loss = − ( y · L_+ + (1 − y) · L_- )
```

Place at `user/murilo/src/plantclef/asl_loss.py`. Reference implementation:
[Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL).

### 2.3 Pseudo-Quadrat Construction (CLS-Domain)

A pseudo-quadrat is a `(16, 768)` tensor + a 7806-dim multi-hot label.

**Common parameters across both samplers:**

- `n_positives = 5` species per pseudo-quadrat, fixed.
- Species sampling weights `w_s ∝ 1 / √count[s]`, where `count[s]` is the
  number of training images of species `s`. Common species still have many
  CLSs to draw from; rare species get an oversampling boost without each rare
  CLS appearing too many times per epoch.
- Per-positive species: sample one CLS uniformly from that species' cached
  embeddings.
- All CLS vectors are read **as-is** (not L2-normalised) into the decoder —
  the input layer norm and projection learn whatever scale is appropriate.
  L2 normalisation is only applied for the FAISS retrieval step in §3.4.

#### Sampler A — naive (random-background)

```python
def sample_pseudo_quadrat_A(
    train_cls_by_species,    # dict[species_id] -> ndarray(n_i, 768)
    species_weights,         # ndarray(7806,) — w_s ∝ 1/sqrt(count[s])
    n_positives: int = 5,
    n_tiles: int = 16,
):
    species = sample_species(species_weights, n=n_positives, replace=False)
    species_cls = [random_choice(train_cls_by_species[s]) for s in species]

    bg_species = sample_species(species_weights, n=n_tiles - n_positives, replace=True)
    bg_cls = [random_choice(train_cls_by_species[s]) for s in bg_species]

    tiles = species_cls + bg_cls
    random_shuffle(tiles)
    label = multi_hot(species, n_classes=7806)
    return torch.stack(tiles), label
```

The label set is exactly the `n_positives` injected species — backgrounds are
labelled as negatives even though they are themselves real plant CLSs from
other species. ASL's hard-margin absorbs the resulting label noise.

#### Sampler B — test-anchored, FAISS-relabelled

```python
def sample_pseudo_quadrat_B(
    test_cls_grid4,          # (2105, 16, 768)
    train_cls_by_species,
    species_weights,
    tile_to_species,         # dict[(quadrat_idx, tile_idx)] -> species_id | None
    n_positives: int = 5,
    n_tiles: int = 16,
):
    quadrat_idx = randint(0, 2104)
    tiles = test_cls_grid4[quadrat_idx].clone()              # (16, 768)

    background_label_set = {
        tile_to_species[(quadrat_idx, t)]
        for t in range(n_tiles)
        if tile_to_species.get((quadrat_idx, t)) is not None
    }

    injected_species = sample_species(species_weights, n=n_positives, replace=False)
    injected_species_set = set(injected_species)
    inject_slots = random_sample(range(n_tiles), n_positives)
    for slot, species in zip(inject_slots, injected_species):
        tiles[slot] = random_choice(train_cls_by_species[species])

    # Drop any background label whose tile we just overwrote with a known species.
    overwritten_labels = {
        tile_to_species.get((quadrat_idx, slot))
        for slot in inject_slots
        if tile_to_species.get((quadrat_idx, slot)) is not None
    }
    surviving_background_labels = (
        background_label_set - injected_species_set - overwritten_labels
    )
    full_label_set = injected_species_set | surviving_background_labels
    label = multi_hot(list(full_label_set), n_classes=7806)
    return tiles, label
```

`tile_to_species` is a precomputed dictionary built once before training,
populated by FAISS retrieval (see §3.4). Sampler A **must not** consume it —
the asymmetry is the experimental contrast.

Place samplers at `user/murilo/src/plantclef/data/samplers.py`. Wrap as a
PyTorch `IterableDataset` at `experiments/003-tileq-decoder/dataset.py`.

### 2.4 Inference

At test time:

1. Load cached `shared/inference/dinov2-vit-base/cls_grid_4x4.parquet`
   (33,680 rows = 2,105 images × 16 tiles, 768-d float16).
2. Reconstruct per-image `(16, 768)` tensors keyed on `image_name`.
3. Forward each through the trained decoder → 7806 logits.
4. Sigmoid → per-class probabilities.
5. *(Optional, see §2.6)* apply SW-Europe geo mask before top-k.
6. Top-k decoding with `k = 10` (matches the existing baseline submission
   format).
7. Map class indices → Pl@ntNet species IDs via `class_to_species_id.json`.
8. Write CSV with columns `quadrat_id; species_ids` matching the Kaggle
   submission format used by exp 002.

For v1, **do not** apply per-class thresholding — top-k=10 alone, optionally
geo-masked. Per-class F1-optimal thresholding is listed as out-of-scope (§8).

### 2.5 Score-Blend (Bonus Ablation)

If time permits, produce additional submissions with:

```
s_c = α · ŷ_c^{decoder} + (1 − α) · max_t p_{t,c}^{cached}
```

for `α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`. No retraining — pure NumPy combination
of the decoder's per-image probability vector with the existing cached
`max_prob` baseline (`grid_4x4.parquet`). This tests whether the decoder is
**complementary** to or **redundant** with the existing aggregator.

Place at `experiments/003-tileq-decoder/blend.py`.

### 2.6 Geo-Mask (Independent Reconstruction, Optional)

Alper's `sw_europe_mask_bool.npy` is out of scope for the independence framing.
If geo-masking is used in v1, rebuild from scratch:

```python
# experiments/003-tileq-decoder/build_geo_mask.py
import pandas as pd, numpy as np
from pathlib import Path

train_meta = pd.read_csv(
    "/storage/project/ps-clef2026_plantclef-0/shared/data/raw/"
    "metadata_PlantCLEF_training_set/PlantCLEF2024singleplanttrainingdata.csv",
    sep=";", low_memory=False,
)
sw_europe = train_meta[
    (train_meta.latitude.between(35, 50)) &
    (train_meta.longitude.between(-10, 10))
]
sw_species = set(sw_europe.species_id.dropna().astype(int))

class_mapping_path = "/storage/project/ps-clef2026_plantclef-0/shared/data/raw/class_id_to_species_mapping/class_mapping.txt"
class_to_species = [int(line.strip()) for line in open(class_mapping_path)]
mask = np.array([sid in sw_species for sid in class_to_species], dtype=bool)
np.save("artifacts/sw_europe_mask.npy", mask)
print(f"Mask: {mask.sum()} / {len(mask)} species in SW Europe")
```

Recommended default for v1: **omit** the mask. Keeps the decoder→submission
path the cleanest possible comparison and removes one confound.

---

## 3. Data and Cached Artefacts

### 3.1 Test-Set CLS Extraction (required, ~2 GPU-h)

Extend `experiments/001-dinov2-grid-ablation/inference.py` with a `--save-cls`
flag. The forward pass currently does:

```python
logits = model(flat_tiles)              # (B*T, 7806)
probs = torch.softmax(logits, dim=1)
```

Add the CLS branch:

```python
if args.save_cls:
    feats = model.forward_features(flat_tiles)              # tokens
    cls = model.forward_head(feats, pre_logits=True)        # (B*T, 768)
    # ... save alongside probs (or instead of, for grid 4x4 specifically)
```

Output schema:

| column | type | notes |
|---|---|---|
| `image_name` | `string` | matches `grid_4x4.parquet` |
| `tile_idx` | `int32` | 0..15 for grid 4×4 |
| `cls_embedding` | `binary` | float16 bytes, shape (768,); reconstruct with `np.frombuffer(b, np.float16)` |

Single grid: 4×4 only for v1. Total rows: 2,105 × 16 = 33,680. Output:
`shared/inference/dinov2-vit-base/cls_grid_4x4.parquet` (alongside the existing
10 prob parquets, so reusable by future experiments).

Run via `experiments/003-tileq-decoder/sbatch/extract_test_cls.sbatch`.

### 3.2 Train-Side CLS (already cached)

Path: `shared/embeddings/train/sample_id={0..19}/`, hive-partitioned, 400 part
files, 1,408,033 rows. Schema:

| column | type | notes |
|---|---|---|
| `image_name` | `string` | joins to `PlantCLEF2024singleplanttrainingdata.csv` |
| `species_id` | `int32` | range `[1355868, 1744937]`, 7806 unique |
| `cls_embedding` | `list<float32>[768]` | **raw** (mean L2 norm ≈18.6, std ≈3.2) — must L2-normalise before any cosine / `IndexFlatIP` retrieval |

Build the per-species lookup once at training start:

```python
import pyarrow.dataset as ds
import numpy as np
from collections import defaultdict

table = ds.dataset(
    "/storage/project/ps-clef2026_plantclef-0/shared/embeddings/train",
    partitioning="hive",
    format="parquet",
).to_table(columns=["species_id", "cls_embedding"])

sids = table["species_id"].to_numpy()
embs = np.asarray(table["cls_embedding"].to_pylist(), dtype=np.float32)  # (1.4M, 768)

# class_idx <-> species_id mapping (apply once here, never per-batch)
class_mapping_path = "/storage/project/ps-clef2026_plantclef-0/shared/data/raw/class_id_to_species_mapping/class_mapping.txt"
species_id_to_class_idx = {
    int(line.strip()): i for i, line in enumerate(open(class_mapping_path))
}

train_cls_by_class_idx = defaultdict(list)
for sid, e in zip(sids, embs):
    train_cls_by_class_idx[species_id_to_class_idx[int(sid)]].append(e)
train_cls_by_class_idx = {k: np.stack(v) for k, v in train_cls_by_class_idx.items()}
# → dict[class_idx 0..7805] -> ndarray(n_i, 768) float32
```

Memory cost: ~4.3 GB held in RAM. Workable on a CPU node; comfortable on a
GPU node with 64 GB system RAM. Persist to `artifacts/train_cls_by_class_idx.npz`
to avoid re-paying the load cost on subsequent runs.

Place builder at `experiments/003-tileq-decoder/build_train_lookup.py`.

### 3.3 Class-Frequency Weights

Compute once from the cached train table: `count[class_idx]` = number of rows
per species after the species_id → class_idx remap. Sampling weight
`w_s ∝ 1 / √count[s]`. Normalise to a probability distribution.

Empirical stats (full 1.4 M):

| stat | value |
|---|---|
| species count | 7,806 |
| rows / species — median | 176 |
| rows / species — mean / max | 180 / 823 |
| rows / species — p10 / p90 | 6 / 389 |
| species with <5 images | 609 (7.8 %) |
| species with <10 images | 1,062 (13.6 %) |

The square-root softens the imbalance without exploding the rare-class weight.

Persist to `artifacts/species_weights.npy` (shape `(7806,)`, float32, sums to
1).

### 3.4 FAISS Background-Cleaning Map for Sampler B

One-time preprocessing, before Sampler B training begins.

```
1. L2-normalise all 1,408,033 train CLS embeddings.
2. Build  faiss.IndexFlatIP  over the normalised train vectors. (~4.3 GB RAM.)
3. L2-normalise all 33,680 test tile CLS embeddings (from §3.1).
4. For each test tile, query top-1 nearest train neighbour → (sim, idx).
5. Plot histogram of top-1 sim across all 33,680 tiles.
6. Pick τ at the visible "shoulder" of the distribution. Default τ = 0.85.
7. If sim ≥ τ : tile_to_species[(quadrat_idx, tile_idx)] = train_species_id_for(idx)
   Else      : tile_to_species[(quadrat_idx, tile_idx)] = None  # treat as true background
```

Document the chosen τ in `artifacts/faiss_threshold.txt`. Save the map at
`artifacts/tile_to_species.pkl` and the histogram figure at
`artifacts/tile_top1_sim_histogram.png`.

Estimated cost: ~5 min on a CPU node (FAISS-CPU `IndexFlatIP` over 1.4 M × 768
+ 33,680 queries). Can also run on GPU FAISS for sub-minute, but unnecessary.

Place at `experiments/003-tileq-decoder/build_faiss_background_map.py`.

---

## 4. Training

### 4.1 Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, weight_decay 0.05 |
| Learning rate | 3e-4 peak |
| Warmup | 1000 steps linear from 0 → 3e-4 |
| Schedule | cosine decay 3e-4 → 1e-5 over remaining steps |
| Batch size | 32 (RTX 6000) or 64 (H100) |
| Total steps | 80,000 |
| Loss | ASL (γ⁻=4, γ⁺=0, m=0.05) |
| Gradient clipping | 1.0 |
| Mixed precision | bfloat16 |
| Seed | 42 (logged per run) |

No staged warmup phase — train directly on pseudo-quadrats from step 0.

Training entrypoint: `experiments/003-tileq-decoder/train.py`. Hyperparameters
serialised to YAML alongside each checkpoint.

### 4.2 Validation During Training

Hold out 1 % of training images by `image_name` hash (deterministic). Build
a validation pseudo-quadrat set (~2,000 examples) once at training start, using
the **same** sampler as the training run. Compute multi-label cross-entropy
plus a coarse "any-positive-found" accuracy every 2,000 steps.

This is a **training-loss sanity check, not a generalisation proxy.** Do not
use it for thresholding, model selection, or hyperparameter search. Each Kaggle
submission is the only evaluation channel for test-set performance. Kaggle
allows ~5 submissions/day → 25 over 5 days; the budget comfortably covers
A0a/A0b/A1/A2 plus the 5-row A3 sweep.

### 4.3 Sanity Checks (run before any long training)

1. **Overfit-one-batch.** Sample 32 pseudo-quadrats once, freeze the data,
   train on this single batch for 1,000 steps. Loss should decrease to near
   zero. If it does not, the model or loss is broken — debug before any
   longer run.
2. **Shape audit.** Print every tensor shape through one forward + backward
   pass at startup. Catch mismatches before they waste GPU hours.
3. **Sampler unit test.** For each sampler, verify:
   - Injected species CLSs land in the correct slots.
   - The multi-hot label matches the species set.
   - For Sampler B: a synthetic test where a known species CLS is retrieved
     from a known tile produces the expected augmented label.

### 4.4 Two Training Runs

| Run | Sampler | Output checkpoint |
|---|---|---|
| Run 1 | A | `artifacts/checkpoints/decoder_samplerA.pt` |
| Run 2 | B | `artifacts/checkpoints/decoder_samplerB.pt` |

Same hyperparameters, same seed, same total steps, same val-set construction.
Save checkpoint every 10 k steps + a final at 80 k. Both runs are
background-able. Launch Run 1 as soon as sanity checks pass; Run 2 in parallel
on a second GPU (or sequentially on the same one).

---

## 5. Submissions and Ablation Table

The paper's experiments table:

| # | Variant | Decoder | Sampler | Score blend | Notes |
|---|---|:---:|:---:|:---:|---|
| **A0a** | Cached `max_prob` top-10, grid 4×4 | — | — | — | Generated **first** via `experiments/002-classification-methods/classify.py --grid-size 4 --method max-prob`. Validates submission pipeline. Expected ≈ 0.23. |
| **A0b** | Cached sorted-prob top-9, grid 4×4 | — | — | — | Reproducible 2025 baseline. F1 = 0.30810 (validated, exp 002). |
| **A1** | TileQ-Decoder, Sampler A | ✓ | A | 1.0 (decoder only) | Tests **H1**: does the decoder beat the cached aggregators without test-side knowledge? |
| **A2** | TileQ-Decoder, Sampler B | ✓ | B | 1.0 (decoder only) | Tests **H2**: does test-anchored sampling help? |
| **A3** *(bonus)* | Score-blend, best of {A1,A2} + cached `max_prob` | ✓ | best | sweep α ∈ {0, 0.25, 0.5, 0.75, 1.0} | Tests complementarity. 5 CSVs. |

**A0a/A0b must be produced before training starts** to derisk the submission
pipeline. A1 and A2 are the headline rows. A3 is run only if the schedule
permits.

CSV naming convention: `submissions/{tag}_{seed}.csv` where tag ∈ {A0a, A0b,
A1_samplerA_seed42, A2_samplerB_seed42, A3_alpha0p25, ...}.

---

## 6. Repository Structure

Final layout for this experiment, anchored to the existing repo conventions
(`user/<name>/src/plantclef/` for shared library code; experiment-specific
scripts in `experiments/NNN-name/`):

```
user/murilo/
├── src/plantclef/                     # shared library (existing + new)
│   ├── model.py                       # existing — load_model(), load_class_mapping()
│   ├── tiling.py                      # existing — split_into_tiles()
│   ├── inference/
│   │   └── dataset.py                 # existing — TileDataset
│   ├── tileq_decoder.py               # NEW — TileQDecoder (§2.1)
│   ├── asl_loss.py                    # NEW — AsymmetricLoss (§2.2)
│   └── data/
│       └── samplers.py                # NEW — Sampler A & B (§2.3)
│
└── experiments/003-tileq-decoder/
    ├── proposal.md
    ├── spec.md                        # this file
    ├── tasks.md
    ├── pyproject.toml
    ├── src -> ../../src               # symlink (already created)
    │
    ├── extract_test_cls.py            # §3.1 — calls into shared model.py + dataset.py;
    │                                  #        OR add --save-cls flag to exp-001 inference.py
    ├── build_train_lookup.py          # §3.2
    ├── build_faiss_background_map.py  # §3.4
    ├── build_geo_mask.py              # §2.6 — optional
    │
    ├── dataset.py                     # PyTorch IterableDataset wrapping samplers
    ├── train.py                       # §4
    ├── infer.py                       # §2.4
    ├── blend.py                       # §2.5
    │
    ├── sbatch/
    │   ├── extract_test_cls.sbatch
    │   ├── build_faiss_background_map.sbatch
    │   ├── train_samplerA.sbatch
    │   ├── train_samplerB.sbatch
    │   └── infer.sbatch
    │
    ├── artifacts/                     # produced locally
    │   ├── species_weights.npy
    │   ├── train_cls_by_class_idx.npz
    │   ├── tile_to_species.pkl
    │   ├── tile_top1_sim_histogram.png
    │   ├── faiss_threshold.txt
    │   ├── sw_europe_mask.npy        # optional, only if §2.6 used
    │   └── checkpoints/
    │       ├── decoder_samplerA.pt
    │       └── decoder_samplerB.pt
    │
    ├── submissions/                   # Kaggle CSVs
    │   ├── A0a_max_prob_grid4.csv
    │   ├── A0b_sorted_prob_topk9_grid4.csv
    │   ├── A1_samplerA_seed42.csv
    │   ├── A2_samplerB_seed42.csv
    │   └── A3_alpha{0p00,0p25,0p50,0p75,1p00}.csv
    │
    └── results.md                     # filled out post-execution
```

Cached test CLS goes to `shared/inference/dinov2-vit-base/cls_grid_4x4.parquet`
(reusable across experiments), not into this experiment's `artifacts/`.

---

## 7. Expected Failure Modes

For each: detection signal and mitigation.

1. **Mode collapse on common species.** Sign: macro-F1 ≪ micro-F1; predictions
   concentrated on ~50 species. Mitigation: verify `1/√count` species
   sampling; raise weight floor; consider a bias-init trick (zero out the
   final bias for rare classes).
2. **Background bleed / over-prediction.** Sign: average predictions per
   quadrat ≫ expected ~5–15. Mitigation: tighten top-k or add a per-class
   threshold floor (out of scope for v1); consider an abstain-query in v2.
3. **Synthetic-distribution overfit (Sampler A).** Sign: validation loss
   decreases but test F1 flat or worse than A0 baseline. Mitigation: this is
   exactly what Sampler B is designed to address; if A0 > A1 ≈ A2 the
   conclusion is "synthetic supervision did not transfer" — a publishable
   negative result.
4. **Group-FFN class-assignment artefacts.** Sign: macro-F1 has periodic
   structure when sorted by class index (rare classes in the same group as
   common ones get starved). Mitigation: shuffle class-to-group assignment
   each epoch (1-line change in `TileQDecoder` indexing).
5. **FAISS τ misset (§3.4).** Sign: Sampler B's effective label count per
   pseudo-quadrat is much higher than A's (over-relabelling, retrievals from
   the long tail being too generous) or roughly equal (under-relabelling,
   threshold too strict). Mitigation: inspect histogram; retune τ; sanity
   check by spot-comparing tile thumbnails to top-1 retrieved train images.
6. **Submission CSV format mismatch.** Sign: Kaggle rejects the file.
   Mitigation: A0a / A0b are produced **before** any training time is
   invested. They share the same writer with A1/A2/A3.
7. **Embedding-scale collapse.** Sign: cross-attn produces uniform attention
   weights regardless of input. Mitigation: verify `proj_kv` actually adapts
   the raw (non-normalised, mean L2 ≈ 18.6) CLS scale; LayerNorm after the
   projection optional but cheap insurance.

---

## 8. Out of Scope (Future Work)

Listed in the paper's Discussion; **not** in this 5-day experiment.

- DINOv2 attention-rollout-based selection of background tiles. Cleaner anchor
  identification than FAISS-based detection.
- Geographic / habitat co-occurrence sampling (LUCAS land-cover priors). The
  77K SW-Europe LUCAS records are sitting in `metadata_lucas/` ready to use.
- Abstain query (K+1 logit absorbing background tiles). Allows the model to
  explicitly predict "no species here" rather than spreading mass.
- Per-class F1-optimal thresholding (Lipton et al. ECML-PKDD 2014).
- Multiscale tile inputs (3×3, 5×5, 6×6 grids — currently only 4×4 used).
  Would require fresh CLS extraction at each scale.
- Ensembling with the existing pipeline beyond simple α-blend.
- Variation of `n_positives` per pseudo-quadrat (currently fixed at 5).
- End-to-end fine-tuning (decoder + backbone). Currently backbone is fully
  frozen.

---

## 9. Reproducibility Checklist

- [ ] Every random seed logged at run start (numpy, torch, python).
- [ ] Hyperparameters serialised to YAML alongside each checkpoint.
- [ ] Training data (the `tile_to_species` map for Sampler B, the train-CLS
      lookup) saved with deterministic build scripts; both build scripts
      print the input file modification times so the result is auditable.
- [ ] Submission CSVs versioned by run (filename includes sampler + seed).
- [ ] Loss curves saved (CSV log + `matplotlib` plot in `artifacts/`).
- [ ] Final paper appendix lists wall-clock times per stage (extract test CLS,
      build train lookup, build FAISS map, train run 1, train run 2,
      inference).
- [ ] All file paths in scripts are absolute or anchored to the experiment
      directory; no `cd` dependencies.

---

## 10. References (for the eventual paper)

- Liu, S. et al. *Query2Label: A Simple Transformer Way to Multi-Label Classification.* arXiv 2107.10834, 2021.
- Ridnik, T. et al. *ML-Decoder: Scalable and Versatile Classification Head.* WACV 2023.
- Ridnik, T. et al. *Asymmetric Loss for Multi-Label Classification.* ICCV 2021.
- Cole, E. et al. *Multi-Label Learning from Single Positive Labels.* CVPR 2021.
- Zhou, D. et al. *Acknowledging the Unknown for Multi-label Learning with Single Positive Labels.* ECCV 2022.
- Bochkovskiy, A. et al. *YOLOv4: Optimal Speed and Accuracy of Object Detection.* arXiv 2004.10934, 2020 (Mosaic augmentation).
- Oquab, M. et al. *DINOv2: Learning Robust Visual Features without Supervision.* arXiv 2304.07193, 2023.
- Goëau, H. et al. *Overview of PlantCLEF 2024/2025.* Working notes, CEUR-WS.
- Lipton, Z. et al. *Optimal Thresholding of Classifiers to Maximize F1 Measure.* ECML-PKDD 2014.

---

## 11. Path Reference (absolute paths, copy-paste safe)

```
# Inputs
ViT-B checkpoint       /storage/project/ps-clef2026_plantclef-0/shared/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth
class_mapping.txt      /storage/project/ps-clef2026_plantclef-0/shared/data/raw/class_id_to_species_mapping/class_mapping.txt
class_to_species.json  /storage/project/ps-clef2026_plantclef-0/shared/data/raw/class_id_to_species_mapping/class_to_species_id.json
species_id_to_name.txt /storage/project/ps-clef2026_plantclef-0/shared/data/raw/class_id_to_species_mapping/species_id_to_name.txt
test images            /storage/project/ps-clef2026_plantclef-0/shared/data/raw/PlanCLEF_test_data/images/*.jpg
test metadata          /storage/project/ps-clef2026_plantclef-0/shared/data/raw/PlanCLEF_test_data/PlantCLEF2025_test.csv
train metadata         /storage/project/ps-clef2026_plantclef-0/shared/data/raw/metadata_PlantCLEF_training_set/PlantCLEF2024singleplanttrainingdata.csv
train CLS (1.4 M)      /storage/project/ps-clef2026_plantclef-0/shared/embeddings/train/sample_id={0..19}/part-*.parquet
test probs (grid 4×4)  /storage/project/ps-clef2026_plantclef-0/shared/inference/dinov2-vit-base/grid_4x4.parquet
LUCAS SW-Europe        /storage/project/ps-clef2026_plantclef-0/shared/data/raw/metadata_lucas/LUCAS_metadata_SWE_Wide_lat35_50_lon-10_10_77k.csv

# Outputs (this experiment)
test CLS (grid 4×4)    /storage/project/ps-clef2026_plantclef-0/shared/inference/dinov2-vit-base/cls_grid_4x4.parquet  # produced by §3.1
checkpoints, artifacts user/murilo/experiments/003-tileq-decoder/artifacts/
submissions            user/murilo/experiments/003-tileq-decoder/submissions/
```
