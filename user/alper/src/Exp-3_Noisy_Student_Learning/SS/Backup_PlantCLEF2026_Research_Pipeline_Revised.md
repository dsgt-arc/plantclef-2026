# PlantCLEF 2026 — Research Pipeline (Revised)

## Context

**Goal**: Train a CrossRegionTransformer student model that outperforms the teacher using pseudo-labels from the unlabeled LUCAS dataset. Primary contribution: learned tile aggregation replaces heuristic methods. Paper requires 10%+ improvement with ablation evidence.

**Starting point**: Pretrained ViT classifier (7806 species, Zenodo PlantCLEF model). LUCAS dataset (77k-212k unlabeled vegetation images). CrossRegionTransformer architecture already designed.

---

## Phase 1 — Teacher Inference Pipeline

**Purpose**: Generate reliable multi-species predictions for vegetation plots.

**Execution order**: Run Phase 1 on the **test set first** (~2 GPU-hrs), then Phase 2 (teacher benchmark + Kaggle submit). Only after confirming teacher F1 >= 0.24, run Phase 1 on LUCAS (~18 GPU-hrs). This gives early feedback before committing GPU time to the full LUCAS extraction.

### 1.1 Image Tiling

For each image:
- Load original image
- Split into 4x4 grid = 16 tiles
- **Tile first, resize after** — each tile resized to 518x518

### 1.2 Weak Augmentation Consistency Check

For each tile, run teacher inference on:
- Original tile
- Horizontal flip

Compute top-16 species lists for both. Keep tile only if:
```python
overlap = len(set(top16_orig) & set(top16_flip))
keep = overlap >= threshold  # start with threshold=8, tune later
```

Unstable tiles (background, blur, ambiguous) are discarded.

### 1.3 Tile Inference + Multi-View Embedding Extraction (3 Views)

For each **kept** tile (passed aug consistency in 1.2), run 3 forward passes and store:
- `image_id`, `tile_id`
- `top64_class_ids`, `top64_probs` (from original view)
- `max_prob` (highest softmax score for this tile)
- `embedding_orig` (768-dim DINOv2 CLS token from original tile)
- `embedding_flip` (768-dim DINOv2 CLS token from horizontally flipped tile)
- `embedding_rot90` (768-dim DINOv2 CLS token from 90°-rotated tile)

**View rationale**:
- **Original**: baseline view
- **Horizontal flip**: already forwarded in Phase 1.2 for consistency check — zero extra cost
- **90° rotation**: strongest natural augmentation for top-down vegetation plots (no canonical "up"). Completely reshuffles spatial tile arrangement while preserving species content. Forces the student to learn content-based aggregation, not position shortcuts.

**Why 3 views, not more**: 180° and 270° are just combinations of flip + 90°. Color jitter introduces randomness (non-reproducible). Random crop changes tile content (breaks species invariance). 3 views hits the sweet spot of signal vs extraction cost.

**Extraction cost**: 3x forward passes per tile. Since flip is already done in Phase 1.2, the additional cost is just the rot90 pass — ~50% more GPU time than the 2-view version.

All 3 views are used during student training for consistency regularization (see Phase 4.2).

### 1.4 Tile Confidence Filtering

Per image, rank tiles by `max_prob`. Keep only the top-K most confident tiles:
```python
--keep-top-tiles K  # default K=8, sweep {4, 6, 8, 12, 16} later
```

This removes background/soil tiles that dilute the aggregation signal.

### 1.5 Noisy-OR Aggregation with Temperature

Aggregate kept-tile predictions into image-level predictions:
```python
p_tile = softmax(logits / T)  # temperature T, default T=2.0
P_species = 1 - prod(1 - p_tile_species)  # over kept tiles only
```

Aggregation covers the **union of tile top-64 candidates** only (not all 7806).

**Sweep T**: {1.0, 1.5, 2.0, 3.0} during Phase 2.

### 1.6 Geo-location Species Masking

```python
species_probabilities[not_in_region] = 0
```

Use the same geographic mask from the PlantCLEF baseline. Note: geo-masking adds only ~0.006-0.008 F1 — it's a minor factor, not the main lever.

**Order matters**: Geo-masking MUST be applied after Noisy-OR aggregation (Phase 1.5) but before top-10 selection (Phase 1.7). Masking before aggregation would incorrectly remove valid high-probability candidates from the Noisy-OR computation.

### 1.7 Final Image Prediction

Sort by aggregated probability, keep **top 10** species per image (top-10 performed best in last year's reference paper).

---

## Phase 2 — Teacher Benchmark

**Purpose**: Verify the teacher pipeline before training the student.

### 2.1 Run Teacher on Test Dataset

Full pipeline: tiling → aug consistency → tile filtering → Noisy-OR → geo mask → top-10.

### 2.2 Kaggle Submission

Submit and record F1 score.

**Minimum threshold**: teacher F1 >= 0.24. If significantly lower, debug before proceeding:
- Check image loading (RGB vs BGR, correct resize)
- Check tile iteration order (column-first matching reference)
- Check softmax is float32, not float16
- Verify class_mapping (class_index → species_id)

### 2.3 Temperature Sweep (CPU-only)

Re-aggregate existing tile predictions with T = {1.0, 1.5, 2.0, 3.0}. Submit each. Pick best temperature for all subsequent phases.

### 2.4 Teacher Baseline Table (for paper)

Run teacher with systematic variations to build baseline table:

| Config | Aug Filter | Tile Filter | Temperature | Geo |
|--------|-----------|-------------|-------------|-----|
| T1 | No | All 16 | T=2 | Yes |
| T2 | Yes | All 16 | T=best | Yes |
| T3 | Yes | Top-6 | T=best | Yes |
| T4 | Yes | Top-6 | T=best | No |

This shows the incremental value of each teacher improvement.

---

## Phase 3 — Pseudo-Label Generation (LUCAS)

**Purpose**: Create training labels for the student.

### 3.1 Run Teacher Pipeline on LUCAS

Same pipeline as Phase 2 best config, applied to LUCAS dataset:
- Tiling → aug consistency → tile filtering → Noisy-OR → geo mask

Also store **tile embeddings** (768-dim) for each kept tile — these are the student's input.

### 3.2 Pseudo-label Quality Filtering

**Step A**: Per image, keep species with aggregated probability above a threshold, capped at 10:
```python
keep = [s for s in species if P_species[s] > min_prob]  # default min_prob=0.05
keep = keep[:10]  # safety ceiling
# Sweep min_prob: {0.02, 0.05, 0.1}
```
This ensures images with 2 real species get 2 pseudo-labels, not 10 padded with noise.

**Step B**: Per species across the full dataset:
```python
for species in all_predicted_species:
    thresh = np.percentile(all_probs_for_species, 99.5)
    keep only predictions with prob > thresh
```

Step A removes per-image noise (low-confidence guesses). Step B removes per-species noise (species the teacher is generally uncertain about).

### 3.3 Pseudo-label Statistics Check

Before proceeding, verify:
- **Average labels per image**: target 2-5 (typical vegetation plots have 2-4 species, rarely 6+). If <1 → filtering too strict. If 8+ → too loose.
- **Species coverage**: how many of 7806 species have any pseudo-labels? Target 2000+.
- **Confidence distribution**: plot histogram of pseudo-label probabilities.

### 3.4 Output Dataset

Each training example:
- `image_id`
- `tile_embeddings_orig`: [N_kept_tiles x 768] (variable N per image)
- `tile_embeddings_flip`: [N_kept_tiles x 768] (flipped view for consistency loss)
- `tile_embeddings_rot90`: [N_kept_tiles x 768] (90°-rotated view for consistency loss)
- `tile_mask`: boolean mask for padding (since N varies)
- `pseudo_labels`: list of positive species IDs only

**Key**: Pseudo-labels are **positive only**. All other species are **unknown, not negative**.

---

## Phase 4 — Student Training

**Purpose**: Learn better aggregation than the heuristic teacher.

### 4.1 Model: CrossRegionTransformer

**Input**: Tile embeddings [N_tiles x 768] with attention mask for variable-length sequences
**Output**: Multi-label species probabilities [7806]

Handle variable tile count:
- Pad tile sequences to max length (16)
- Use attention mask so transformer ignores padded positions
- CLS token or attention pooling aggregates across real tiles only

### 4.2 Training Objective — Three Loss Components

The student is trained with a composite loss where two complementary losses serve fundamentally different roles:

- **Pseudo-label loss** provides the **initial learning signal** — without it, the student has no notion of which species exist. It bootstraps the student's knowledge from the teacher's predictions. However, the student is bounded by the teacher's accuracy here: it can only be as good as the pseudo-labels.
- **Multi-view consistency loss** is **how the student surpasses the teacher**. It provides a teacher-independent, self-supervised signal that forces the student to learn robust, content-based aggregation. This is a clean signal even when pseudo-labels are noisy, because it doesn't depend on the teacher at all — only on the geometric invariance of species identity under view transforms.

**Training dynamics**: In early epochs, pseudo-label loss dominates (the student is learning what species look like). As training progresses, the consistency loss becomes increasingly important — it regularizes the student's representations and pushes it beyond the teacher's ceiling by learning patterns the heuristic Noisy-OR aggregation cannot capture.

**Why the student can surpass the teacher**: The teacher uses a fixed heuristic (Noisy-OR) that treats all tiles independently. The student's transformer can learn tile interactions — e.g., "if tile A contains leaf shape X and tile B contains flower Y, they likely belong to the same species." The consistency loss ensures these learned interactions are robust to spatial rearrangement (rotation) and reflection (flip), not just memorized from the training set.

---

**Component 1: Pseudo-Label Loss (two variants — ablation)**

*Purpose*: Bootstrap the student's species knowledge from the teacher. This is the student's only source of "what species are in this image" — without it, training cannot begin.

*Variant A: Negative Sampling + BCE*
```python
positive_species = pseudo_labels  # from Phase 3
negative_species = random_sample(all_species - positive_species, n=64)
L_pseudo = BCE(pred[positives], 1.0) + BCE(pred[negatives], 0.0)
```

*Variant B: Soft Distillation (KL-divergence)*
```python
teacher_probs = noisy_or_output  # continuous values from teacher
L_pseudo = KL_div(log(student_probs), teacher_probs)
```

Compare both in ablation — this is itself a research finding.

**Recommended default: Variant B (KL-div)**. KL-div preserves the teacher's uncertainty across the top-64 candidates rather than collapsing to hard 0/1 targets. This gives the student richer supervision — it learns not just "species X is present" but "species X is likely, species Y is possible, species Z is unlikely." This soft signal is especially important early in training when the student needs nuanced guidance. Negative sampling (Variant A) risks injecting false negatives (a "negative" species might actually be present but not in the pseudo-label set). Still run Variant A for the ablation comparison.

**Component 2: Multi-View Consistency Loss (core component — key to surpassing teacher)**

*Purpose*: This is the student's path to outperforming the teacher. It provides a **teacher-independent** learning signal based on a simple invariance: the same scene viewed from different angles contains the same species.

Adapted from FixMatch/MeanTeacher for embedding-space training, using all 3 views from Phase 1.3:
```python
# All 3 views extracted in Phase 1.3
pred_orig = student(embedding_orig_tiles)   # original view
pred_flip = student(embedding_flip_tiles)   # flipped view
pred_rot  = student(embedding_rot90_tiles)  # 90°-rotated view

# Student predictions should be consistent across all views
L_consistency = (MSE(pred_orig, stop_gradient(pred_flip))
               + MSE(pred_orig, stop_gradient(pred_rot))) / 2
```

This is applied to ALL training images, not just those with pseudo-labels. Even when pseudo-labels are noisy, consistency regularization provides a clean self-supervised signal: "if the input is the same scene under a minor perturbation, the predicted species should not change."

**Why this works**: Neither flipping nor rotating a top-down vegetation plot changes which species are in it. The 90° rotation is a particularly strong perturbation — it completely reshuffles the spatial layout of tiles, forcing the student to learn **content-based** aggregation rather than memorizing positional patterns. This makes the consistency signal much richer than flip alone.

**Why the teacher can't do this**: The teacher's Noisy-OR aggregation is a fixed formula — it has no parameters to optimize and no way to learn from consistency signals. The student's transformer weights are updated by both losses simultaneously, allowing it to discover tile-interaction patterns that are both species-predictive (from pseudo-labels) and view-invariant (from consistency).

**Component 3: Embedding Dropout (regularization)**
```python
# Applied to input embeddings during training
self.embed_dropout = nn.Dropout(0.1)
tile_embeddings = self.embed_dropout(tile_embeddings)
```

Free regularization — prevents overfitting to specific embedding features.

**Combined Loss with Consistency Warmup:**
```python
# Pseudo-label loss: fixed weight 1.0 throughout (anchor signal)
# Consistency loss: linear ramp from 0 → lambda_cons over warmup epochs
# Co-graph loss: fixed small lambda

lambda_cons_t = min(1.0, epoch / warmup_epochs) * lambda_cons

L_total = L_pseudo + lambda_cons_t * L_consistency + lambda_co * L_co_graph
```

**Why warmup**: In early epochs the student knows nothing — pseudo-label loss must dominate so it learns basic species recognition. As training progresses, consistency loss ramps up to push the student beyond the teacher's ceiling. This follows FixMatch/MeanTeacher convention.

**Defaults:**
- `lambda_cons`: 0.1 (sweep {0.01, 0.05, 0.1, 0.5})
- `warmup_epochs`: 5 (sweep {3, 5, 8})
- `lambda_co`: 0.01 (sweep {0.001, 0.01, 0.05})

### 4.3 Training Configuration

- **Optimizer**: AdamW, lr=1e-4, weight_decay=0.01
- **Epochs**: 10-20 with early stopping on training loss plateau (no labeled validation set — LUCAS is fully unlabeled)
- **Batch size**: tune to fit GPU memory
- **True evaluation**: Kaggle F1 is the only ground truth signal
- **Compute**: ~4-6 GPU-hours per run

### 4.4 Ablation Runs

| ID | Configuration | What It Proves | GPU-hrs |
|----|---------------|----------------|---------|
| A0 | Teacher baseline (best from Phase 2) | Heuristic aggregation ceiling | 0 |
| A1 | Student, neg sampling BCE, no consistency | Learned agg > heuristic (minimal) | 5 |
| A2 | Student, soft distillation KL, no consistency | Which pseudo-label loss is better | 5 |
| A3 | Best loss + multi-view consistency | Consistency regularization helps | 5 |
| A4 | A3 + aug consistency filtering (Phase 1.2) | Filtering unstable tiles helps | 5 |
| A5 | A4 + top-K tile filtering (Phase 1.4) | Removing background tiles helps | 5 |
| A6 | A5 + species percentile filtering (Phase 3.2) | Full filtering pipeline | 5 |
| A7 | A6 + co-graph consistency loss | Ecological priors help | 5 |
| A8 | A7 + 1 round self-training (Phase 6) | Self-training adds value | 7 |
| A9 | A8 + 2nd round self-training | Diminishing returns | 7 |

**Total ablation cost**: ~49 GPU-hrs (~6-7 days at 8 hrs/day)

**Ablation story for paper**: Each row adds one component → monotonically increasing F1 → proves every piece matters.

---

## Phase 5 — Student Inference

Run the trained student on the test dataset:
1. 4x4 tiling (same as teacher)
2. Extract tile embeddings (frozen DINOv2)
3. CrossRegionTransformer aggregation
4. Geo species mask
5. Top-10 output

Generate Kaggle submission. Record F1.

**Key comparison**: Student F1 vs Teacher F1 (Phase 2). Target: 10%+ relative improvement.

---

## Phase 6 — Self-Training

**Purpose**: Improve pseudo-labels using the trained student.

### 6.1 Round 1: Student Generates New Pseudo-Labels

- Run best student (from Phase 4) on LUCAS
- Apply same filtering pipeline (top-10, 99.5 percentile)
- Compare pseudo-label stats with teacher pseudo-labels:
  - More species per image? Higher confidence? Better coverage?

### 6.2 Retrain Student

Train a new student on the updated pseudo-labels.
Same architecture, same loss, same config.

### 6.3 Round 2 (If Round 1 Improved)

Repeat 6.1-6.2. Track F1 per round:
```
Teacher → Student R0 → Student R1 → Student R2
```

Stop when F1 improvement < 0.005 between rounds.

### 6.4 Self-Training Statistics (for paper)

Track across rounds:
- Pseudo-label count and avg labels/image
- Species coverage
- Confidence distribution shift
- F1 on Kaggle

---

## Phase 7 — Final Evaluation + Paper

### 7.1 Hyperparameter Sensitivity (CPU-mostly)

| Sweep | Values | Phase |
|-------|--------|-------|
| Noisy-OR temperature T | {1.0, 1.5, 2.0, 3.0} | Teacher |
| Aug consistency threshold | {4, 8, 12} overlap out of 16 | Filtering |
| Keep-top-tiles K | {4, 6, 8, 12, 16} | Filtering |
| Percentile cutoff | {99.0, 99.5, 99.9} | Pseudo-labels |
| Negative sample count | {32, 64, 128} | Student training |
| Co-graph lambda | {0.001, 0.01, 0.05} | Student training |
| Consistency lambda | {0.01, 0.05, 0.1, 0.5} | Student training |
| Submission top-K | {4, 6, 8, 10, 12} | Output |

### 7.2 Qualitative Analysis (for paper figures)

| Analysis | How | Shows |
|----------|-----|-------|
| Attention heatmap | Extract pooling_attention weights, overlay on 4x4 grid | Student learns which tiles are informative |
| Consistency improvement | Show cases where multi-view loss corrected predictions | Augmentation regularization works |
| Co-graph case study | Diff predictions with/without co-graph | Ecological reasoning |
| Self-training convergence | Plot F1 + pseudo-label stats per round | Iterative improvement |
| Per-species P/R | Bin by frequency, compare teacher vs student | Where student helps most |
| Efficiency table | Wall-clock times per stage | 100x cheaper than end-to-end |

### 7.3 Score Maximization (if time permits)

- JPEG compression on test dataset tiles (quality 85, 4:2:2 subsampling [Pillow value 1], Lanczos interpolation, tiles downscaled to 518×518) — expected +0.035-0.04 F1
- Ensemble 3 student seeds
- Teacher-student weighted average
- Test-time augmentation (flip)

---

## Compute Budget

| Phase | GPU-hrs | Days (at 8 hrs/day) |
|-------|---------|---------------------|
| 1a: Teacher pipeline on TEST SET only | 2 | <1 |
| 1b: Teacher pipeline on LUCAS (after Phase 2 gate) | 18 | 2-3 |
| 2: Teacher benchmark + sweeps | 4 | 1 |
| 3: Pseudo-label generation | 8 | 1 |
| 4: Student ablations (10 runs) | 49 | 6-7 |
| 5: Student inference | 2 | <1 |
| 6: Self-training (2 rounds) | 16 | 2 |
| 7: Hyperparams + analysis | 16 | 2 |
| **Total** | **~115** | **~15 days GPU** |

Fits comfortably in 1-2 months with days between experiments for analysis.

---

## Paper Narrative

**Title**: "CrossRegionTransformer: Self-Supervised Multi-Label Plant Identification via Knowledge Distillation from Unlabeled Vegetation Plots"

**Contributions**:
1. CrossRegionTransformer: learned tile aggregation that outperforms heuristic Noisy-OR/majority vote
2. Multi-view consistency regularization adapted from semi-supervised learning to embedding-space KD
3. Quality-filtered pseudo-label pipeline: aug consistency + tile confidence + species percentile
4. Comparison of negative sampling vs soft distillation for positive-only multi-label KD
5. Self-training loop that iteratively improves pseudo-labels
6. All on frozen embeddings — 100x more efficient than end-to-end

**Ablation table proves**: Each component adds measurable F1 improvement.

---

## Expected Outcome

### Performance Trend

```
Teacher baseline (heuristic Noisy-OR)
    ↓
Student R0 (learned aggregation) — improves over teacher
    ↓
Student R0 + consistency + filtering — improves further
    ↓
Student R1 (self-training) — improves further
```

**Key result**: Student model outperforms heuristic teacher aggregation by 10%+ F1.

### Confidence Assessment

- ~60-70% chance student > teacher (with all components)
- ~30-40% chance student ≈ teacher
- Unlikely student is worse if pseudo-labels are filtered well

**Biggest risk**: Pseudo-label precision too low → student learns noise. The species percentile filtering (Phase 3.2) and augmentation consistency (Phase 1.2) are the primary safeguards.

**Early signals**:
- Teacher F1 >= 0.24 after Phase 2 → pipeline is healthy
- Avg 4-10 pseudo-labels per LUCAS image → filtering is calibrated

---

## Early Warning Signals

| Signal | Check After | Healthy Range | Action If Bad |
|--------|-------------|---------------|---------------|
| Teacher F1 | Phase 2 | >= 0.24 | Debug preprocessing |
| Pseudo-labels/image | Phase 3 | 2-5 species avg | Adjust percentile |
| Species coverage | Phase 3 | 2000+ species | Loosen filtering |
| Student > Teacher | Phase 5 | +0.01 minimum | Check loss function, try other variant |
| Self-training gain | Phase 6 | +0.005 per round | Stop if no gain |

---

## Critical Files to Create/Modify

| File | Action |
|------|--------|
| `teacher_classify_tiles.py` | CREATE: standalone PyTorch, aug consistency, 3-view embedding extraction (orig/flip/rot90), tile filtering |
| `teacher_aggregate.py` | CREATE: Noisy-OR with temperature, geo mask, submission CSV |
| `generate_pseudolabels.py` | CREATE: percentile filtering, output dataset with 3-view embeddings |
| `train_student.py` | CREATE/MODIFY: BCE neg-sampling + KL distillation + consistency loss, variable tile handling |
| `student_inference.py` | CREATE: test-time student pipeline |
| `self_train.py` | CREATE: student → pseudo-labels → retrain loop |
| `Cross_Region_Transformer_v3.py` | MODIFY: attention mask for variable tile count, embedding dropout, attention viz hook |
