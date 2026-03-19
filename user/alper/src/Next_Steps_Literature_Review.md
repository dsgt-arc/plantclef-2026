# Next Steps: Literature Review & Improvement Roadmap

**Date**: 2026-03-15
**Current Best F1**: 0.38466 (Exp-7, 4x4+5x5+6x6, thr=0.075)
**Baseline**: 0.28093 (Exp-3) | **Total improvement so far**: +0.10373 (+36.9%)

---

## Sources Reviewed

| # | Paper | Place | Private F1 | Key Innovation |
|---|---|---|---|---|
| 1 | Preprocessing Is All You Need (Espitalier) | 1st | 0.36479 | JPEG regularization, multi-scale tiling, VaMIS |
| 2 | GT Tile-Based ViT with Visual-Cluster Priors | 2nd | 0.34489 | (Could not read .docx — our team DS@GT) |
| 3 | Metadata-Enhanced Multi-Head ViTs (Hamburg) | 3rd | 0.33655 | Taxonomic fusion (species+genus+family), Hydra |
| 4 | Optimizing ViT with Ecological Context (Semenova) | 4th | 0.33447 | SAM noise masking, GBIF seasonal, EIVE niche |
| 5 | Zero-Shot Segmentation via Prototypes (ADAM-UEFS) | 5th | 0.33331 | Attention-guided segmentation, grid assembly |
| 6 | Tiles-wise Inference with ViTs (VerbaNexAI) | 13th | 0.28324 | 4x2 patch division baseline |
| 7 | Self-Distillation Enables Continual Learning (Shenfeld et al.) | arxiv | — | On-policy self-distillation, EMA teacher |
| 8 | FathomNet CVPR 2025 Winner (dhlee-work) | 1st FathomNet | — | MCEAM multi-context attention, hierarchical aux classification |

---

## Cross-Paper Theme Analysis

Three techniques appear independently in multiple winning solutions:

| Theme | PlantCLEF 1st | PlantCLEF 3rd | PlantCLEF 4th | FathomNet Winner |
|---|---|---|---|---|
| Multi-scale tiling/context | Scales 1-6 | Scales 4,5 | 518px + 732px | ROI + 3x/5x/full |
| Taxonomic hierarchy | -- | Species+Genus+Family | -- | Order+Family+Species |
| Artifact/noise handling | JPEG regularization | Border cropping | SAM+GroundingDINO | -- |
| Ecological priors | -- | -- | GBIF+EIVE | Environmental attention |

---

## Techniques: What We've Done vs. What's Available

| Technique | Paper Source | Our Status | Est. Impact |
|---|---|---|---|
| Multi-scale tiling (4-6) | 1st | **Done** (Exp-7) | -- |
| JPEG85 compression | 1st | **Done** (Exp-4) | -- |
| Lanczos interpolation | 1st | **Done** (Exp-4) | -- |
| max_prob aggregation | 1st | **Done** (Exp-4) | -- |
| Temporal fusion (location priors) | ours | **Done** (Exp-4) | -- |
| Threshold/alpha/T re-tuning for multi-scale | -- | **Done** (Exp-7 sweeps) | -- |
| **Border margin cropping** | 3rd, 4th | Not tried | +0.3-1.0% |
| **Per-transect threshold optimization** | 1st | Not tried | +0.3-0.7% |
| **Multi-head taxonomic fusion** | 3rd, FathomNet | Not tried | +1.0-2.0% |
| **SAM/GroundingDINO noise masking** | 4th | Not tried | +1.0-2.5% |
| **GBIF seasonal filtering** | 4th | Not tried | +0.1-0.5% |
| **EIVE niche similarity filtering** | 4th | Not tried | +0.5-1.0% |
| **Cross-year plot aggregation** | 4th | Partial (temporal fusion) | +0.1-0.5% |
| **Overlapping/sliding window tiles** | 4th | Not tried | +0.5-1.5% |
| **Multi-context environmental attention** | FathomNet | Not tried | +0.5-2.0% |
| **Self-distillation student training** | arxiv paper | Not tried | +1.0-3.0% |
| 13-crop TTA per tile | 4th | Tried (hurts with max_prob) | ~0 |
| VaMIS high-res inference | 1st | Not feasible (GPU) | Unknown |

---

## Ranked Improvement Plan

### Phase A: Quick Wins (CPU-only, hours of work)

#### A1. Border Margin Cropping
- **Source**: 3rd place (5-10% crop), 4th place (150px margin)
- **What**: Skip 100-200px from all image edges before tiling. Removes wooden frames, rulers, color cards that generate false positives.
- **Implementation**: Modify `extract_multiscale_embeddings.py` to crop before tiling. Re-extract embeddings for all scales.
- **GPU cost**: ~9 hours (re-extract 5x5 + 6x6 scales)
- **Expected**: +0.3-1.0%

#### A2. Per-Transect Threshold Optimization
- **Source**: 1st place (gained +0.007 private F1)
- **What**: Parse transect IDs from test image filenames. Optimize detection threshold per transect using bisection search to target ~3-4 species/image average. Some transects are harder and need different thresholds.
- **Implementation**: Group images by transect name prefix. For each transect, sweep threshold to match expected species count. Use a multiplicative parameter to adjust globally.
- **GPU cost**: 0 (CPU-only, uses existing predictions)
- **Expected**: +0.3-0.7%

### Phase B: Medium Effort, High Impact (days of work)

#### B1. Multi-Head Taxonomic Classification
- **Source**: 3rd place PlantCLEF (biggest contribution), FathomNet winner (hierarchical aux classification)
- **What**: Train additional genus (1,446 classes) and family (181 classes) classification heads on the frozen DINOv2 backbone. At inference, compute `final_score = P(species) x P(genus) x P(family)`, discarding species whose genus/family predictions are inconsistent.
- **Why it works**: The model may confuse visually similar species but correctly identify the genus. Multiplying probabilities along the taxonomy suppresses cross-genus false positives.
- **Implementation**:
  1. Map each species_id to its genus_id and family_id from training metadata
  2. Train two new linear heads (genus, family) on the frozen DINOv2 CLS embeddings from training data
  3. At inference, extract genus/family logits per tile alongside species logits
  4. Multiply: `score(s) = P_species(s) x P_genus(genus_of_s) x P_family(family_of_s)`
  5. Aggregate with max_prob as before
- **GPU cost**: ~3-6 hours to train heads on 1.4M training images (frozen backbone, only train linear layers)
- **Expected**: +1.0-2.0%
- **Confidence**: HIGH — independently validated by 3rd place PlantCLEF AND FathomNet winner

#### B2. Artifact Masking with SAM + GroundingDINO
- **Source**: 4th place PlantCLEF (+2.61 pp public score)
- **What**: Use GroundingDINO (open-vocabulary object detector) with text prompts to detect non-plant objects. Use SAM to produce pixel-accurate masks. Soft-penalty tile confidence by mask coverage fraction.
- **Prompts**: "stone, shell, ruler, plastic, metal, hand, ice, snow, measure, wood, board, paper"
- **Formula**: `p' = p * (1 - alpha * mask_fraction)`, alpha=0.35
- **Implementation**:
  1. Run GroundingDINO on each test image (2,105 images)
  2. Run SAM on detected bounding boxes to get pixel masks
  3. For each tile, compute fraction of masked pixels
  4. Scale tile predictions by `(1 - 0.35 * mask_fraction)` before aggregation
- **GPU cost**: ~2-4 hours for GroundingDINO + SAM inference on 2,105 images
- **Expected**: +1.0-2.5%
- **Note**: Requires installing GroundingDINO and SAM models. Could simplify to rectangular mask heuristic (4th place: only -0.3pp vs full SAM).

#### B3. Overlapping/Sliding Window Tiles
- **Source**: 4th place PlantCLEF (518px windows, 172px stride, ~144 tiles/image)
- **What**: Instead of non-overlapping NxN grid, use sliding windows with ~1/3 overlap. Plants on tile boundaries get captured whole by overlapping tiles.
- **Why it might help**: With max_prob, more tiles = more chances to capture a species optimally framed. The 4th place got ~144 tiles/image with dense overlap vs. our 77 tiles with no overlap.
- **Caution**: 3rd place found overlap didn't help them. May be redundant with multi-scale approach.
- **Implementation**: Modify extraction to use sliding window with configurable stride per scale.
- **GPU cost**: ~2-3x current extraction time depending on stride
- **Expected**: +0.5-1.5% (uncertain — could be redundant with multi-scale)

### Phase C: Higher Effort, Strategic (week+ of work)

#### C1. EIVE Niche Similarity Filtering
- **Source**: 4th place PlantCLEF (+0.97 pp from niche drop + re-add combined)
- **What**: Use Ecological Indicator Values for Europe (EIVE) Ellenberg indicators to check ecological compatibility of predicted species within each plot.
- **Algorithm**:
  1. For each image, take the top-9 predicted species + 9 reserve candidates (top-18 total)
  2. For each species pair, compute Niche Similarity Index S from 5 indicators (moisture, nitrogen, soil reaction, light, temperature)
  3. S = (center_overlap + bhattacharyya_coefficient) / 2
  4. Remove species with mean S < 0.015 to all others (ecologically incompatible)
  5. Add reserve species with mean S >= 0.75 to remaining set (ecologically compatible)
- **Data needed**: EIVE 1.0 dataset (freely available, contains mean+std for each species on each indicator)
- **Implementation**: Post-processing step on existing predictions. No GPU needed.
- **Expected**: +0.5-1.0%

#### C2. GBIF Seasonal Filtering
- **Source**: 4th place PlantCLEF (+0.15 pp)
- **What**: Use GBIF Europe occurrence records to check if a predicted species is seasonally plausible for the image's acquisition month.
- **Algorithm**:
  1. Parse acquisition date from test image filenames (YYYYMMDD format)
  2. For each predicted species, check GBIF monthly occurrence count in Europe (2020-2024)
  3. Remove species where count falls between 1-10 in the observation month (too rare to be plausible; count=0 preserved to avoid GBIF gaps)
- **Data needed**: GBIF Europe occurrence statistics by month per species (downloadable)
- **Implementation**: Post-processing on existing predictions. No GPU needed.
- **Expected**: +0.1-0.5%

#### C3. Multi-Context Environmental Attention (Novel)
- **Source**: FathomNet CVPR 2025 winner (MCEAM module)
- **What**: Instead of classifying tiles independently, let each tile attend to its spatial neighbors. A tile at position (i,j) receives context from surrounding tiles via cross-attention. This captures co-occurrence patterns (e.g., species A and B often appear together).
- **Architecture**: Small transformer that takes per-tile CLS embeddings as input, applies positional encoding based on tile grid position, and outputs context-enriched embeddings for classification.
- **Implementation**:
  1. Extract per-tile CLS embeddings (already have these)
  2. For each image, arrange tile embeddings in spatial grid with positional encoding
  3. Train small attention module: input = tile embeddings, output = refined embeddings
  4. Use refined embeddings for classification
  5. Training data: use temporal fusion priors or pseudo-labels as supervision
- **GPU cost**: Moderate (small model, but needs training loop)
- **Expected**: +0.5-2.0% (novel, uncertain)

#### C4. Self-Distillation for Student Recovery
- **Source**: arxiv 2601.19897 (Shenfeld et al.)
- **What**: Your Exp-3 student failed due to mode collapse from off-policy KL distillation on ~7742 near-zero targets. Self-distillation approach could fix this:
  1. Use teacher model conditioned on pseudo-labels as guide (demonstration-conditioned)
  2. Student generates predictions on-policy
  3. Distill from teacher's conditioned output (not raw pseudo-labels)
  4. EMA teacher update: phi <- alpha*theta + (1-alpha)*phi
- **Why it might fix Exp-3**: The mode collapse happened because 7742/7806 targets were near-zero, overwhelming the ~64 positive signals. Self-distillation generates training signal from the model's own distribution, which is already concentrated on plausible species, avoiding the sparse target problem.
- **Key insight from paper**: On-policy learning essential; offline distillation from same teacher underperforms.
- **Implementation**: Major rework of Exp-3 phase 4 training loop.
- **Expected**: +1.0-3.0% (high variance, research-grade)

#### C5. Image Similarity Graph (Exp-5, previously planned)
- **Source**: Your own Exp-5 plan (never implemented)
- **What**: Build k-NN graph of test images using DINOv2 embeddings. Propagate predictions across similar images iteratively.
- **Algorithm**:
  1. Extract image-level embeddings (mean-pool tile CLS tokens)
  2. Build k-NN graph (k~10, cosine similarity)
  3. Row-normalize adjacency matrix W
  4. Propagate: `P_new = (1-alpha)*P + alpha*W*P` for T iterations
  5. Re-threshold
- **Expected**: +0.5-2.0%

---

## Priority Matrix

| Recommendation | Impact | Effort | Risk | Priority |
|---|---|---|---|---|
| A1. Border margin cropping | Medium | Low | Low | **Do first** |
| A2. Per-transect threshold | Medium | Low | Low | **Do first** |
| B1. Multi-head taxonomic fusion | High | Medium | Low | **Do second** |
| B2. SAM artifact masking | High | Medium | Medium | **Do second** |
| B3. Overlapping tiles | Medium | Medium | Medium | Consider |
| C1. EIVE niche filtering | Medium | Medium | Low | **Do third** |
| C2. GBIF seasonal filtering | Low-Med | Low | Low | **Do third** |
| C3. Multi-context attention | Med-High | High | High | Research |
| C4. Self-distillation student | High | Very High | High | Research |
| C5. Image similarity graph | Med-High | Medium | Medium | Consider |

---

## Estimated Cumulative Impact

```
Current best:                              0.38466
+ Phase A (border crop + per-transect):   ~0.390 - 0.395
+ Phase B (taxonomy + SAM masking):       ~0.405 - 0.425
+ Phase C (ecological + novel):           ~0.415 - 0.445

Target: 0.42+ would significantly exceed all 2025 competitors
```

---

## Key References

- 1st Place repo: https://github.com/v-espitalier/PlantCLEF2025
- 3rd Place repo: https://github.com/geranium12/plant-clef-2025/tree/v1.0.0
- 5th Place repo: https://github.com/ADAM-UEFS/PlantCLEF2025
- FathomNet winner: https://github.com/dhlee-work/fathomnet-cvpr2025-ssl
- Self-Distillation paper: https://arxiv.org/abs/2601.19897
- EIVE dataset: Dengler et al. (2023), Vegetation Classification and Survey 4, 7-29. doi:10.3897/VCS.98324
- GBIF Europe occurrences: https://www.gbif.org/occurrence/search
