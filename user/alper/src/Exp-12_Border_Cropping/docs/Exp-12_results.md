# Exp-12: Border Cropping — Results Report

**Date:** 2026-04-12
**Status:** Concluded (negative result)
**Baseline F1:** 0.35837 (4x4 single-scale, T=1.5, alpha=0.6, thr=0.040)

## Objective

Test whether removing border artifacts (wooden frames, measuring tape, rulers, color cards) from test quadrat images improves F1 score by eliminating false-positive species predictions from non-plant edge content.

## Prior Art

| Source | Technique | Reported Gain |
|--------|-----------|---------------|
| 3rd place PlantCLEF 2025 | 5-10% crop from edges | +0.3-1.0% |
| 4th place PlantCLEF 2025 | 150px fixed margin | +0.3-1.0% |

## Steps Taken

### Step 1: EDA — Visual Inspection of Border Artifacts

Visually inspected 12 test images across all 5 data sources (CBN, LISAH, GUARDEN, RNNB, OPTMix) to catalog border artifact types and estimate their thickness.

**Findings:**

| Source | Count | % of test set | Border Type | Thickness |
|--------|-------|---------------|-------------|-----------|
| CBN-PdlC/Pla/Pyr | 1,446 | 69% | Measuring tape, thin rulers | ~20-50px |
| CBN-can | 30 | 1% | Thick wooden frame + tape | ~80-120px |
| LISAH | 208 | 10% | Thin metal/plastic markers | ~15-30px |
| GUARDEN | 201 | 10% | Thick wooden frame + metal hardware | ~100-130px |
| RNNB | 141 | 7% | Thick wooden frame | ~80-100px |
| OPTMix | 78 | 4% | Thin frame edges | ~20-40px |

- 78% of images have thin borders (3% crop sufficient)
- 17% have thick wooden frames (5-7% crop needed)
- Image sizes vary: 1416px to 3492px

Generated visualizations: crop overlay grid, corner zooms on worst-case frames, before/after comparison, area retention chart, border severity pie chart (see `docs/*.png`).

### Step 2: Experiment Design

Designed a quick 4x4 single-scale test (16 tiles/image) to validate border cropping before committing to full 77-tile multi-scale re-extraction. Three crop levels tested: 3%, 5%, 7% (uniform pixels removed from all four sides).

| Crop % | Pixels removed (3024px) | Pixels removed (1864px) | Area retained |
|--------|------------------------|------------------------|---------------|
| 3% | 90px each side | 55px each side | 88.4% |
| 5% | 151px each side | 93px each side | 81.0% |
| 7% | 211px each side | 130px each side | 74.0% |

### Step 3: First Run — Crop Without Resize

Cropped all 2,105 test images at 3%, 5%, 7%. Extracted 4x4 DINOv2 embeddings (JPEG85, ViT-B/14). Aggregated with max_prob (T=1.5), temporal fusion (alpha=0.6), and swept thresholds 0.04, 0.07, 0.10.

**Results (Kaggle public leaderboard F1):**

| Condition | thr=0.040 | thr=0.070 | thr=0.100 | Best F1 | vs Baseline |
|-----------|-----------|-----------|-----------|---------|-------------|
| **Baseline (no crop)** | **0.35837** | 0.34632 | 0.33285 | **0.35837** | -- |
| Crop 3% | 0.32596 | 0.30403 | 0.29224 | 0.32596 | **-0.032** |
| Crop 5% | 0.31565 | 0.30838 | 0.27987 | 0.31565 | **-0.043** |
| Crop 7% | 0.32096 | 0.32024 | 0.30230 | 0.32096 | **-0.037** |

All crop levels substantially worse than baseline. F1 dropped by 3.2-4.3 points.

### Step 4: Diagnosing the Cause — Crop + Resize Back Control

Suspected the F1 drop could be partially caused by the dimension change rather than the content removal. When images are cropped, their dimensions shrink, which changes the resize ratio during tiling:

- Baseline 3024px image: downscaled to 2072px (ratio 0.685)
- Cropped 5% (2722px): downscaled to 2072px (ratio 0.761, less downscaling)
- GUARDEN 1864px baseline: upscaled to 2072px (ratio 1.112)
- GUARDEN 1864px cropped 5% (1678px): upscaled to 2072px (ratio 1.235, more upscaling)

Added a `--resize-back` flag to the crop script: crop the borders, then Lanczos resize back to the original image dimensions. This isolates the content-removal effect from the dimension-change effect.

**Results with resize-back (Kaggle public leaderboard F1):**

| Condition | thr=0.040 | thr=0.070 | thr=0.100 | Best F1 | vs Baseline |
|-----------|-----------|-----------|-----------|---------|-------------|
| **Baseline (no crop)** | **0.35837** | 0.34632 | 0.33285 | **0.35837** | -- |
| Crop 3% resized | 0.33161 | 0.33280 | 0.31801 | 0.33280 | **-0.026** |
| Crop 5% resized | 0.33074 | 0.30605 | 0.28788 | 0.33074 | **-0.028** |
| Crop 7% resized | 0.32404 | 0.32725 | 0.30200 | 0.32725 | **-0.031** |

### Step 5: Decomposing the Effects

Comparing crop-only vs crop+resize at each level quantifies how much of the F1 drop is from dimensions vs content:

| Crop % | Total F1 drop | Dimension effect | Content effect | Dimension share |
|--------|--------------|-----------------|---------------|-----------------|
| 3% | -0.032 | -0.007 | -0.026 | 20% |
| 5% | -0.043 | -0.015 | -0.028 | 35% |
| 7% | -0.037 | -0.006 | -0.031 | 17% |

The dimension change explains 17-35% of the F1 drop. The remaining 65-83% is from the content removal itself.

## Full Results Table

| Condition | thr=0.040 | thr=0.070 | thr=0.100 | Best F1 | vs Baseline |
|-----------|-----------|-----------|-----------|---------|-------------|
| **Baseline** | **0.35837** | **0.34632** | **0.33285** | **0.35837** | -- |
| Crop 3% | 0.32596 | 0.30403 | 0.29224 | 0.32596 | -0.032 |
| Crop 3% resized | 0.33161 | 0.33280 | 0.31801 | 0.33280 | -0.026 |
| Crop 5% | 0.31565 | 0.30838 | 0.27987 | 0.31565 | -0.043 |
| Crop 5% resized | 0.33074 | 0.30605 | 0.28788 | 0.33074 | -0.028 |
| Crop 7% | 0.32096 | 0.32024 | 0.30230 | 0.32096 | -0.037 |
| Crop 7% resized | 0.32404 | 0.32725 | 0.30200 | 0.32725 | -0.031 |

## Conclusions

1. **Border cropping hurts performance.** Every crop level at every threshold performed worse than the uncropped baseline, both with and without resize-back. The best cropped result (0.33280) is still 2.6 points below baseline (0.35837).

2. **The dimension change accounts for ~25% of the damage.** Resizing cropped images back to original dimensions recovered roughly a quarter of the F1 loss, confirming the pipeline is sensitive to input resolution. However, the majority of the damage comes from removing content.

3. **Border regions contain useful plant signal.** Vegetation frequently overhangs the quadrat frames. Removing these edge pixels loses real species observations that the model relies on. The border artifacts (wood, tape, rulers) cause less harm to predictions than the plant content that gets removed alongside them.

4. **The existing pipeline is already robust to border artifacts.** The combination of max_prob aggregation (one good interior tile overrides noisy edge tiles), temporal fusion (95% location prior weight in the best config), aug consistency filtering, and high thresholds effectively suppresses false positives from border content without needing to crop.

5. **The 3rd/4th place gains likely came from simpler pipelines.** Those solutions likely used single-scale tiling, lower thresholds, or weaker temporal priors where border noise had more impact on final predictions.

## Recommendation

Do not pursue border cropping further. For artifact mitigation, consider SAM-based soft tile weighting (4th place: `p' = p * (1 - alpha * mask_fraction)`) which penalizes artifact-heavy tiles without removing plant content. Adrian's SAM3 masks (1.3M training images processed) provide existing infrastructure for this approach.

Higher-impact next steps:
- **Exp-10/11: ConvNeXt + ViT ensemble** (architecture diversity, already planned)
- **SAM artifact masking** (surgical tile confidence weighting, +2.61pp reported by 4th place)
- **Single-visit image improvements** (509/2105 images lack temporal prior)

## Files

| File | Purpose |
|------|---------|
| `crop_test_images.py` | Batch-crop images with optional `--resize-back` |
| `slurm_crop_extract.sbatch` | GPU job: crop + extract (dimension-changing) |
| `slurm_crop_extract_resized.sbatch` | GPU job: crop + resize back + extract (control) |
| `slurm_aggregate_compare.sbatch` | CPU job: aggregate all conditions + generate CSVs |
| `compare_submissions.py` | Local prediction distribution analysis |
| `docs/plan.md` | Original experiment plan |
| `docs/results.md` | This report |
| `docs/*.png` | EDA visualizations |

## Compute Cost

- Extraction: 6 GPU jobs (RTX 6000, ~35 min each, QoS=embers)
- Aggregation: 1 CPU job (~10 min)
- Kaggle submissions: 21 CSVs submitted
- Total wall clock: ~1.5 hours
