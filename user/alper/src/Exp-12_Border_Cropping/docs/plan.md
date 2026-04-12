# Exp-12: Border Cropping — Experiment Plan

## Hypothesis

Test images contain non-plant border artifacts (wooden frames, measuring tape, rulers, color cards) that may cause false-positive species predictions when edge tiles are processed by DINOv2. Cropping a uniform border margin before tiling may improve F1 by removing these artifacts.

**Expected impact:** +0.3-1.0% F1 (based on 3rd/4th place PlantCLEF 2025 solutions).

**Skepticism:** The current pipeline already has defenses against border noise (max_prob aggregation, aug consistency filter, temporal fusion at 95% location prior weight, high threshold at 0.100). The gain may be negligible. This experiment is a quick validation to confirm or reject.

## Prior Art

| Source | Technique | Reported Gain |
|--------|-----------|---------------|
| 3rd place PlantCLEF 2025 | 5-10% crop from edges | +0.3-1.0% |
| 4th place PlantCLEF 2025 | 150px fixed margin | +0.3-1.0% |
| 4th place PlantCLEF 2025 | SAM + GroundingDINO artifact masking | +2.61pp |

Source: `user/alper/src/Next_Steps_Literature_Review.md`, Phase A Quick Wins.

## EDA Findings

Visual inspection of 12 images across all 5 data sources revealed:

| Source | Count | % of test set | Border Type | Thickness |
|--------|-------|---------------|-------------|-----------|
| CBN-PdlC/Pla/Pyr | 1,446 | 69% | Measuring tape, thin rulers | ~20-50px |
| CBN-can | 30 | 1% | Thick wooden frame + tape | ~80-120px |
| LISAH | 208 | 10% | Thin metal/plastic markers | ~15-30px |
| GUARDEN | 201 | 10% | Thick wooden frame + metal hardware | ~100-130px |
| RNNB | 141 | 7% | Thick wooden frame | ~80-100px |
| OPTMix | 78 | 4% | Thin frame edges | ~20-40px |
| 2024-CEV3 | 1 | 0% | Thin wooden sticks | ~10px |

**Summary:** 78% of images have thin borders (3% crop sufficient), 17% have thick wooden frames (5-7% needed), 4% have minimal borders.

Area retention at each crop level:

| Crop % | Pixels removed (3024px img) | Pixels removed (1864px img) | Area retained |
|--------|----------------------------|----------------------------|---------------|
| 3% | 90px each side | 55px each side | 88.4% |
| 5% | 151px each side | 93px each side | 81.0% |
| 7% | 211px each side | 130px each side | 74.0% |

See `docs/*.png` for visualizations (crop overlays, corner zooms, before/after, severity distribution).

## Experiment Design

**Quick validation using 4x4 single-scale only** (16 tiles/image instead of 77). This isolates the cropping effect without the cost of full multi-scale extraction.

**Comparison:** 4 conditions at 3 threshold levels = 12 submission CSVs.

| Condition | Crop | Tiles | Source |
|-----------|------|-------|--------|
| Baseline | 0% | 16 | Existing Exp-4 4x4 parquets |
| Crop 3% | 3% per side | 16 | New extraction |
| Crop 5% | 5% per side | 16 | New extraction |
| Crop 7% | 7% per side | 16 | New extraction |

**Parameters held constant** (matching Exp-7 4x4 baseline for fair comparison):
- Temperature: 1.5
- Aggregation: max_prob
- Temporal fusion: prior_blend, alpha=0.6
- Thresholds swept: 0.04, 0.07, 0.10
- TopK: 10
- Geo mask: sw_europe_mask_bool.npy

## Scripts

### 1. `crop_test_images.py`
Batch-crops all test images at specified percentages. Removes the same number of pixels from all four sides (uniform crop). Saves as JPEG quality=95 to avoid introducing additional compression artifacts before the pipeline's own JPEG85 step.

- **Input:** Original test images directory
- **Output:** One subdirectory per crop percentage (e.g., `crop_3pct/`, `crop_5pct/`, `crop_7pct/`)
- **Runs on:** PACE (called by SLURM script), but also works locally

### 2. `slurm_crop_extract.sbatch`
SLURM array job (3 tasks) that crops images and extracts 4x4 DINOv2 embeddings.

- **Array mapping:** Task 0=3%, Task 1=5%, Task 2=7%
- **Resources:** 1x RTX 6000 GPU, 6 CPUs, 64GB RAM, QoS=embers
- **Time:** ~35 min per task
- **Step 1:** Runs `crop_test_images.py` to create cropped images on PACE storage
- **Step 2:** Runs `Exp-7/extract_multiscale_embeddings.py --scale 4` on the cropped images
- **Output:** `exp12_border_crop/scale_4x4_crop{3,5,7}pct.parquet`
- **Based on:** `Exp-7/slurm_extract_multiscale.sbatch` (same PACE paths, venv, model)

### 3. `slurm_aggregate_compare.sbatch`
SLURM CPU job that aggregates tiles, applies temporal fusion, and generates submission CSVs for all 4 conditions (baseline + 3 crops).

- **Resources:** 4 CPUs, 48GB RAM, QoS=embers, cpu-medium partition
- **Time:** ~10 min total
- **Runs:** `Exp-7/multiscale_aggregate.py --schemes A` for each condition
- **Output:** 12 submission CSVs in `exp12_border_crop/{baseline,crop_3pct,crop_5pct,crop_7pct}/`
- **Based on:** `Exp-7/slurm_phase1b_sweep.sbatch` (same aggregation pipeline)

### 4. `compare_submissions.py`
Local analysis script to compare prediction distributions across crop levels before submitting to Kaggle. Computes Jaccard similarity, counts changed predictions, identifies which species were added/removed by cropping.

- **Input:** Submission CSVs from Step 3
- **Output:** Summary table with change metrics
- **Interpretation:** High Jaccard (>0.95) means cropping barely changes predictions and is unlikely to move F1

## Dependencies on Prior Experiments

This experiment reuses code from Exp-7 and Exp-8 without modification:

| Script | Source | Purpose |
|--------|--------|---------|
| `extract_multiscale_embeddings.py` | Exp-7 | DINOv2 embedding extraction (center_crop_square, tile, JPEG85, inference) |
| `multiscale_aggregate.py` | Exp-7 | Tile aggregation + temporal fusion + CSV generation |
| `aggregate_tiles.py` | Exp-7 | Core aggregation engine (AggregationConfig, max_prob) |
| `temporal_fusion.py` | Exp-7 | Location-based prior blending |
| `generate_submission.py` | Exp-7 | Parquet to Kaggle CSV conversion |
| `utils/aggregator.py` | Exp-7 | MaxProbabilityAggregator |
| `utils/io_bytes.py` | Exp-7 | float16 byte serialization |

The SLURM scripts set `PYTHONPATH` to include the Exp-7 source directory so these imports resolve at runtime.

**Baseline 4x4 parquets** are reused from Exp-4 JPEG85 extraction (path: `phase1_tiles_test_jpeg85/`).

## Execution Steps

```
1. sbatch slurm_crop_extract.sbatch     # GPU array job (3 parallel tasks, ~35 min)
2. sbatch slurm_aggregate_compare.sbatch # CPU job (after step 1, ~10 min)
3. Download CSVs from PACE              # 12 files
4. python compare_submissions.py ...    # Local quick analysis (optional)
5. Submit CSVs to Kaggle                # Get F1 scores
```

## Success Criteria

- **Positive result:** Any crop level beats baseline at the same threshold by >= 0.003 F1
- **Negative result:** All crop levels within 0.002 of baseline (confirms pipeline is already robust)
- **If negative:** Skip full multi-scale border cropping, move to ConvNeXt ensemble (Exp-10/11) or SAM-based artifact masking instead

## PACE Storage Paths

```
$PLANTCLEF_ROOT = /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0

Inputs:
  Images:    $PLANTCLEF_ROOT/data/raw/PlanCLEF_test_data/images/
  Model:     $PLANTCLEF_ROOT/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth
  Baseline:  $PLANTCLEF_ROOT/aerten3/parquet/plantclef_2026_test/phase1_tiles_test_jpeg85/
  Geo mask:  $PLANTCLEF_ROOT/aerten3/species_masking/sw_europe_mask_bool.npy
  Species:   $PLANTCLEF_ROOT/data/raw/class_id_to_species_mapping/class_to_species_id.json

Intermediates:
  Cropped images: $PLANTCLEF_ROOT/aerten3/data/exp12_cropped_images/crop_{3,5,7}pct/
  Embeddings:     $PLANTCLEF_ROOT/aerten3/parquet/plantclef_2026_test/exp12_border_crop/

Outputs:
  Submissions:    $PLANTCLEF_ROOT/aerten3/submissions/exp12_border_crop/{baseline,crop_{3,5,7}pct}/
```
