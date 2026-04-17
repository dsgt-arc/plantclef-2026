# Exp-12 Chat Context Summary

**Date:** 2026-04-12
**Purpose:** Resume context for future Claude Code sessions about this experiment.

## What Was Done in This Session

### Phase 1: EDA and Border Analysis
- Explored all 2,105 test images across 5 data sources (CBN, LISAH, GUARDEN, RNNB, OPTMix)
- Visually inspected 12+ images using Claude's image reading capability to catalog border artifact types
- Ran Python scripts locally (using `/c/Users/alper/anaconda3/python`) to compute image size statistics, estimate border widths via edge variance analysis, and generate visualizations
- Created 6 visualization PNGs in `docs/`: crop overlay grid, corner zooms, before/after 5%, area retention chart, border severity pie chart
- Found: 78% thin borders (tape), 17% thick wooden frames, 4% minimal. Image sizes range 1416-3492px.

### Phase 2: Experiment Design
- Decided on quick 4x4 single-scale test (not full 77-tile multi-scale) to validate before committing compute
- Chose 3%, 5%, 7% crop levels based on visual inspection
- Created 4 scripts reusing Exp-7/Exp-8 pipeline code (no modifications to upstream scripts)
- Initially discussed whether border cropping would even help — skepticism was raised because max_prob aggregation, temporal fusion, and high thresholds already handle border noise

### Phase 3: First Run (Crop Without Resize)
- Ran on PACE: `slurm_crop_extract.sbatch` (3 GPU array tasks) + `slurm_aggregate_compare.sbatch` (CPU)
- **Bug fix during run:** `slurm_aggregate_compare.sbatch` was missing `--model-path` argument — user caught this from the SLURM error output, fixed by adding `MODEL_PATH` variable and passing it to all `multiscale_aggregate.py` calls
- **QoS change:** User requested GPU jobs use `embers` instead of `inferno`, matching Exp-7 setup. Changed in `slurm_crop_extract.sbatch`.
- **Result:** All crop levels hurt F1 by 3.2-4.3 points vs baseline (0.358)

### Phase 4: Dimension Change Hypothesis
- User suspected the F1 drop was from image dimension changes, not content removal
- Key insight: cropped images are smaller, so the resize-to-2072px step in the pipeline produces different interpolation. Especially bad for small GUARDEN images (~1864px) which get upscaled even more after cropping.
- Added `--resize-back` flag to `crop_test_images.py`: crop borders then Lanczos resize back to original dimensions
- Created `slurm_crop_extract_resized.sbatch` for the control experiment
- Updated `slurm_aggregate_compare.sbatch` to process both crop-only and crop+resize variants (7 total conditions)

### Phase 5: Control Experiment Results
- Resize-back recovered ~25% of the F1 loss (dimension effect)
- Remaining ~75% is genuine content loss — borders contain overhanging vegetation
- Conclusion: border cropping is harmful, experiment closed as negative result

## Key Decisions Made
1. **5% crop as starting point** — matched 4th place PlantCLEF 2025 (~150px on ~3000px images)
2. **4x4 single-scale only** — quick validation, not full 77-tile pipeline
3. **alpha=0.6 for comparison** — Exp-7 4x4 baseline params, not the later-optimized alpha=0.05 (which is for multi-scale)
4. **Thresholds 0.04, 0.07, 0.10** — covering the Exp-7 baseline range
5. **JPEG quality=95 for cropped saves** — minimize extra compression artifacts before pipeline's JPEG85 step

## Files in This Experiment

```
Exp-12_Border_Cropping/
├── crop_test_images.py              # Batch-crop with optional --resize-back
├── slurm_crop_extract.sbatch        # GPU: crop + extract (dimension-changing)
├── slurm_crop_extract_resized.sbatch # GPU: crop + resize back + extract (control)
├── slurm_aggregate_compare.sbatch   # CPU: aggregate all 7 conditions
├── compare_submissions.py           # Local prediction comparison (not used in final analysis)
├── .claude/settings.local.json
└── docs/
    ├── plan.md                      # Original experiment plan
    ├── results.md                   # Full results report with tables
    ├── chat_context.md              # This file
    ├── border_crop_visualization.png
    ├── border_corner_zoom.png
    ├── before_after_5pct.png
    ├── area_retention_chart.png
    ├── border_severity_analysis.png
    └── Image_Border_Artifact_Types_Table.png
```

## PACE Paths Used

```
PLANTCLEF_ROOT=/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0

Inputs:
  Test images:  $PLANTCLEF_ROOT/data/raw/PlanCLEF_test_data/images/
  DINOv2 model: $PLANTCLEF_ROOT/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth
  Baseline 4x4: $PLANTCLEF_ROOT/aerten3/parquet/plantclef_2026_test/phase1_tiles_test_jpeg85/
  Geo mask:     $PLANTCLEF_ROOT/aerten3/species_masking/sw_europe_mask_bool.npy
  Species map:  $PLANTCLEF_ROOT/data/raw/class_id_to_species_mapping/class_to_species_id.json

Outputs:
  Cropped images:   $PLANTCLEF_ROOT/aerten3/data/exp12_cropped_images/crop_{3,5,7}pct[_resized]/
  Parquets:         $PLANTCLEF_ROOT/aerten3/parquet/plantclef_2026_test/exp12_border_crop/
  Submissions:      $PLANTCLEF_ROOT/aerten3/submissions/exp12_border_crop/{baseline,crop_*}/
```

## Dependencies on Other Experiments
- **Exp-7** (`Exp-7_Multiscale_tiling/`): All extraction and aggregation scripts are imported from here via PYTHONPATH. No copies made.
- **Exp-4**: Baseline 4x4 parquets (`phase1_tiles_test_jpeg85/`) were extracted in Exp-4 with JPEG85 compression.
- SLURM venv: `/storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/experiments/001-embedding-extraction/.venv/`

## Gotchas / Things to Remember
- `multiscale_aggregate.py` requires `--model-path` even for CPU-only aggregation (it's a required arg, not used for inference in aggregation mode)
- GPU jobs on PACE should use `qos=embers` (not `inferno`) for this project
- Local Python is at `/c/Users/alper/anaconda3/python` (not `python3` which hits Windows Store alias)
- The `compare_submissions.py` script was created but the actual F1 comparison was done via Kaggle submissions, not locally (no ground truth available)
- A test crop run was started locally in `test_crop_output/` during EDA — this was killed and deleted, it's not needed

## What to Do Next (If Revisiting)
- This experiment is concluded. No further work needed.
- If exploring artifact mitigation, look at SAM-based soft tile weighting instead (Adrian's SAM3 masks exist at `user/adrian/experiments/001-sam3-pcs/`)
- The `crop_test_images.py` script with `--resize-back` could be reused if someone wants to test different preprocessing approaches
- PACE storage in `exp12_cropped_images/` and `exp12_border_crop/` can be cleaned up to free disk space
