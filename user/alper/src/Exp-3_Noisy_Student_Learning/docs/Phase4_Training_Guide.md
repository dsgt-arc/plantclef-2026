# Phase 4 — Student Training Execution Guide

## Prerequisites
- Phase 3 pseudo-labels generated at:
  `${PLANTCLEF_ROOT}/aerten3/parquet/lucas_sw77k/phase3_student_dataset/lucas_pseudo_labels.parquet`
  (62,185 images, T=2.0, k=6, thr=0.03)

## Execution Order

### Step 1: Run baseline (BCE with soft targets + neg sampling)
```bash
sbatch phase4/slurm_phase4_train.sbatch
```
- Uses BCE loss with soft teacher targets + 256 random negatives (proven by Exp-1, F1≈0.14)
- KL loss over all 7806 classes caused mode collapse (F1=0) — do NOT use
- `logit_mse` consistency with `lambda_cons=0.01`
- Output: `${PLANTCLEF_ROOT}/aerten3/checkpoints/phase4_checkpoints/run_kl_default/`
- ~30 min on RTX 6000

### Step 2: Submit ablation sweeps (priority order)
All are independent — can submit simultaneously, but if GPU budget is limited, prioritize:

```bash
# Priority 1 — Tile filtering (Phase 2 showed this is the biggest lever)
sbatch phase4/slurm_ablation_tiles.sbatch          # 5 runs (array 0-4)
#   0: none (baseline)
#   1: fixed_k, k=6
#   2: fixed_k, k=8
#   3: threshold, min_prob=0.05
#   4: relative, alpha=0.3

# Priority 2 — Consistency (find the right lambda for BCE baseline)
sbatch phase4/slurm_ablation_consistency.sbatch    # 5 runs (array 0-4)
#   0: off (lambda=0)
#   1: logit_mse, lambda=0.001
#   2: logit_mse, lambda=0.01 (baseline)
#   3: logit_mse, lambda=0.1
#   4: cosine, lambda=0.01

# Priority 3 — Negative sampling count (BCE hyperparameter)
sbatch phase4/slurm_ablation_loss.sbatch           # 4 runs (array 0-3)
#   0: neg=256 (baseline)
#   1: neg=64
#   2: neg=128
#   3: neg=512

# Priority 4 — Loss focus (top-K class focusing for consistency loss)
sbatch phase4/slurm_ablation_loss_focus.sbatch     # 4 runs (array 0-3)
#   0: all 7806 classes (baseline)
#   1: top 32 classes
#   2: top 64 classes
#   3: top 128 classes

# Priority 5 — Data quality (image confidence filter)
sbatch phase4/slurm_ablation_data_quality.sbatch   # 4 runs (array 0-3)
#   0: no filter (baseline)
#   1: min_image_conf=0.04
#   2: min_image_conf=0.05
#   3: min_image_conf=0.07

# Priority 6 — Learning rate (standard hyperparameter)
sbatch phase4/slurm_ablation_lr.sbatch             # 3 runs (array 0-2)
#   0: lr=3e-5
#   1: lr=1e-4 (baseline)
#   2: lr=3e-4
```

Total: 25 ablation runs + 1 baseline = 26 training runs

### Step 3: Analyze results
- Each run saves `training_history.json` + `best_model.pt`
- Compare best_loss across runs within each category
- Pick optimal config from each category

### Step 4: Phase 5 inference
```bash
sbatch phase5/slurm_phase5_inference.sbatch
```
- Update CHECKPOINT path to point to best ablation model

## Baseline Config (anchor for all ablations)
| Parameter        | Value      |
|------------------|------------|
| loss_type        | bce        |
| num_negatives    | 128        |
| cons_mode        | logit_mse  |
| cons_augment     | dropout    |
| lambda_cons      | 0.01       |
| warmup_epochs    | 5          |
| loss_topk        | 0 (all)    |
| tile_strategy    | threshold  |
| tile_min_prob    | 0.05       |
| min_image_conf   | 0.0        |
| lr               | 1e-4       |
| batch_size       | 32         |
| epochs           | 20         |
| hidden_dim       | 512        |
| num_heads        | 8          |
| num_layers       | 4          |
| dropout          | 0.1        |

## Output Directory Structure
```
phase4_checkpoints/
├── run_combined_baseline/        # combined OVAT winners + tile-drop consistency
├── ablation_loss/
│   ├── bce_neg256_baseline/
│   ├── bce_neg64/
│   ├── bce_neg128/
│   └── bce_neg512/
├── ablation_tiles/
│   ├── tile_none_baseline/
│   ├── tile_fixedk6/
│   ├── tile_fixedk8/
│   ├── tile_thr005/
│   └── tile_rel030/
├── ablation_consistency/
│   ├── cons_off/
│   ├── cons_logit_lam0.001/
│   ├── cons_logit_lam0.01_baseline/
│   ├── cons_logit_lam0.1/
│   └── cons_cosine_lam0.01/
├── ablation_lr/
│   ├── lr_3e-5/
│   ├── lr_1e-4_baseline/
│   └── lr_3e-4/
├── ablation_data_quality/
│   ├── conf_0.00_baseline/
│   ├── conf_0.04/
│   ├── conf_0.05/
│   └── conf_0.07/
└── ablation_loss_focus/
    ├── topk_0_baseline/
    ├── topk_32/
    ├── topk_64/
    └── topk_128/
```

## Ablation Priority Rationale

| Priority | Category | Runs | Why |
|----------|----------|------|-----|
| 1 | **Tile filtering** | 5 | Phase 2 proved tile filtering is the biggest lever for teacher predictions (+0.034 F1). Need to know if filtering at student training time also helps. |
| 2 | **Consistency** | 5 | Finds the right lambda for BCE baseline. Previous KL-based results are invalid. Lambda=0.1 may help; lambda=0 tells us if consistency hurts. |
| 3 | **Neg sampling** | 4 | BCE hyperparameter: how many random negatives per sample? Exp-1 used 256. Sweep 64/128/256/512. |
| 4 | **Loss focus** | 4 | Top-K class focusing now only affects consistency loss (BCE already focuses on positives+negatives). Tests if narrowing consistency to top-K classes helps. |
| 5 | **Data quality** | 4 | Tests whether removing low-confidence images helps training. Previous results (conf_0.04=0.017, conf_0.07=0.015) were with KL loss — re-run with BCE. |
| 6 | **Learning rate** | 3 | Standard hyperparameter tuning. |

**If GPU budget is tight**: Run tiles + consistency first (10 runs). These are the biggest levers now that loss is fixed.

## Post-Ablation: Combined Baseline + Ensemble + Sharpening

### OVAT Results Summary
| Category | Winner | F1 |
|---|---|---|
| Tiles | tile_thr005 | 0.21634 |
| Consistency | cons_off | 0.20316 |
| Neg sampling | bce_neg128 | 0.20299 |
| Loss focus | topk_0 (baseline) | 0.20061 |
| Data quality | conf_0.00 (baseline) | 0.20061 |
| LR | 1e-4 (baseline) | 0.20061 |
| **Teacher baseline** | — | **0.28093** |

### Step A: Combined OVAT Baseline
Winners merged: tile_thr005 + cons_off + neg128.
```bash
sbatch phase4/slurm_phase4_train.sbatch    # combined baseline (lambda_cons=0.0)
# After training:
sbatch phase5/slurm_phase5_inference.sbatch
```

### Step B: Ensemble Student + Teacher
Blends student and teacher predictions: `ensemble = alpha * teacher + (1-alpha) * student`.
```bash
# After Step A completes:
sbatch phase5/slurm_phase5_ensemble.sbatch  # sweeps alpha=0.3,0.5,0.7,0.9
```
Output: `${PLANTCLEF_ROOT}/aerten3/submissions/phase5_ensemble/`

### Step C: Pseudo-Label Sharpening
Re-generate Phase 3 with stricter thresholds / fewer retained classes.
```bash
# Step C.1: Generate sharpened pseudo-labels (CPU, 4 configs)
sbatch phase3/slurm_phase3_sharpened.sbatch
#   0: thr=0.05, image_topk=64
#   1: thr=0.07, image_topk=64
#   2: thr=0.03, image_topk=30
#   3: thr=0.05, image_topk=30

# Step C.2: Train on each (GPU, after C.1)
sbatch phase4/slurm_phase4_sharpened.sbatch

# Step C.3: Evaluate (GPU, after C.2)
sbatch phase5/slurm_phase5_sharpened_eval.sbatch
```

## Key Files
- `phase4/train_student.py` — training script (all CLI args)
- `phase4/ablation_sweep.py` — OVAT config definitions + launcher
- `phase4/dataset.py` — dataset with tile filtering + image confidence filtering
- `phase4/losses.py` — KL, BCE, and consistency loss (logit_mse/cosine/prob_mse modes)
- `phase5/ensemble_inference.py` — student + teacher ensemble blending
- `phase3/generate_pseudo_labels.py` — pseudo-label generation with configurable sharpness
