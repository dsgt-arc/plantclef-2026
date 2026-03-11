# Phase 4 — Student Training Execution Guide

## Prerequisites
- Phase 3 pseudo-labels generated at:
  `${PLANTCLEF_ROOT}/aerten3/parquet/lucas_sw77k/phase3_student_dataset/lucas_pseudo_labels.parquet`
  (62,185 images, T=2.0, k=6, thr=0.03)

## Execution Order

### Step 1: Re-run baseline with fixed consistency loss
```bash
sbatch phase4/slurm_phase4_train.sbatch
```
- Uses `logit_mse` consistency mode (was broken `prob_mse` → cons=0.000000)
- New default: `lambda_cons=0.01` (was 0.1, reduced because logit MSE is ~100x larger)
- **VERIFY**: `cons` column in logs should no longer be `0.000000`
- Output: `${PLANTCLEF_ROOT}/aerten3/checkpoints/phase4_checkpoints/run_kl_default/`
- ~30 min on RTX 6000

### Step 2: Submit ablation sweeps (priority order)
All are independent — can submit simultaneously, but if GPU budget is limited, prioritize:

```bash
# Priority 1 — Loss focus (top-K class focusing — fixes noisy all-class losses)
sbatch phase4/slurm_ablation_loss_focus.sbatch     # 4 runs (array 0-3)
#   0: all 7806 classes (baseline)
#   1: top 32 classes
#   2: top 64 classes
#   3: top 128 classes

# Priority 2 — Consistency (validates logit_mse fix, finds right lambda)
sbatch phase4/slurm_ablation_consistency.sbatch    # 5 runs (array 0-4)
#   0: off (lambda=0)
#   1: logit_mse, lambda=0.001
#   2: logit_mse, lambda=0.01 (baseline)
#   3: logit_mse, lambda=0.1
#   4: cosine, lambda=0.01

# Priority 3 — Loss type (core design: soft distillation vs hard pseudo-labels)
sbatch phase4/slurm_ablation_loss.sbatch           # 2 runs (array 0-1)
#   0: kl (baseline)
#   1: bce with negative sampling

# Priority 4 — Tile filtering (Phase 2 showed this is the biggest lever)
sbatch phase4/slurm_ablation_tiles.sbatch          # 5 runs (array 0-4)
#   0: none (baseline)
#   1: fixed_k, k=6
#   2: fixed_k, k=8
#   3: threshold, min_prob=0.05
#   4: relative, alpha=0.3

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

Total: 23 ablation runs + 1 baseline = 24 training runs

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
| loss_type        | kl         |
| cons_mode        | logit_mse  |
| lambda_cons      | 0.01       |
| warmup_epochs    | 5          |
| loss_topk        | 0 (all)    |
| tile_strategy    | none       |
| min_image_conf   | 0.0        |
| lr               | 1e-4       |
| batch_size       | 32         |
| epochs           | 15         |
| hidden_dim       | 512        |
| num_heads        | 8          |
| num_layers       | 4          |
| dropout          | 0.1        |

## Output Directory Structure
```
phase4_checkpoints/
├── run_kl_default/              # baseline
├── ablation_loss/
│   ├── kl_baseline/
│   └── bce_neg64/
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
| 1 | **Loss focus** | 4 | Top-K class focusing. Baseline trains on all 7806 classes but only ~2-10 have signal — the rest is noise. Baseline showed consistency loss GROWING (views diverge) because MSE is dominated by irrelevant classes. Top-K should fix this and change interpretation of all other ablations. |
| 2 | **Consistency** | 5 | Validates the `logit_mse` fix works and finds the right lambda. Baseline showed lambda=0.01 is too weak (cons grows unchecked). Lambda=0.1 may balance pseudo_loss vs cons_loss. |
| 3 | **Loss type** | 2 | Only 2 runs (fast). KL vs BCE is a fundamental design choice — soft distillation preserves teacher uncertainty, BCE uses hard labels. Note: BCE already focuses on positive labels + 64 negatives (inherently top-K). |
| 4 | **Tile filtering** | 5 | Phase 2 proved tile filtering is the biggest lever for teacher predictions (+0.034 F1). Need to know if filtering at student training time also helps. |
| 5 | **Data quality** | 4 | Tests whether removing low-confidence images (teacher says "I don't know") helps training. The histogram printed at load time reveals the image confidence distribution — useful info even if optimal threshold = 0. |
| 6 | **Learning rate** | 3 | Lowest priority. Standard hyperparameter tuning, least likely to produce a surprising result. |

**If GPU budget is tight**: Run loss_focus + consistency first (9 runs). These fix the two biggest issues — noisy all-class losses and finding the right consistency lambda.

## Key Files
- `phase4/train_student.py` — training script (all CLI args)
- `phase4/ablation_sweep.py` — OVAT config definitions + launcher
- `phase4/dataset.py` — dataset with tile filtering + image confidence filtering
- `phase4/losses.py` — KL, BCE, and consistency loss (logit_mse/cosine/prob_mse modes)
