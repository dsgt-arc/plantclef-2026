# Tasks

## 1. Setup
- [x] 1.1 Verify model checkpoint loads correctly (fast-dev-run)
- [x] 1.2 Verify test images are accessible and correctly discovered
- [x] 1.3 Validate submission CSV format matches Kaggle requirements

## 2. Profiling
- [x] 2.1 Run gpu-check to verify GPU utilization
- [x] 2.2 Estimate wall-clock time per grid size

## 3. Validate inference against 2025 baseline
- [x] 3.1 Run grid 4x4 with topk=5 and various aggregations → compared against 2025's `dsgt_run_topk_20_species_grid_4x4.csv`
- [x] 3.2 CSVs differ significantly: 2026 produces avg 37 species/image vs 2025's avg 10.3
- [x] 3.3 Species overlap is high (85-92% of 2025 species found in 2026 output) → model predictions are consistent, aggregation logic differs
- [ ] 3.4 Identify exactly how the 2025 reference CSV was generated (SubmissionTask was commented out in 2025 workflow.py, unknown code generated it)

## 4. Verify checkpoint identity
- [ ] 4.1 Confirm `vit_base_patch14_reg4_dinov2.pth` (2026) is the same weights as `model_best.pth.tar` (2025)
- [ ] 4.2 If different, locate and use the correct 2025 checkpoint

## 5. Pipeline refactor: 2-stage pipeline
- [x] 5.1 Refactor classify.py into inference.py (Stage 1: GPU → parquet) + aggregate.py (Stage 2: CPU → CSV)
- [x] 5.2 Update sbatch scripts for Stage 1 only (aggregation runs separately on CPU)
- [x] 5.3 Add pyarrow dependency
- [x] 5.4 Split aggregation into Experiment 002
- [x] 5.5 Validate: `inference.py --grid-size 4 --fast-dev-run` → parquet written with correct shape
- [x] 5.6 Move parquet output to shared storage (`shared/inference/dinov2-vit-base/`)

## 6. Execution (grid sweep)
- [x] 6.1 Run no-grid baseline (grid_size=0) → F1 = 0.14506
- [~] 6.2 Run all grid sizes (--all) with Stage 1 — jobs submitted, awaiting completion
- [ ] 6.3 Record Kaggle scores in results.md (aggregate with exp 002)

## 7. Checkpoint verification
- [ ] 7.1 Confirm `vit_base_patch14_reg4_dinov2.pth` (2026) is the same weights as `model_best.pth.tar` (2025)
- [ ] 7.2 If different, locate and use the correct 2025 checkpoint

## 8. Analysis
- [ ] 8.1 Plot F1 vs grid size curve
- [ ] 8.2 Document findings in results.md
