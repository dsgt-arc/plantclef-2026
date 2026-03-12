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

## 5. Aggregation experiments (grid 4x4, top-k=5 per tile)
- [x] 5.1 Union, no cap → avg 37.2 species/image → F1 = 0.21313
- [x] 5.2 Union, max_species=20 (cap on unique) → avg 19.8 species/image → F1 ~0.21
- [x] 5.3 Union, max_species=5/8/10 (cap on unique) → F1 ~0.21
- [x] 5.4 Expanded-list aggregation, max_species=20 → avg ~10 species/image → **F1 = 0.23**
- [x] 5.5 Quadrat-level aggregation (no cap) on union topk5 → avg 183 species/image → F1 worse
- [x] 5.6 Quadrat-level aggregation (top-k=10) on expanded-list max20 → F1 = 0.17103
- **Conclusion**: Quadrat aggregation hurts performance with current approach. Expanded-list aggregation (0.23) is best so far.

## 6. Execution (grid sweep)
- [x] 6.1 Run no-grid baseline (grid_size=0) → F1 = 0.14506
- [ ] 6.2 Run all grid sizes (--all) with best aggregation and upload to Kaggle
- [ ] 6.3 Record Kaggle scores in results.md

## 7. Reproduce 2025 score (0.30810)
- [ ] 7.1 Verify checkpoint identity (task 4)
- [ ] 7.2 Understand 2025 quadrat-level aggregation — why it helped in 2025 but hurts now (see notes in proposal.md)
- [ ] 7.3 Try the organizers' max-probability aggregation (score-threshold approach, scored 0.23 in their notebook)
- [ ] 7.4 Try different max_species values with expanded-list aggregation (e.g., 10, 15, 25, 30)
- [ ] 7.5 Try lower top-k per tile (1, 2, 3) to see if fewer species per tile improves F1
- [ ] 7.6 Investigate if 2025's 0.30810 used a different pipeline entirely (different checkpoint, different aggregation, prior probabilities)

## 8. Analysis
- [ ] 8.1 Plot F1 vs grid size curve
- [ ] 8.2 Compare aggregation methods
- [ ] 8.3 Document findings in results.md
