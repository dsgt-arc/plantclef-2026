# Tasks

## 1. Setup
- [x] 1.1 Copy classify.py from exp 001 (self-contained, no plantclef imports)
- [x] 1.2 Create CPU-only pyproject.toml (pandas, pyarrow, numpy)
- [ ] 1.3 Validate: `python classify.py --help` parses correctly
- [ ] 1.4 Validate: `classify.py --input ../001-dinov2-grid-ablation/logits/grid_4x4.parquet --method expanded-list --top-k 5 --max-species 20` reproduces existing CSV

## 2. Aggregation experiments (grid 4x4, top-k=5 per tile)
- [x] 2.1 Union, no cap → avg 37.2 species/image → F1 = 0.21313
- [x] 2.2 Union, max_species=20 (cap on unique) → avg 19.8 species/image → F1 ~0.21
- [x] 2.3 Union, max_species=5/8/10 (cap on unique) → F1 ~0.21
- [x] 2.4 Expanded-list aggregation, max_species=20 → avg ~10 species/image → **F1 = 0.23**
- [x] 2.5 Quadrat-level aggregation (no cap) on union topk5 → avg 183 species/image → F1 worse
- [x] 2.6 Quadrat-level aggregation (top-k=10) on expanded-list max20 → F1 = 0.17103
- **Conclusion**: Quadrat aggregation hurts performance with current approach. Expanded-list aggregation (0.23) is best so far.

## 3. New aggregation experiments
- [ ] 3.1 Try max-prob method (organizers' approach, scored 0.23 in their notebook)
- [ ] 3.2 Try different max_species values with expanded-list (10, 15, 25, 30)
- [ ] 3.3 Try lower top-k per tile (1, 2, 3) to see if fewer species per tile improves F1

## 4. Reproduce 2025 score (0.30810)
- [ ] 4.1 Try the organizers' max-probability aggregation (score-threshold approach)
- [ ] 4.2 Try different max_species values with expanded-list aggregation
- [ ] 4.3 Try lower top-k per tile (1, 2, 3)
- [ ] 4.4 Investigate if 2025's 0.30810 used a different pipeline entirely (different checkpoint, different aggregation, prior probabilities)

## 5. Analysis
- [ ] 5.1 Compare aggregation methods
- [ ] 5.2 Document findings in results.md
