---
experiment: "003-tileq-decoder"
status: in-progress
date: 2026-05-03
author: Murilo Gustineli
---

# Experiment 003: TileQ-Decoder — Results

## Status: IN PROGRESS

A0 submission-pipeline derisk **complete**. A0b reproduces the validated 2025
baseline F1=0.30810 exactly; A0a lands at the expected ~0.23. Writer + Kaggle
submission format confirmed clean. Decoder training (A1/A2) blocked on test-CLS
extraction (Priority 3) and library code (Priority 4).

See [proposal.md](proposal.md) for hypothesis framing (H1/H2) and
[spec.md](spec.md) for the technical specification.

## Kaggle Leaderboard Results — A0 baselines

Competition metric: **macro-averaged F1** per image. All scores are public LB.

| # | Method | Avg species/img | Public LB | Expected | Match? |
|---|--------|----------------:|----------:|---------:|--------|
| **A0a** | `max-prob`, top-10, grid_4x4 | 10.0 | **0.22986** | ~0.23 | ✓ |
| **A0b** | `sorted-prob`, top-9, grid_4x4 | 5.0 | **0.30810** | 0.30810 | ✓ exact |

Both submissions generated from the existing
[`experiments/002-classification-methods/classify.py`](../002-classification-methods/classify.py)
against the cached
`shared/inference/dinov2-vit-base/grid_4x4.parquet` (525 MB, 33,680 rows ×
float16 probs[7806]). No new code.

### Reproduction commands

```
# A0a
python ../002-classification-methods/classify.py \
    --grid-size 4 --method max-prob --max-species 10 \
    --output-dir submissions/

# A0b
python ../002-classification-methods/classify.py \
    --grid-size 4 --method sorted-prob --top-k 9 \
    --output-dir submissions/
```

CSV format: `quadrat_id, species_ids` with `species_ids` as `"[id1, id2, ...]"`,
double-quoted, header row included.

## Interpretation

1. **Submission pipeline derisked.** Kaggle accepted both CSVs without
   formatting errors. The writer in `classify.py:451-462` is wired correctly
   for the 2026 competition format.
2. **`grid_4x4.parquet` has not drifted** since the 2025 reference CSV was
   generated. A0b's 0.30810 LB is byte-exact reproduction of the
   `dsgt_run_topk_9_grid_4x4.csv` reference confirmed in
   [exp 002](../002-classification-methods/results.md) (99.9 % per-image
   match, 2103/2105 identical). Re-confirmed under the 2026 competition LB.
3. **A0a vs A0b restates the exp 002 finding**: probability magnitude
   (sorted-prob) > tile frequency / count-based methods (max-prob).
   Δ = +0.078 F1 on the same per-tile predictions. This is the load-bearing
   prior for the decoder approach in spec §2.1 — the decoder consumes
   per-tile CLS embeddings (not just probabilities) and learns its own
   confidence-weighted aggregation.
4. **A0b is the comparison anchor for H1.** Decoder must clear F1=0.30810
   to confirm H1 (decoder beats `max_prob` and the sorted-prob aggregator
   without test-side knowledge).

## What's next (priority-ordered)

Per [tasks.md](tasks.md) and [CONTEXT.md](../../CONTEXT.md):

- [x] **Priority 2** — A0a/A0b submissions, Kaggle pipeline derisked
- [ ] **Priority 3** — Test-CLS extraction (~2 GPU-h). Add `--save-cls` flag
      to `experiments/001-dinov2-grid-ablation/inference.py`. Output:
      `shared/inference/dinov2-vit-base/cls_grid_4x4.parquet`
      (33,680 rows × 768-d float16, ~50 MB). Validate row count + per-image
      groups = 16 before training.
- [ ] **Priority 4** — Library code: `tileq_decoder.py`, `asl_loss.py`,
      `data/samplers.py`, `build_train_lookup.py`,
      `build_faiss_background_map.py`. Sanity checks (overfit-one-batch,
      shape audit, sampler unit tests) must pass before any long training.
- [ ] **Priority 5** — Training Run 1 (Sampler A) + Run 2 (Sampler B),
      ~4 GPU-h each on RTX 6000.
- [ ] **Priority 6** — Inference + headline submissions A1, A2.
- [ ] **Priority 7** — A3 score-blend bonus (5 α values, no retraining).
- [ ] **Priority 8** — Writeup; decide H1, H2 outcomes.

## Files

| File | Purpose |
|------|---------|
| `submissions/A0a_max_prob_grid4.csv` | A0a Kaggle submission (LB 0.22986) |
| `submissions/A0b_sorted_prob_topk9_grid4.csv` | A0b Kaggle submission (LB 0.30810) |
| `../002-classification-methods/classify.py` | A0a/A0b producer — no new code |
| `shared/inference/dinov2-vit-base/grid_4x4.parquet` | Cached test probs (input) |
| `shared/data/raw/class_id_to_species_mapping/class_mapping.txt` | class_idx → species_id |

Branch: `mg/tileq-decoder` (PR #15)
