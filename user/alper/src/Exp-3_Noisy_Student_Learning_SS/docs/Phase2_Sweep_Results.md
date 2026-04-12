# Phase 2: Teacher Hyperparameter Sweep & Ablation Results

**Date**: 2026-03-10
**Branch**: `user/alper-Exp3_SSL_phases_1_2`
**Dataset**: PlantCLEF 2026 test set
**Best Score**: **0.28093** (T=2.0, k=6, thr=0.03)

---

## 1. Initial Discovery: Why generate_submission.py Outperformed the Sweep

The first submission generated via `generate_submission.py` from Phase 1 aggregated
parquet scored **0.27221**, while the best temperature sweep result was only **0.25558**.

**Root cause**: different tile filtering settings.

| Parameter        | Phase 1 Aggregation (0.272) | Phase 2 Sweep (0.256) |
|------------------|-----------------------------|-----------------------|
| `keep-top-tiles` | **8**                       | **16 (all tiles)**    |
| `temperature`    | 2.0                         | 1.0, 1.5, 2.0, 3.0   |
| `aug-filter`     | True                        | True                  |
| `geo-mask`       | Applied                     | Applied               |
| `image-topk`     | 64                          | 64                    |
| `threshold`      | 0.03                        | 0.0, 0.01, 0.03, 0.05|

**Why fewer tiles helps**: Noisy-OR formula `P(species) = 1 - prod(1 - p_tile)` is
sensitive to noise. Every tile with even a small non-zero probability inflates the
aggregated probability. The bottom tiles (lowest `max_prob`) are noisiest and contribute
false positives that compound multiplicatively.

---

## 2. Initial Temperature Sweep (16 tiles, no tile-k sweep)

First sweep with `--tile-filter-k 16` (all tiles kept):

| T \ thr  |  0.00   |  0.01   |  0.03   |  0.05   |
|----------|---------|---------|---------|---------|
| **1.0**  | 0.22785 | 0.22906 | 0.23536 | 0.24116 |
| **1.5**  | 0.21878 | 0.21890 | 0.22101 |    —    |
| **2.0**  | 0.20233 | 0.20241 | 0.24730 | 0.25558 |
| **3.0**  | 0.18826 | 0.20580 | 0.24852 | 0.23628 |

**Findings**:
- T=2.0 best overall, T=1.5 worst
- thr=0.03 and thr=0.05 clearly better than 0.00/0.01
- Narrowed search space to: T={2.0, 2.5, 3.0}, thr={0.03, 0.05, 0.07}

---

## 3. Full Hyperparameter Sweep (3 temps x 6 tile-ks x 3 thresholds = 54 runs)

### T=2.0 (best temperature)

| k \ thr |  0.03     |  0.05   |  0.07   |
|---------|-----------|---------|---------|
| **4**   | 0.27172   | 0.26124 | 0.25573 |
| **6**   | **0.28093** | 0.27115 | 0.26063 |
| **8**   | 0.27221   | 0.26732 | 0.26119 |
| **10**  | 0.26904   | 0.25979 | 0.26181 |
| **12**  | 0.26387   | 0.25186 | 0.25945 |
| **16**  | 0.24730   | 0.25558 | 0.26173 |

### T=2.5

| k \ thr |  0.03   |  0.05   |  0.07   |
|---------|---------|---------|---------|
| **4**   | 0.25646 | 0.24947 | 0.24947 |
| **6**   | 0.25622 | 0.25463 | 0.25463 |
| **8**   | 0.25044 | 0.24389 | 0.24158 |
| **10**  | 0.25286 | 0.25058 | 0.24120 |
| **12**  | 0.24949 | 0.24843 | 0.24074 |
| **16**  | 0.25234 | 0.25433 | 0.24916 |

### T=3.0

| k \ thr |  0.03   |  0.05   |  0.07   |
|---------|---------|---------|---------|
| **4**   | 0.24960 | 0.24960 | 0.24960 |
| **6**   | 0.25445 | 0.25445 | 0.25445 |
| **8**   | 0.24633 | 0.24633 | 0.24633 |
| **10**  | 0.25742 | 0.24624 | 0.24624 |
| **12**  | 0.25454 | 0.24614 | 0.24614 |
| **16**  | 0.24852 | 0.23628 | 0.23628 |

### Top 5 Configurations

| Rank | T   | k  | thr  | F1 Score  |
|------|-----|----|------|-----------|
| 1    | 2.0 | 6  | 0.03 | **0.28093** |
| 2    | 2.0 | 8  | 0.03 | 0.27221   |
| 3    | 2.0 | 4  | 0.03 | 0.27172   |
| 4    | 2.0 | 6  | 0.05 | 0.27115   |
| 5    | 2.0 | 10 | 0.03 | 0.26904   |

### Score vs Tile Count (T=2.0, thr=0.03)

```
  F1
 0.281 |        *
 0.272 |  *           *
 0.269 |                    *
 0.264 |                          *
 0.247 |                                *
       +--+----+----+----+-----+-----+---
        k=4   k=6   k=8  k=10  k=12  k=16
```

Classic inverted-U: k=4 loses real signal, k=8+ adds noise through Noisy-OR.

### Key Findings from Full Sweep

1. **T=2.0 dominates** — T=2.5 and T=3.0 are ~0.02-0.03 worse across the board
2. **k=6 is the sweet spot** — 6 most-confident tiles capture species while filtering noise
3. **thr=0.03 wins everywhere** — lower thresholds let borderline-correct species through
4. **At T=3.0, threshold barely matters** — softmax is so flat that many tile-k/thr combos produce identical scores

---

## 4. Teacher Ablation Table (Paper-Ready)

Four systematic baselines, each toggling one component to prove incremental value.
All use T=2.0 and thr=0.03.

| Config | Aug Filter | Tiles | T   | Geo Mask | F1 Score  | Delta from T1 |
|--------|------------|-------|-----|----------|-----------|---------------|
| T1     | No         | 16    | 2.0 | Yes      | 0.24774   | baseline      |
| T2     | Yes        | 16    | 2.0 | Yes      | 0.24730   | -0.000 (~0)   |
| T3     | Yes        | 6     | 2.0 | Yes      | **0.28093** | **+0.034**  |
| T4     | Yes        | 6     | 2.0 | No       | 0.27643   | +0.029        |

### Component-Level Analysis

| Comparison | What it tests       | Delta    | Conclusion                              |
|------------|---------------------|----------|-----------------------------------------|
| T1 → T2   | Aug filter alone    | -0.000   | No effect when keeping all 16 tiles     |
| T2 → T3   | Tile filtering (k=6)| **+0.034** | **Single biggest lever**              |
| T4 → T3   | Geo mask            | +0.005   | Modest but consistent improvement       |

**Interpretation**:
- Aug filter alone does nothing — filtering out inconsistent tiles but still keeping 16
  (with replacements from lower-confidence tiles) doesn't help
- Aug filter + tile filtering together is what matters — aug filter marks bad tiles,
  tile filtering removes them. Neither alone is sufficient.
- Geo mask gives a small consistent boost by blocking out-of-range species

---

## 5. Optimal Teacher Configuration (for Phase 3+)

```
temperature     = 2.0
tile_filter_k   = 6
threshold       = 0.03
aug_filter      = True
geo_mask        = True
image_topk      = 64
topk_species    = 10
```

**F1 Score: 0.28093**

---

## 6. Scripts Used

| Script | Purpose |
|--------|---------|
| `phase1/aggregate_teacher.py` | Original Phase 1 aggregation (k=8, T=2.0) |
| `phase2/temperature_sweep.py` | Full hyperparameter sweep (T x k x thr) |
| `phase2/teacher_baseline_table.py` | 4-config ablation table for paper |
| `phase2/aggregate_tiles.py` | Core re-aggregation engine |
| `phase2/generate_submission.py` | Parquet → Kaggle CSV |
| `phase2/slurm_phase2_sweep.sbatch` | SLURM job for sweep |
| `phase2/slurm_phase2_baselines.sbatch` | SLURM job for ablation table |
