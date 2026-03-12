# Exp-4: Execution Guide

## Directory Structure

```
Exp-4_Enhanced_Tile_Aggregation/
├── utils/
│   ├── __init__.py
│   ├── aggregator.py          # 5 aggregators + factory function
│   └── io_bytes.py            # f16 byte encoding helpers
├── aggregate_tiles.py         # Core re-aggregation engine (Noisy-OR baseline)
├── generate_submission.py     # Aggregated parquet → Kaggle CSV
├── multiview_aggregate.py     # Multi-view TTA + aggregation method sweep
├── temperature_sweep.py       # Temperature × tile-k × threshold sweep
├── topk_sweep.sh              # Submission topk sweep (shell script)
├── slurm_multiview.sbatch     # SLURM job for Steps 1 + 3
├── slurm_sweep.sbatch         # SLURM job for temperature sweep
└── docs/
    ├── research_plan.md       # Research motivation and experimental design
    └── execution_guide.md     # This file
```

## Script Descriptions

### 1. `utils/aggregator.py` — Aggregation Methods

Contains 5 tile-to-image aggregation classes and a factory function:

- **TemperatureNoisyORAggregator** — Standard Noisy-OR with temperature softmax. Current baseline (F1=0.28093 at T=2.0, k=6).
- **MeanProbabilityAggregator** — Averages softmax probabilities across tiles. Linear — no multiplicative noise inflation.
- **MaxProbabilityAggregator** — Takes max softmax probability per class across tiles. Conservative — one confident tile is sufficient.
- **WeightedNoisyORAggregator** — Noisy-OR with per-tile confidence weights (w_t = max_prob_t / sum). Suppresses noisy low-confidence tiles.
- **`create_aggregator(method, ...)`** — Factory function dispatching by method name: `noisy_or`, `weighted_noisy_or`, `mean_prob`, `max_prob`.

### 2. `multiview_aggregate.py` — Multi-View TTA + Sweep

Main script for this experiment. Loads the DINOv2 classifier head (just a 768x7806 linear layer), computes logits from stored flip/rot90 embeddings via numpy matmul, and sweeps view strategies × aggregation methods.

**View strategies**:
- `avg_logits` — Average orig/flip/rot90 logits per tile, then aggregate across tiles
- `all_tiles` — Treat each view as a separate tile (3x tiles in aggregator)
- `per_view_or` — Aggregate each view separately, average 3 image-level prob vectors
- `orig_only` — Baseline (only original logits, for comparison)

**CLI arguments**:
```
--tile-parquet-root    Per-tile parquet from Exp-3 Phase 1
--out-dir              Output directory for parquets + CSVs
--model-path           DINOv2 checkpoint (.pth) for head extraction
--strategies           Comma-separated view strategies (default: all 4)
--aggregation-methods  Comma-separated agg methods (default: noisy_or)
--temperatures         Comma-separated temperatures (default: 2.0)
--tile-filter-ks       Comma-separated tile-filter-k values (default: 6)
--thresholds           Comma-separated prob thresholds (default: 0.03)
--topk                 Top-K species in submission CSV (default: 10)
--species-map          Class-to-species JSON mapping
--class-mask-npy       Geo species mask .npy (optional)
```

### 3. `aggregate_tiles.py` — Core Re-Aggregation Engine

Reads per-tile parquet, applies tile filtering (aug consistency + top-k by confidence), runs Noisy-OR aggregation. Used by `temperature_sweep.py` for baseline sweeps.

Note: parquet column `teacher_logits_f16` is a legacy name from the Exp-3 extraction pipeline — it refers to the DINOv2 model's per-tile logits.

### 4. `generate_submission.py` — Submission CSV Generator

Reads aggregated image-level parquet and produces Kaggle CSV with top-K species per image. Can be run standalone for quick topk/threshold experiments on any aggregated parquet.

### 5. `temperature_sweep.py` — Baseline Hyperparameter Sweep

Sweeps temperature × tile-filter-k × threshold on the original (single-view) pipeline. Useful for re-establishing baselines or testing new temperature ranges.

### 6. `topk_sweep.sh` — Submission topk Sweep

Shell script that regenerates CSVs from an existing aggregated parquet with different topk values. CPU-only, takes seconds.

```
Usage: bash topk_sweep.sh <aggregated-parquet> [out-dir]
```

Sweeps topk in {5, 8, 10, 12, 15, 20} × threshold in {0.03, 0.0}.

## Execution Order on PACE

### Step 1 + Step 3: Multi-View Strategies + Aggregation Methods

Single SLURM job runs all sweeps sequentially:

```bash
cd /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-4_Enhanced_Tile_Aggregation
sbatch slurm_multiview.sbatch
```

This runs 4 sub-sweeps:

| Sub-step | Description | Configs |
|----------|-------------|---------|
| Step 1 | All 4 view strategies at T=2.0, k=6 | 4 submissions |
| Step 1b | `all_tiles` k sweep (k=2,4,6,8,10,12,16) | 7 submissions |
| Step 1c | Temperature sweep (T=1.5,2.0,2.5) for `avg_logits` + `per_view_or` at k=4,6,8 | 18 submissions |
| Step 3 | All 4 aggregation methods × {`avg_logits`, `orig_only`} × k={6,10,16} | 24 submissions |

**Output**: `$PLANTCLEF_ROOT/aerten3/submissions/exp4_multiview/`
- Aggregated parquets: `mv_{strategy}_{agg_method}_T{temp}_k{k}.parquet`
- Submission CSVs: `mv_{strategy}_{agg_method}_T{temp}_k{k}_thr{thr}_top{topk}.csv`

**Resource**: CPU-medium, 48 GB RAM, ~4 hours

### Step 2: topk Sweep (run in parallel with Step 1)

Run on the current best baseline parquet AND on the best multi-view parquet after Step 1 completes:

```bash
# On the existing Exp-3 best baseline (can run immediately)
bash topk_sweep.sh \
  $PLANTCLEF_ROOT/aerten3/submissions/phase2_sweep/sweep_T2.0_k6.parquet

# On the best multi-view result (after Step 1 completes)
bash topk_sweep.sh \
  $PLANTCLEF_ROOT/aerten3/submissions/exp4_multiview/step1_views/mv_avg_logits_noisy_or_T2.0_k6.parquet
```

**Output**: `topk_sweep/` subdirectory next to the input parquet
- CSVs: `{basename}_top{K}_thr{THR}.csv` for each K × THR combo

**Resource**: CPU, ~1 minute total

### Step 4: Combined Best Config

After submitting Step 1-3 CSVs to Kaggle:

1. Identify the best view strategy from Step 1
2. Identify the best topk from Step 2
3. Identify the best aggregation method from Step 3
4. Run one final config combining all winners:

```bash
python multiview_aggregate.py \
  --tile-parquet-root $PLANTCLEF_ROOT/aerten3/parquet/plantclef_2026_test/phase1_tiles_test \
  --out-dir $PLANTCLEF_ROOT/aerten3/submissions/exp4_multiview/step4_combined \
  --model-path $PLANTCLEF_ROOT/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth \
  --species-map $PLANTCLEF_ROOT/data/raw/class_id_to_species_mapping/class_to_species_id.json \
  --class-mask-npy $PLANTCLEF_ROOT/aerten3/species_masking/sw_europe_mask_bool.npy \
  --strategies "<best_strategy>" \
  --aggregation-methods "<best_method>" \
  --temperatures "<best_temp>" \
  --tile-filter-ks "<best_k>" \
  --thresholds "0.03" \
  --topk <best_topk>
```

## Submitting to Kaggle

After generating CSVs, submit to the PlantCLEF 2026 competition:

```bash
# From any directory with kaggle CLI configured
kaggle competitions submit \
  -c plantclef-2026 \
  -f /path/to/submission.csv \
  -m "Exp4: <strategy> <agg_method> T=<temp> k=<k> topk=<topk>"
```

## Baseline for Comparison

All results should be compared against the Exp-3 Phase 2 optimal config:

- **Configuration**: T=2.0, k=6, thr=0.03, topk=10, Noisy-OR, orig_only
- **F1 Score**: 0.28093
- **Parquet**: `$PLANTCLEF_ROOT/aerten3/submissions/phase2_sweep/sweep_T2.0_k6.parquet`

The `orig_only` strategy in Step 1 serves as a sanity check — it should reproduce this baseline exactly.
