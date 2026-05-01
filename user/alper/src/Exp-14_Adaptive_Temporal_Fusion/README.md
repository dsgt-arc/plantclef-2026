# Exp-14 — Adaptive Temporal Fusion

Per-image adaptive α + alternative prior aggregators for the Exp-11B
ensemble. Aims to fix the lush-prior-clobbers-barren-image pathology
on multi-visit sites (RNNB pairs, CBN-PdlC sequences) without breaking
the working alpine cases.

See [docs/Exp14_Plan.md](docs/Exp14_Plan.md) for the full plan, or
the `Context` and `Anti-overfit principles` sections for the short
version.

## Quickstart

Phase 0 (run on local after pulling parquets from PACE):

```bash
# 1. Sanity check: pre-fusion parquet rows must differ between paired RNNB visits
python diagnostics/prefusion_sanity_check.py \
    --prefusion-parquet /local/path/knn_bM0.85_bS1.00_T1.50_prefusion.parquet \
    --out-md docs/Exp14_Diagnostic_Findings.md

# 2. Vegetation density (only test JPEGs needed)
python signals/compute_density.py \
    --images-dir 'C:/Users/alper/Documents/PlantCLEF_2026_Local/PlantCLEF2025test/images' \
    --out outputs/signals/density.parquet

# 3. Photo confidence stats (per branch)
python signals/compute_confidence.py \
    --parquet /local/path/knn_bM0.85_bS1.00_T1.50_prefusion.parquet \
    --out outputs/signals/confidence_vit.parquet

# 4. Visualize: side-by-side image + P_photo / P_loc / P_final for the 8 user-flagged + random multi-visit pairs
python diagnostics/viz_per_image.py \
    --prefusion-parquet /local/path/knn_bM0.85_bS1.00_T1.50_prefusion.parquet \
    --images-dir 'C:/Users/alper/Documents/PlantCLEF_2026_Local/PlantCLEF2025test/images' \
    --quadrat-ids RNNB-3-1-20230512 RNNB-3-1-20240117 RNNB-3-2-20230512 RNNB-3-2-20240117 \
                  RNNB-2-7-20230512 RNNB-2-7-20240117 RNNB-1-2-20230512 RNNB-1-2-20240117 \
    --out-pdf outputs/viz/rnnb_pairs.pdf

# 5. Baseline proxy metrics
python diagnostics/proxy_metrics.py \
    --csv 'C:/Users/alper/Desktop/exp11b_ensemble_arcface_knn/ens_knn_w0.77_thr0.100_top10.csv' \
    --density outputs/signals/density.parquet \
    --prefusion-parquet /local/path/knn_bM0.85_bS1.00_T1.50_prefusion.parquet \
    --species-map /local/path/species_map.json \
    --out-json outputs/baseline_proxies.json
```

Phase 4 (PACE):

```bash
# Compute visit-similarity once
sbatch slurm/slurm_signals.sbatch

# Stage 0 parity check — must reproduce ens_knn_w0.77_thr0.100_top10.csv
STAGE_SPEC=stage_configs/stage0_baseline.json \
VIT_PREFUSION=... CNXT_SCALE_4=... CNXT_SCALE_5=... CNXT_SCALE_6=... \
SPECIES_MAP=... CLASS_MASK=... BASELINE_CSV=... \
sbatch slurm/slurm_stage_sweep.sbatch

# Stage 1 — ViT prior aggregator sweep
STAGE_SPEC=stage_configs/stage1_vit_aggregators.json ... sbatch slurm/slurm_stage_sweep.sbatch
```

## Key files

| Path | Purpose |
|---|---|
| `docs/Exp14_Plan.md` | Full plan |
| `docs/Exp14_Diagnostic_Findings.md` | Filled by Phase 0–1 |
| `docs/Exp14_Results_Summary.md` | Filled by Phase 5 |
| `diagnostics/prefusion_sanity_check.py` | Phase 0.2 hard gate |
| `diagnostics/viz_per_image.py` | Side-by-side PDF viz |
| `diagnostics/proxy_metrics.py` | 4 internal proxies |
| `signals/compute_density.py` | ExG/HSV vegetation density (local) |
| `signals/compute_confidence.py` | c_max / c_top5 / c_entropy from parquet |
| `signals/compute_visit_similarity.py` | Pairwise CLS-cosine within location (PACE) |
| `signals/compute_season_weights.py` | Cyclical month-distance weights |
| `controller/alpha_controller.py` | Per-image α controller |
| `controller/prior_aggregators.py` | max / mean / top2_mean / sim_w / conf_w / season_w |
| `scripts/fuse_v2.py` | Fusion library function |
| `scripts/ensemble_v2.py` | Full ensemble pipeline (sibling of Exp-11/ensemble_with_knn.py) |
| `scripts/stage_runner.py` | Run a stage's variant sweep, score by proxies |
| `stage_configs/` | One JSON per stage |
| `slurm/slurm_signals.sbatch` | Compute visit similarity |
| `slurm/slurm_stage_sweep.sbatch` | Run a stage on PACE |

## Anti-overfit guardrails (binding)

1. Internal proxies (RNNB pair divergence, stable-site invariance,
   density-count correlation, photo faithfulness) are the primary
   decision signal — not the public LB.
2. Coarse parameter precision: α at {0.15, 0.30, 0.50, 0.70} grain.
3. ≤8 LB submissions across the entire experiment.
4. Mechanism must be explicable in one sentence.

## Out of scope

- SAM / GroundingDINO masking
- Backbone retraining
- New ensemble members (Exp-13 ViT-L)
- Per-class beta recalibration (Exp-9B β=0.85 held fixed)
