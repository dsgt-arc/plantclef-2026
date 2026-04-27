# Exp-15 — Photometric Test-Time Augmentation for Low-Contrast Vegetation

Photometric TTA on the ViT branch (the high-leverage 77 %-weight branch of
the Exp-14 ensemble): blend post-aggregation per-image probabilities from
the original tile pipeline with predictions on photometrically-enhanced
tiles. Goal: lift F1 above the locked Exp-14 winner of 0.45572 on
brownish/dormant cases without breaking working alpine cases.

See [docs/Exp15_Plan.md](docs/Exp15_Plan.md) for the full plan and rationale.

## Key design decisions

- **ViT branch only (initially).** ViT carries w = 0.77 ensemble weight; ConvNeXt only 0.23. Same standalone F1 lift on ViT translates to ~3.3× more in the blend. ConvNeXt symmetric attempt is gated to Phase 4b (only if Phase 3 wins clearly).
- **TTA via probability-level blend, not wholesale replacement.** ViT was trained with zero photometric augmentation; replacing test preprocessing pushes it OOD. Instead, run inference twice (original + photometric) and blend per-image probabilities. β = 1.0 recovers the Exp-14 winner exactly — graceful fallback by construction.
- **Fully data-driven Phase 1 gate.** No manual visual review. Auto-stratified probe sample (40 quadrats: 15 brownish + 10 lush + 10 very-barren + user-flagged seeds) + 6 quantitative proxy metrics decide which photometric variant survives. Brownish-vs-barren confidence gap is the primary mechanism check.

## Phases

| Phase | What | Where | Cost |
|---|---|---|---|
| 1 | Probe ViT 4× per tile on 40 quadrats with 4 variants; score by 6 proxies; pick survivor(s) | PACE RTX-6000 | ≤ 2 GPU-h |
| 2 | Full ViT re-extraction with surviving variant (per-tile teacher_logits + emb_arcface) | PACE RTX-6000 | ~4 GPU-h × N_surviving |
| 3 | Re-run multiscale_aggregate_knn → blend β-sweep → Exp-14 ensemble_v2 → habitat mask → LB | PACE CPU + Local | up to 7 LB |
| 4a | Bracket the Phase 3 winner | LB | up to 5 LB |
| 4b | ConvNeXt symmetric (gated) | PACE H200 | +8 GPU-h, up to 5 LB |

Worst-case total: ~25 GPU-h, ~17 LB submissions. Cheap path (negative Phase 1): ~2 GPU-h, 0 LB.

## Quickstart

### Step 0 (local, 30 sec): build probe sample

```bash
cd Exp-15_Photometric_TTA
python scripts/build_probe_sample.py
# wrote 39 quadrats -> outputs/probe/probe_sample.csv
```

### Step 1 (PACE, ~2h): submit Phase 1 probe job

```bash
sbatch slurm/slurm_probe_phase1.sbatch
# outputs/phase1/probe_logits.parquet
```

### Step 2 (local CPU, seconds): score Phase 1, decide survivors

```bash
python scripts/score_phase1.py \
    --probe-parquet outputs/phase1/probe_logits.parquet \
    --probe-csv outputs/probe/probe_sample.csv \
    --species-map ../Exp-14_Adaptive_Temporal_Fusion/pace_downloads/class_to_species_id.json \
    --aquatic-list ../Exp-14_Adaptive_Temporal_Fusion/outputs/signals/aquatic_blacklist.json \
    --out-json outputs/phase1/scores.json
# Prints survival decision per variant; if zero survivors -> close experiment.
```

### Step 3 (PACE, ~5h per surviving variant): full Phase 2 re-extraction

For each surviving variant V from Step 2:

```bash
PHOTO=unsharp  sbatch slurm/slurm_extract_vit_photo.sbatch
# (or PHOTO=clahe_lab, PHOTO=saturation)
# outputs to ${PLANTCLEF_ROOT}/aerten3/parquet/exp15_vit_lora_${PHOTO}/
```

### Step 4 (PACE CPU, ~30 min per variant): re-run multiscale_aggregate_knn

Use the same args as the Exp-9B `slurm_diff_beta_phase2a.sbatch` but
point `--in-dir` at the photometric extraction directory. Produces
`knn_bM0.85_bS1.00_T1.50_prefusion.parquet` for the photometric path.

### Step 5 (local CPU, seconds): β-sweep blend

```bash
python scripts/blend_vit_photo_tta.py \
    --orig-prefusion ../Exp-14_Adaptive_Temporal_Fusion/pace_downloads/knn_bM0.85_bS1.00_T1.50_prefusion.parquet \
    --photo-prefusion <local copy of the photometric prefusion parquet from PACE> \
    --out-dir outputs/stage1_vit_photo_tta/ \
    --betas "0.30,0.45,0.55,0.65,0.75,0.85,1.00" \
    --tag vit_photo_unsharp
```

### Step 6 (local CPU, ~1 min per blend): ensemble + habitat mask + proxies

For each β:

```bash
python ../Exp-14_Adaptive_Temporal_Fusion/scripts/ensemble_v2.py \
    --vit-prefusion-parquet outputs/stage1_vit_photo_tta/vit_photo_unsharp_beta0.65_prefusion.parquet \
    --convnext-scales <scale_4x4 5x5 6x6 paths> \
    --species-map ../Exp-14_Adaptive_Temporal_Fusion/pace_downloads/class_to_species_id.json \
    --class-mask-npy ../Exp-14_Adaptive_Temporal_Fusion/pace_downloads/sw_europe_mask_bool.npy \
    --out-dir outputs/stage1_vit_photo_tta/ensemble_beta0.65/ \
    --weights 0.77 --thresholds 0.100 --topk 10 \
    --tag exp15_unsharp_b065 \
    --vit-aggregator max --cnxt-aggregator max \
    --vit-aggregator-per-source '{"RNNB":"sim_w"}' \
    --cnxt-aggregator-per-source '{"RNNB":"sim_w"}' \
    --vit-controller ../Exp-14_Adaptive_Temporal_Fusion/stage_configs/controllers/vit_rnnb030_only.json \
    --cnxt-controller ../Exp-14_Adaptive_Temporal_Fusion/stage_configs/controllers/cnxt_baseline.json \
    --sim-json ../Exp-14_Adaptive_Temporal_Fusion/outputs/signals/visit_similarity.json

python ../Exp-14_Adaptive_Temporal_Fusion/scripts/apply_habitat_mask.py \
    --ensemble-parquet outputs/stage1_vit_photo_tta/ensemble_beta0.65/exp15_unsharp_b065_w0.77_ensemble.parquet \
    --species-map ../Exp-14_Adaptive_Temporal_Fusion/pace_downloads/class_to_species_id.json \
    --geo-mask    ../Exp-14_Adaptive_Temporal_Fusion/pace_downloads/sw_europe_mask_bool.npy \
    --habitat-mask ../Exp-14_Adaptive_Temporal_Fusion/outputs/signals/aquatic_blacklist_mask.npy \
    --out-csv outputs/stage1_vit_photo_tta/exp15_unsharp_b065_habitat.csv

python diagnostics/proxy_metrics_photo.py \
    --candidate-csv outputs/stage1_vit_photo_tta/exp15_unsharp_b065_habitat.csv \
    --winner-csv 'C:/Users/alper/Desktop/exp14_adaptive_fusion/v_winner_habitat_w0.77_thr0.100_top10.csv' \
    --density-parquet ../Exp-14_Adaptive_Temporal_Fusion/outputs/signals/density.parquet \
    --out-json outputs/stage1_vit_photo_tta/proxy_b065.json
```

### Step 7: LB submission

Pick the β with best proxy gates passing → submit `exp15_*_habitat.csv` to Kaggle.

## Anti-LB-overfit guardrails

1. Internal proxies primary; LB confirms only.
2. Coarse parameter precision: β ∈ {0.30, 0.45, 0.55, 0.65, 0.75, 0.85, 1.00}.
3. β = 1.00 graceful-fallback variant must reproduce 0.45572 winner ≤ 0.001.
4. Stop escalating when the spread of bracket variants is < ±0.003.

## Files reused (no modification)

- `Exp-9B_Improved_KNN_Retrieval/extract_embeddings_lora_test.py` — clone-and-modify for `--photometric` flag (this is what Phase 2 wraps).
- `Exp-9B_Improved_KNN_Retrieval/multiscale_aggregate_knn.py` — re-run on photometric tile parquets to produce blended kNN+aggregation parquet.
- `Exp-14_Adaptive_Temporal_Fusion/scripts/ensemble_v2.py` — feeds photometric ViT prefusion parquet via `--vit-prefusion-parquet`.
- `Exp-14_Adaptive_Temporal_Fusion/scripts/apply_habitat_mask.py` — final post-processing.
- `Exp-14_Adaptive_Temporal_Fusion/diagnostics/proxy_metrics.py` — reuse 4 Exp-14 proxies; new metric in proxy_metrics_photo.py.
- `Exp-14_Adaptive_Temporal_Fusion/outputs/signals/density.parquet` — Phase 1 cohort filter.

## Out of scope

- ViT replacement (not blend) of the original prefusion parquet.
- Per-source photometric (e.g., apply only to RNNB images).
- Geometric TTA (Exp-4 documented this hurts with max_prob).
- Multi-head taxonomic, SAM masking, EIVE.
- Tuning photometric parameters past defaults.
