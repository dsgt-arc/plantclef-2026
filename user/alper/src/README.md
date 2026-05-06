# PlantCLEF 2026 — README SECTION

End-to-end inference pipeline for the **F1 = 0.50623** PlantCLEF 2026 submission (Exp-16C lock, 2026-05-05).

For a plain-English description of *what* each technique does and *why*, see [`Working_Techniques_Summary.md`](Working_Techniques_Summary.md). This README covers *how to run the code*.

---

## 1. Results

| Milestone | Mechanism added | LB F1 | Δ |
|---|---|---:|---:|
| Exp-3 baseline | noisy_or aggregator, T=2.0 | 0.28093 | — |
| Exp-4 | `max_prob` + JPEG q=85 + temporal `prior_blend` | 0.35698 | +0.07605 |
| Exp-7/8 | multiscale tiling 4+5+6 (77 tiles) | 0.40100 | +0.04402 |
| Exp-9 | ViT FAISS kNN blend (β_multi=0.80, β_single=1.00) | 0.41754 | +0.01654 |
| Exp-11 | ConvNeXt-L DINOv3 + ViT probability ensemble (w=0.77) | 0.43790 | +0.02036 |
| Exp-14 | per-source `sim_w` on RNNB + α=0.30 | 0.45410 | +0.01620 |
| Exp-14 Stage 12 | aquatic-species habitat mask | 0.45572 | +0.00162 |
| Exp-14 Stage 13 | per-source biogeographic mask on alpine | 0.45611 | +0.00039 |
| Exp-16 N3_r0.5 | N-floor admission + ratio gate | 0.46168 | +0.00557 |
| Exp-16 | habitat-fit demotion (R=200 km, β=0.25) | 0.49842 | +0.03674 |
| **Exp-16C lock** | **ViT-L+ConvNeXt cooldown backbones; per-source β override on alpine (β=0.45); habitat-fit retune (n_near_min=40, β_global=0.15)** | **0.50623** | **+0.00781** |

Final submission CSV: `${PLANTCLEF_ROOT}/aerten3/submissions/exp16c/exp16c_s12B_alpine_b0.45.csv`

---

## 2. Environment and prerequisites

- **Cluster**: Georgia Tech PACE.
- **SLURM account**: `paceship-clef2026_plantclef`.
- **GPU partitions**: `gpu-rtx6000` (default) or `gpu-h200` (training); QoS `inferno`.
- **CPU partitions**: `cpu-medium`; QoS `embers`.
- **Email for job notifications**: `aerten3@gatech.edu`.
- **Root path** (set before running anything):
  ```bash
  export PLANTCLEF_ROOT=/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0
  ```
- **Python deps**: `torch`, `timm`, `faiss-cpu`, `pandas`, `pyarrow`, `numpy`, `Pillow`, `pyyaml`. The PACE conda environment used during the run is pinned in `Exp-10_ConvNext_finetuning/configs/`.
- **Data assumed present**:
  - Test images: `${PLANTCLEF_ROOT}/data/raw/PlanCLEF_test_data/images/`
  - Training images (1.4 M, used for FAISS index + habitat-fit lat/lon): `${PLANTCLEF_ROOT}/data/raw/train_full/`
  - Species map: `${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json`
  - SW-Europe geo mask: `${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy`
  - PlantCLEF2024 training metadata for habitat-fit: `${PLANTCLEF_ROOT}/data/raw/PlantCLEF2024singleplanttrainingdata.csv`

---

## 3. Repository layout

| Folder | What it does | Status |
|---|---|---|
| `Exp-1_steps_with_lucas_212k_set/` | Original cross-region transformer + noisy-student attempt on the 212K LUCAS set | Abandoned (domain gap) |
| `Exp-2-JPEG_Compression/` | JPEG round-trip preprocessing | In pipeline |
| `Exp-4_Enhanced_Tile_Aggregation/` | `MaxProbabilityAggregator`, `prior_blend` temporal fusion | In pipeline |
| `Exp-7_Multiscale_tiling/` | Multi-scale tile extraction (ViT) at scales 4+5+6 | In pipeline |
| `Exp-9_kNN_Retrieval_Ensemble/` | FAISS index build + per-tile kNN blend (ViT) | In pipeline |
| `Exp-10_ConvNext_finetuning/` | ConvNeXt-Large DINOv3 2-phase finetune + tile extraction | In pipeline |
| `Exp-10C_ConvNext_Cooldown/` | 10-epoch cooldown of ConvNeXt-L on combined train+val+test | In pipeline (Exp-16C) |
| `Exp-11_Conv_ViT_Ensemble/` | ConvNeXt + ViT probability-level ensemble ("Approach A") | In pipeline |
| `Exp-13_ViT_finetuning/` | ViT-L DINOv2 2-phase finetune + Phase 2v3 cooldown | In pipeline (Exp-16C backbone) |
| `Exp-14_Adaptive_Temporal_Fusion/` | Per-source `sim_w` aggregator + aquatic + biogeographic masks | In pipeline |
| `Exp-16_Multi_Species_Discovery/` | N-floor admission + habitat-fit demotion | In pipeline |
| `Exp-16C_HabitatFit_Retune/` | Habitat-fit re-tune at the new ViT-L+ConvNeXt-cooldown backbones; per-source β override | **Final lock** |
| `Final_pipeline_methods_Summary/` | Detailed methods writeup (frozen at F1=0.4379 — superseded by Exp-16C) | Reference |

---

## 4. End-to-end run (final submission)

The minimal command sequence to reproduce **F1 = 0.50623** from raw test images and pre-trained checkpoints. Each line is a SLURM submission; later stages depend on earlier ones.

### 4.1 Train backbones (skip if checkpoints already exist)

```bash
# ConvNeXt-Large DINOv3, 2-phase finetune (4×H200, ~3 days total)
sbatch --export=VARIANT=large Exp-10_ConvNext_finetuning/slurm/phase1_train.sbatch
sbatch --export=VARIANT=large,GPU_TYPE=h200,NGPUS=4 Exp-10_ConvNext_finetuning/slurm/phase2_train.sbatch

# ViT-L DINOv2, 2-phase + Phase 2v3 cooldown
sbatch Exp-13_ViT_finetuning/slurm/phase1_train.sbatch
sbatch Exp-13_ViT_finetuning/slurm/phase2_train.sbatch
sbatch Exp-13_ViT_finetuning/slurm/phase2v3_cooldown.sbatch

# ConvNeXt cooldown (Exp-10C) — same recipe applied to ConvNeXt
sbatch Exp-10C_ConvNext_Cooldown/slurm/cooldown.sbatch
```

### 4.2 Extract per-tile embeddings + logits at scales 4+5+6

```bash
# ViT-L tile extraction (3 scales × ~1.5 h on rtx6000)
for SCALE in 4 5 6; do
  sbatch --export=SCALE=$SCALE Exp-13_ViT_finetuning/slurm/extract_phase2v3_rtx6000.sbatch
done

# ConvNeXt-L cooldown tile extraction
sbatch Exp-10C_ConvNext_Cooldown/slurm/extract_4x4.sbatch
sbatch Exp-10C_ConvNext_Cooldown/slurm/extract_5x5_6x6.sbatch
```

Both extractors apply JPEG round-trip (q=85, 4:2:2) before forward pass and write parquet at `${PLANTCLEF_ROOT}/aerten3/parquet/.../scale_{4x4,5x5,6x6}.parquet`.

### 4.3 Build FAISS kNN index (ViT only, one-time)

```bash
sbatch Exp-9_kNN_Retrieval_Ensemble/slurm_build_index.sbatch
# Produces: faiss_index_geo.bin, faiss_labels_geo.npy, class_counts_geo.npy
```

### 4.4 ViT branch: kNN blend + temporal fusion

```bash
# Per-tile kNN blend (β_multi=0.80, β_single=1.00, k=20, τ=0.07)
# then sim_w aggregator on RNNB + max elsewhere, α_RNNB=0.30, α_default=0.15
sbatch Exp-14_Adaptive_Temporal_Fusion/slurm/run_stage4_simw_BOTH_vitRNNB30.sbatch
```

### 4.5 ConvNeXt branch: aggregation + temporal fusion

```bash
# max_prob over 77 tiles, T=1.5, image_topk=64, geo mask
# then sim_w on RNNB + max elsewhere, α=0.30
sbatch Exp-14_Adaptive_Temporal_Fusion/slurm/run_convnext_branch.sbatch
```

### 4.6 Probability-level ensemble + ecological masks

```bash
# w = 0.77 ViT+kNN + 0.23 ConvNeXt
# applies aquatic mask (Stage 12) and per-source biogeographic mask (Stage 13)
sbatch Exp-14_Adaptive_Temporal_Fusion/slurm/run_stage13_biogeo.sbatch
```

### 4.7 Habitat-fit demotion + per-source admission + per-source β (Exp-16C)

```bash
# Build per-species geographic profile (one-time, ~30 min CPU)
python Exp-16_Multi_Species_Discovery/diagnostics/build_species_geo_profile.py
python Exp-16_Multi_Species_Discovery/diagnostics/refine_source_centroids.py
python Exp-16_Multi_Species_Discovery/diagnostics/add_radius_columns.py

# Final lock: alpine β=0.45 override on top of global β=0.15
sbatch Exp-16C_HabitatFit_Retune/slurm/stage12B_alpine_beta_refine.sbatch
```

The final stage is implemented by `Exp-16_Multi_Species_Discovery/scripts/apply_habitat_fit.py`, invoked as:

```bash
python Exp-16_Multi_Species_Discovery/scripts/apply_habitat_fit.py \
  --ensemble-parquet  ${PLANTCLEF_ROOT}/aerten3/exp16c/ensemble_phase2v3_cnxt_cooldown.parquet \
  --refined-ref-json  ${PLANTCLEF_ROOT}/aerten3/exp16/outputs/habitat_classifier/refined_source_ref.json \
  --species-geo-csv   ${PLANTCLEF_ROOT}/aerten3/exp16/outputs/habitat_classifier/species_geo_profile.csv \
  --geo-mask          ${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy \
  --aquatic-mask      ${PLANTCLEF_ROOT}/aerten3/exp14/outputs/signals/aquatic_mask.npy \
  --biogeo-json       ${PLANTCLEF_ROOT}/aerten3/exp14/outputs/signals/biogeo_per_source.npz \
  --radius-km 200 --beta 0.15 --n-near-min 40 --use-altitude \
  --per-source-json '{"CBN-PdlC":{"beta":0.45},"CBN-Pla":{"beta":0.45}}' \
  --n-floor 3 --ratio-floor 0.5 \
  --count-bound-sources GUARDEN,LISAH,OPTMix,RNNB \
  --threshold 0.085 --topk 10 \
  --out-csv ${PLANTCLEF_ROOT}/aerten3/submissions/exp16c/exp16c_s12B_alpine_b0.45.csv
```

This is the single command that produces the **F1 = 0.50623** submission CSV.

---

## 5. Final hyperparameter manifest

```yaml
preprocessing:
  jpeg_quality: 85
  jpeg_subsampling: 1   # 4:2:2
  resize: lanczos

tiling:
  scales: [4, 5, 6]
  tiles_per_image: 77
  tile_size: 518

vit_branch:
  backbone: vit_large_patch14_reg4_dinov2.lvd142m  # Exp-13 phase2v3 cooldown checkpoint
  temperature: 1.5
  aggregator: max_prob
  image_topk: 64
  knn:
    index_type: IndexFlatIP
    k: 20
    tau: 0.07
    beta_multi: 0.80
    beta_single: 1.00
    blend_point: per_tile_pre_aggregation
  temporal_fusion:
    aggregator_per_source: {RNNB: sim_w, default: max}
    alpha_per_source: {RNNB: 0.30, default: 0.15}

convnext_branch:
  backbone: convnext_large.dinov3_lvd1689m  # Exp-10C cooldown checkpoint
  temperature: 1.5
  aggregator: max_prob
  image_topk: 64
  temporal_fusion:
    aggregator_per_source: {RNNB: sim_w, default: max}
    alpha: 0.30

ensemble:
  weights: {vit_plus_knn: 0.77, convnext: 0.23}

masks:
  geo: sw_europe_mask_bool.npy
  aquatic: 91 species (genus + epithet pattern match)
  biogeo_alpine: 5 species zeroed on CBN-PdlC, CBN-Pla

habitat_fit:
  beta_global: 0.15
  per_source_beta:
    CBN-PdlC: 0.45
    CBN-Pla:  0.45
  radius_km: 200
  n_near_min: 40
  use_altitude: true

admission:
  n_floor: 3
  ratio_floor: 0.5
  count_bound_sources: [GUARDEN, LISAH, OPTMix, RNNB]

submission:
  threshold: 0.085
  topk: 10
```

---

## 6. Key file index

| Artefact | Path |
|---|---|
| Final submission CSV | `${PLANTCLEF_ROOT}/aerten3/submissions/exp16c/exp16c_s12B_alpine_b0.45.csv` |
| ViT-L Phase 2v3 cooldown checkpoint | `${PLANTCLEF_ROOT}/aerten3/checkpoints/exp13_vitl/phase2v3_cooldown/model_best.pth.tar` |
| ConvNeXt-L cooldown checkpoint | `${PLANTCLEF_ROOT}/aerten3/checkpoints/exp10c_convnext_cooldown/model_best.pth.tar` |
| FAISS index + labels | `${PLANTCLEF_ROOT}/aerten3/exp9_knn/faiss_index/{faiss_index_geo.bin, faiss_labels_geo.npy}` |
| Geo mask (SW Europe) | `${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy` |
| Aquatic mask | `${PLANTCLEF_ROOT}/aerten3/exp14/outputs/signals/aquatic_mask.npy` |
| Per-source biogeographic mask | `${PLANTCLEF_ROOT}/aerten3/exp14/outputs/signals/biogeo_per_source.npz` |
| Refined source centroids | `${PLANTCLEF_ROOT}/aerten3/exp16/outputs/habitat_classifier/refined_source_ref.json` |
| Species geo profile | `${PLANTCLEF_ROOT}/aerten3/exp16/outputs/habitat_classifier/species_geo_profile.csv` |
| ViT tile parquets | `${PLANTCLEF_ROOT}/aerten3/parquet/plantclef_2026_test/exp13_phase2v3/scale_{4x4,5x5,6x6}.parquet` |
| ConvNeXt tile parquets | `${PLANTCLEF_ROOT}/aerten3/parquet/exp10c_convnext_cooldown/scale_{4x4,5x5,6x6}.parquet` |

---

