# CLAUDE.md

## What This Is

Exp-3_Noisy_Student_Learning implements a multi-view teacher pipeline with augmentation consistency filtering for the PlantCLEF 2026 challenge. It builds on Exp-2's Spark/Luigi orchestration but introduces aspect-ratio-aware tiling, 3-view DINOv2 inference, and configurable tile filtering.

## Pipeline Architecture

```
Phase 1: Teacher Inference Pipeline
  1. create_manifest_shards.py  - Shard image paths for SLURM parallelization
  2. tiling.py                  - Aspect-ratio-aware tiling (4x3 + 4 random crops for LUCAS, 4x4 for test)
  3. multiview_embedder.py      - 3-view DINOv2 (orig/flip/rot90) + aug consistency check
  4. extract_embeddings.py      - Luigi task: tile + embed -> per-tile parquet
  5. aggregate_teacher.py       - Aug filter + confidence filter + Noisy-OR + geo mask

Phase 2: Teacher Benchmark
  1. aggregate_tiles.py         - Configurable re-aggregation engine
  2. generate_submission.py     - Aggregated probs -> Kaggle CSV (top-10 species)
  3. temperature_sweep.py       - Re-aggregate with T={1.0, 1.5, 2.0, 3.0}
  4. teacher_baseline_table.py  - 4 systematic config variations (T1-T4)
```

## Key Differences from Exp-2

- **Tiling**: Aspect-ratio-aware (4x3 grid + 4 random crops for LUCAS 4:3 images)
- **Multi-view**: 3 forward passes per tile (original, horizontal flip, 90-degree rotation)
- **Aug consistency**: Top-16 logit overlap between orig and flip views; tiles failing threshold are filtered
- **Tile confidence**: Tiles ranked by max softmax prob, top-K kept
- **Output**: Top-10 species (was top-8 in Exp-2)
- **Self-contained utils**: Copied from Exp-2, adapted paths for 77k LUCAS subset

## Data Paths on PACE

| Resource | Path |
|----------|------|
| LUCAS 77k CSV | `$PLANTCLEF_ROOT/data/raw/metadata_lucas/LUCAS_metadata_SWE_Wide_lat35_50_lon-10_10_77k.csv` |
| 212k images | `$PLANTCLEF_ROOT/data/raw/lucas_212k_images` |
| Test images | `$PLANTCLEF_ROOT/data/raw/PlanCLEF_test_data/images` |
| DINOv2 checkpoint | `$PLANTCLEF_ROOT/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth` |
| Parquet output | `$PLANTCLEF_ROOT/aerten3/parquet/lucas_sw77k/` |
| Manifests | `$PLANTCLEF_ROOT/manifests/` |

## PYTHONPATH

SLURM scripts set:
```bash
export PYTHONPATH=".../Exp-3_Noisy_Student_Learning:${PYTHONPATH}"
```

## Running on PACE

```bash
# Phase 1: Test set (run first, ~2 GPU-hrs)
python phase1/create_manifest_shards.py --mode test --image-dir ... --output-dir ...
sbatch phase1/slurm_extract_test.sbatch
sbatch phase1/slurm_aggregate_test.sbatch
python phase2/generate_submission.py --in-parquet ... --out-csv ...

# GATE: Submit to Kaggle. Teacher F1 >= 0.24?

# Phase 1: LUCAS 77k (~18 GPU-hrs)
python phase1/create_manifest_shards.py --mode lucas --metadata-csv ... --image-dir ... --output-dir ...
sbatch phase1/slurm_extract_lucas.sbatch
sbatch phase1/slurm_aggregate_lucas.sbatch

# Phase 2: Benchmarks
sbatch phase2/slurm_phase2_sweep.sbatch
sbatch phase2/slurm_phase2_baselines.sbatch
```

## Code Style

Python 3.10+. Uses ruff (line-length 88, double quotes, isort).
