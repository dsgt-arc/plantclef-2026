# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Exp-2-JPEG_Compression is a multi-stage plant species classification pipeline for the PlantCLEF 2026 challenge. It uses DINOv2 embeddings from LUCAS dataset images (212k) with a teacher-student knowledge distillation approach, run on Georgia Tech's PACE Phoenix cluster via SLURM.

## Pipeline Architecture

```
Step 1: Embedding Extraction
  Raw LUCAS images → 4x4 tile grid (16 tiles) → DINOv2 → 768-dim embeddings + 7806-class logits
  Orchestrated via Luigi tasks, parallelized across 213 SLURM array jobs

Step 2: Teacher Aggregation
  Per-tile logits → Temperature-Noisy-OR → per-image probabilities (top-64 sparse)
  Spark-based, processes each shard independently

Step 2_B / 2_C: Teacher Baselines & Co-graph
  2_B: Apply Step2 aggregation directly to test set (baseline submission)
  2_C: Ecological co-occurrence graph (PMI-based) to rerank/mask teacher predictions

Step 3: Student Training
  CrossRegionTransformer (512-dim, 4 layers, 8 heads) trained with:
    - Distillation loss from teacher probs (threshold 0.02)
    - Co-graph consistency loss
    - Self-training via pseudolabels (Step 3b)

Step 4: Test-Set Inference
  Extract test embeddings → student model inference → geographic mask → top-8 submission CSV
```

### Data Flow

Step1 outputs (`feature_bank/shard_*/`) → Step2 reads them → Step2 outputs (`step2_teacher_agg/shard_*/`) → Step3 reads both for training → Step4 extracts fresh test embeddings and runs trained student checkpoint.

## Key Conventions

- **Environment variable**: `PLANTCLEF_ROOT` points to the data root on PACE scratch
- **Sharding**: Datasets split into manifest files (`shard_NNNN.txt`), one image path per line. SLURM `$SLURM_ARRAY_TASK_ID` selects which shard to process
- **Storage**: Embeddings/logits stored as float16 bytes in parquet columns to halve storage
- **Sparse predictions**: Image-level probs kept as top-64 classes per image throughout
- **Species mapping**: Internal class indices 0–7805 (7806 classes), mapped to species IDs via a JSON/CSV mapping file at submission time

## Running on PACE

```bash
# Step 1: Extract embeddings (213 parallel GPU jobs)
sbatch step1/slurm-embed-lucas212k.sbatch

# Step 2: Aggregate teacher predictions (213 parallel CPU jobs)
sbatch step2/step2_aggregate_lucas212k.sbatch

# Step 3: Training scripts run directly with python/uv run
python step3/train_step3a_student_semisupervised.py [args]
python step3/train_step3b2_selftrain_refinement.py [args]

# Step 4: Extract test embeddings (4 parallel GPU jobs), then generate submission
sbatch step4_test_set_inference/slurm-embed-PlantCLEF_testset.sbatch
python step4_test_set_inference/step4_generate_test_submit.py [args]
```

SLURM jobs use: RTX6000 GPU, 6 CPUs, 64GB RAM. Set `PYTORCH_ALLOC_CONF=expandable_segments:True` to reduce CUDA fragmentation.

## Utils Module

| File | Purpose |
|------|---------|
| `spark.py` | `get_spark()` / `spark_resource()` — create and manage local Spark sessions |
| `io_paths.py` | `plantclef_root()` and path helpers resolving from `PLANTCLEF_ROOT` env var |
| `model_setup.py` | `setup_fine_tuned_model()` — locate DINOv2 checkpoint on PACE |
| `serde.py` | Image and mask serialization (bytes ↔ PIL/numpy) |
| `workflow.py` | `ProcessTransform` Luigi base task, `make_image_tiles_udf()` for 4x4 tiling |
| `embedding/transform.py` | `WrappedFineTunedDINOv2` — Spark ML transformer wrapping DINOv2 inference |
| `detection/` | Detection-related classes, params, and transforms |

## Code Style

Python 3.10+. Project uses ruff (line-length 88, double quotes, isort). Pre-commit runs `ruff check --fix` and `ruff format` automatically.
