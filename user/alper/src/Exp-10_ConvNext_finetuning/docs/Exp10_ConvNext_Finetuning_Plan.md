# Exp-10: DINOv3 ConvNeXt Fine-tuning for PlantCLEF 2026

## Context

The PlantCLEF 2026 pipeline currently uses a DINOv2 ViT-B/14 backbone (F1=0.41754). Experiment 10 introduces DINOv3 ConvNeXt models as an alternative/complementary backbone. These models (distilled from a ViT-7B teacher) bring architecture diversity (convolutions vs attention) that should improve ensemble performance. We need to:

1. Fine-tune DINOv3 ConvNeXt on the same 1.4M PlantCLEF training images (7806 species)
2. Create inference scripts to extract embeddings/logits in the same parquet format
3. Integrate into the existing aggregation + kNN + temporal fusion pipeline

```
TRAINING FLOW (per variant, all 4 run in parallel via SLURM):
                                                    
  DINOv3 ConvNeXt pretrained    PlantCLEF 2024 ImageFolder
  (timm registry, num_classes=0)    (1.4M images, 7806 species)
         |                              |
         v                              v
  +----------------------------------------------+
  | Phase 1: Freeze backbone, train head         |
  | (5 epochs, lr=1e-3, multi-GPU DDP on H200s)  |
  +----------------------------------------------+
         |
         v  (model_best.pth.tar)
  +----------------------------------------------+
  | Phase 2: Full fine-tune with layer-decay     |
  | (100 epochs, lr=8e-5, multi-GPU DDP on H200s)|
  +----------------------------------------------+
         |
         v  (model_best.pth.tar)
  +----------------------------------------------+
  | Export checkpoint for inference              |
  +----------------------------------------------+

PARALLELIZATION STRATEGY:

  SLURM Job 1: tiny   (2-4x H200) --+
  SLURM Job 2: small  (2-4x H200) --+--> All 4 variants train simultaneously
  SLURM Job 3: base   (2-4x H200) --+    Each uses multi-GPU DDP via torchrun
  SLURM Job 4: large  (2-4x H200) --+

INFERENCE FLOW (post-training):

  test image (3000x3000)
    -> multi-scale tiling (5x5+9x9+13x13 @ 224px = 275 tiles)
    -> ConvNeXt forward pass per tile
       outputs: embedding (768/1024/1536-dim) + logits (7806)
    -> same parquet schema as ViT pipeline
    -> reuse Exp-8/9 aggregation + kNN + temporal fusion
    -> ensemble with ViT predictions at probability level
```

---

## Key Technical Facts

| Property | ViT (current) | ConvNeXt (new) |
|----------|--------------|----------------|
| timm model | `vit_base_patch14_reg4_dinov2.lvd142m` | `convnext_{tiny,small,base,large}.dinov3_lvd1689m` |
| Input size | 518x518 | **224x224** |
| Embedding | CLS token, `features[:, 0, :]` -> `[B, 768]` | **Global avg pool, `forward_head(feats, pre_logits=True)` -> `[B, D]`** |
| `forward_features()` output | `[B, seq_len, 768]` | **`[B, D, H, W]` (spatial, NOT pooled)** |
| Embed dim | 768 | 768 (tiny/small), 1024 (base), 1536 (large) |
| Pretrained num_classes | 7806 (finetuned) | **0 (feature extractor, head added by timm)** |

### ConvNeXt Variants — H200 HGX (141GB HBM3e)

| Variant | Params | Embed Dim | Per-GPU Batch (H200) | Grad Checkpoint |
|---------|--------|-----------|----------------------|-----------------|
| tiny | 27.8M | 768 | 512 | No |
| small | 49.5M | 768 | 384 | No |
| base | 87.6M | 1024 | 256 | No |
| large | 196.2M | 1536 | 128 | No |

> With 141GB VRAM, no variant needs gradient checkpointing. All fit comfortably.

### Multi-GPU DDP Training

timm's `train.py` natively supports PyTorch DDP via `torchrun`:
```bash
torchrun --nproc_per_node=NUM_GPUS train_convnext.py --batch-size BATCH_PER_GPU ...
```
- Effective batch = `batch_per_gpu * num_gpus`
- Linear speedup with near-perfect scaling for data-parallel training
- timm auto-adjusts LR scaling: `lr = lr_base * sqrt(global_batch / 256)` for Adam-like optimizers

### Data Path on PACE

```
/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/train/PlantCLEF2024/
  train/{species_id}/*.jpg   (training images)
  val/{species_id}/*.jpg     (validation images)
  test/{species_id}/*.jpg    (test images with labels)
```

---

## GPU & SLURM Configuration

**Target GPU:** NVIDIA H200 HGX (141GB HBM3e) on PACE inferno

| Setting | Value |
|---------|-------|
| Partition | `gpu-h200` |
| GRES | `gpu:h200:4` (4 GPUs per job; max 8 per node) |
| QoS | `inferno` |
| Account | `paceship-dsgt_clef2026` |
| Nodes | 12 available (96 H200s total) |
| CPUs per node | 64 |
| Memory per node | ~2TB |
| Max wall time | 3 days |

```bash
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:4
#SBATCH --qos=inferno
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --cpus-per-task=32    # 8 workers/GPU * 4 GPUs
#SBATCH --mem=256G
```

## Training Priority

**Start with ConvNeXt-Base first** — most comparable to the existing ViT-Base:
- ViT-Base: ~86M params, 768-dim -> ConvNeXt-Base: 87.6M params, 1024-dim
- Compare `summary.csv` directly against ViT results (target: 76% top-1)
- Once Base is validated, submit tiny/small/large in parallel

---

## File Structure

```
Exp-10_ConvNext_finetuning/
+-- train_convnext.py                    # timm train.py + freeze-backbone support
+-- configs/
|   +-- phase1_tiny.yaml                 # Head-only training configs
|   +-- phase1_small.yaml
|   +-- phase1_base.yaml
|   +-- phase1_large.yaml
|   +-- phase2_tiny.yaml                 # Full fine-tune configs
|   +-- phase2_small.yaml
|   +-- phase2_base.yaml
|   +-- phase2_large.yaml
+-- slurm/
|   +-- phase1_train.sbatch              # SLURM for Phase 1 (multi-GPU DDP)
|   +-- phase2_train.sbatch              # SLURM for Phase 2 (multi-GPU DDP)
|   +-- run_training.sh                  # Submit Phase 1 -> Phase 2 chain
|   +-- run_all_variants.sh              # Submit all 4 variants in parallel
|   +-- extract_embeddings.sbatch        # SLURM for inference embedding extraction
+-- extract_convnext_embeddings.py       # Inference: tile + forward pass -> parquet
+-- scripts/
|   +-- verify_model_loading.py          # Smoke test: load all 4 variants, check shapes
|   +-- export_checkpoint.py             # Convert timm checkpoint -> inference .pth
```

---

## Implementation Steps

### Step 1: Download and Modify timm's train.py

**Source:** `https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/train.py`

**Modifications (3 additions, ~20 lines total):**

1. **Add CLI arguments** (in argparse, after "Model parameters" group):
   - `--freeze-backbone` (store_true): freeze all params except classifier head
   - `--freeze-backbone-prefix` (default `'head'`): param name prefix for trainable params

2. **Add freeze logic** (after `model = create_model(...)`, before `if args.grad_checkpointing:`):
   ```python
   if args.freeze_backbone:
       for name, param in model.named_parameters():
           if not name.startswith(args.freeze_backbone_prefix):
               param.requires_grad = False
       trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
       total = sum(p.numel() for p in model.parameters())
       _logger.info(f'Backbone frozen: {total - trainable:,} frozen, {trainable:,} trainable')
   ```

3. **Ensure `--class-map` works** with timm's pretrained_cfg_overlay for our class_mapping.txt.

**Why this works:** timm's `create_optimizer_v2` automatically skips params with `requires_grad=False` (confirmed in `_param_groups.py`). No optimizer changes needed.

### Step 2: Create YAML Config Files (8 files)

**Phase 1 template** (classifier-only, `configs/phase1_{variant}.yaml`):
```yaml
model: convnext_{variant}.dinov3_lvd1689m
pretrained: true
num-classes: 7806
freeze-backbone: true
input-size: [3, 224, 224]
epochs: 5
batch-size: {see table}  # PER-GPU batch size
opt: adamw
lr: 1.0e-3
weight-decay: 0.0
sched: cosine
warmup-lr: 1.0e-5
warmup-epochs: 1
min-lr: 0
smoothing: 0.1
drop-path: 0.0           # No drop path when backbone frozen
mixup: 0.0               # No mixup for Phase 1
cutmix: 0.0
amp: true
amp-dtype: float16
channels-last: true
model-ema: true
model-ema-decay: 0.9998
workers: 8
pin-mem: true
seed: 42
checkpoint-hist: 3
log-interval: 100
eval-metric: top1
```

**Phase 2 template** (full fine-tune, `configs/phase2_{variant}.yaml`):
```yaml
model: convnext_{variant}.dinov3_lvd1689m
num-classes: 7806
# pretrained: false -- weights come via --initial-checkpoint from Phase 1
input-size: [3, 224, 224]
epochs: 100
batch-size: {see table}  # PER-GPU batch size
opt: adamw
lr: 8.0e-5
weight-decay: 0.05       # ConvNeXt standard (vs ViT's 0.0)
layer-decay: {see table}
sched: cosine
warmup-lr: 1.0e-6
warmup-epochs: 5
min-lr: 1.0e-7
smoothing: 0.1
drop-path: {see table}
aa: rand-m9-mstd0.5-inc1
mixup: 0.8
cutmix: 1.0
amp: true
amp-dtype: float16
channels-last: true
model-ema: true
model-ema-decay: 0.9998
workers: 8
pin-mem: true
seed: 42
checkpoint-hist: 5
log-interval: 100
eval-metric: top1
patience-epochs: 10
```

**Variant-specific values (H200, per-GPU batch sizes):**

| Param | tiny | small | base | large |
|-------|------|-------|------|-------|
| P1 batch-size (per GPU) | 512 | 384 | 256 | 128 |
| P2 batch-size (per GPU) | 256 | 192 | 128 | 64 |
| P2 lr | 8e-5 | 8e-5 | 5e-5 | 3e-5 |
| P2 drop-path | 0.1 | 0.15 | 0.2 | 0.3 |
| P2 layer-decay | 0.7 | 0.7 | 0.75 | 0.8 |

> Note: With multi-GPU DDP, effective batch = per_gpu_batch * num_gpus.
> timm auto-scales LR with `--lr-base-scale sqrt` (default for Adam).
> If using 4x H200, effective batch for tiny = 1024, so LR auto-scales up.

### Step 3: Create SLURM Scripts (Multi-GPU DDP)

**`slurm/phase1_train.sbatch`:**
```bash
#!/bin/bash
#SBATCH --job-name=cnxt-p1-${VARIANT}
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32               # 8 workers/GPU * 4 GPUs
#SBATCH --mem=256G
#SBATCH --time=04:00:00                  # ~1hr for Phase 1 with 4x H200
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

VARIANT="${VARIANT:-base}"               # Start with base (most comparable to ViT)
NGPUS=4

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-10_ConvNext_finetuning"
DATA_DIR="${PLANTCLEF_ROOT}/train/PlantCLEF2024"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp10_convnext"

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=${NGPUS} "${EXP_DIR}/train_convnext.py" \
    -c "${EXP_DIR}/configs/phase1_${VARIANT}.yaml" \
    --data-dir "${DATA_DIR}" \
    --train-split train \
    --val-split val \
    --output "${OUTPUT_DIR}" \
    --experiment "convnext_${VARIANT}_phase1"
```

**`slurm/phase2_train.sbatch`:**
```bash
#!/bin/bash
#SBATCH --job-name=cnxt-p2-${VARIANT}
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00                  # ~6-10hrs with 4x H200 for 100 epochs
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=aerten3@gatech.edu

VARIANT="${VARIANT:-base}"
NGPUS=4

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP_DIR="${REPO_ROOT}/user/alper/src/Exp-10_ConvNext_finetuning"
DATA_DIR="${PLANTCLEF_ROOT}/train/PlantCLEF2024"
OUTPUT_DIR="${PLANTCLEF_ROOT}/aerten3/checkpoints/exp10_convnext"
PHASE1_BEST="${OUTPUT_DIR}/convnext_${VARIANT}_phase1/model_best.pth.tar"

source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"
export PYTORCH_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=${NGPUS} "${EXP_DIR}/train_convnext.py" \
    -c "${EXP_DIR}/configs/phase2_${VARIANT}.yaml" \
    --data-dir "${DATA_DIR}" \
    --train-split train \
    --val-split val \
    --initial-checkpoint "${PHASE1_BEST}" \
    --output "${OUTPUT_DIR}" \
    --experiment "convnext_${VARIANT}_phase2"
```

**`slurm/run_training.sh`** -- Submit Phase 1 -> Phase 2 chain for one variant:
```bash
VARIANT="${1:-base}"
P1_JOB=$(sbatch --export=VARIANT=${VARIANT} slurm/phase1_train.sbatch | awk '{print $4}')
echo "Phase 1 job: ${P1_JOB}"
sbatch --export=VARIANT=${VARIANT} --dependency=afterok:${P1_JOB} slurm/phase2_train.sbatch
```

**`slurm/run_all_variants.sh`** -- Submit all 4 variants in parallel:
```bash
for VARIANT in base tiny small large; do
    bash slurm/run_training.sh ${VARIANT}
done
```
This submits 8 SLURM jobs total (4 Phase 1 + 4 Phase 2 with dependencies).
All Phase 1 jobs run simultaneously. Each Phase 2 starts after its Phase 1 completes.

### Step 4: Create Embedding Extraction Script

**`extract_convnext_embeddings.py`** - Adapted from `Exp-7/extract_multiscale_embeddings.py`

**Key differences from ViT version:**

1. **Tile size**: `TILE_SIZE = 224` (was `DINOV2_SIZE = 518`)

2. **Embedding extraction** (critical change):
   ```python
   # ViT (current, line 127-130 of Exp-7 script):
   features = model.forward_features(tensors)     # [B, seq_len, 768]
   cls_tokens = features[:, 0, :]                  # [B, 768]
   logits = model.head(cls_tokens)                  # [B, 7806]

   # ConvNeXt (new):
   features = model.forward_features(tensors)       # [B, D, H, W] spatial
   embeddings = model.forward_head(features, pre_logits=True)  # [B, D] pooled
   logits = model.forward_head(features)             # [B, 7806]
   ```

3. **Dynamic embed dim**: Read from model instead of hardcoding 768
   ```python
   EMB_DIM = model.num_features  # 768/1024/1536 depending on variant
   ```

4. **Model loading**: Add `--model-name` default to ConvNeXt, accept `--checkpoint-path`

5. **Output schema**: Identical to Exp-7 (image_name, tile_index, embedding_orig_f16, teacher_logits_f16, max_prob, passed_aug_consistency)

6. **Tiling scales**: For 224px tiles on 3000x3000 images, recommended scales:
   - 5x5 (25 tiles) ~ ViT 2x2
   - 9x9 (81 tiles) ~ ViT 4x4
   - 13x13 (169 tiles) ~ ViT 6x6

### Step 5: Create Utility Scripts

**`scripts/verify_model_loading.py`:**
- Load all 4 ConvNeXt variants with `num_classes=7806`
- Verify `forward_features` -> `forward_head(pre_logits=True)` -> `forward_head()` shapes
- Print parameter counts, classifier head structure
- Run a dummy forward pass at 224x224

**`scripts/export_checkpoint.py`:**
- Load timm checkpoint (model_best.pth.tar)
- Extract `state_dict_ema` (if EMA enabled) or `state_dict`
- Save as standalone `.pth` file for inference loading

### Step 6: Pipeline Integration (future, post-training)

**Downstream scripts that work WITHOUT changes** (they consume parquet, not model internals):
- `Exp-8/aggregate_tiles.py` - reads `teacher_logits_f16` (always 7806-dim)
- `Exp-8/generate_submission.py` - reads aggregated probabilities
- `Exp-3/phase2/temporal_fusion.py` - blends probabilities

**Downstream scripts needing MINOR changes:**
- `Exp-9/multiscale_aggregate_knn.py` line ~149: hardcoded `768` -> dynamic `emb_dim` from FAISS index
- `Exp-9/build_faiss_index.py`: embedding dimension parameter (768 -> model-specific)
- Scale arguments: add support for `--scale9-dir`, `--scale13-dir` etc.

**New script needed:**
- `ensemble_probabilities.py`: weighted average of ViT + ConvNeXt probability vectors

---

## Estimated Training Times (4x H200 HGX per variant)

| Variant | Phase 1 (5 ep) | Phase 2 (100 ep) | Total | Notes |
|---------|---------------|-----------------|-------|-------|
| tiny | ~20 min | ~4 hrs | ~4.5 hrs | Fastest, good for validation |
| small | ~25 min | ~5 hrs | ~5.5 hrs | |
| base | ~35 min | ~7 hrs | ~8 hrs | Start here (most comparable to ViT) |
| large | ~50 min | ~10 hrs | ~11 hrs | Largest, best accuracy expected |

> All 4 variants in parallel = ~11 hrs wall-clock total (limited by large variant).
> Compared to single RTX 6000: ~5-8x speedup from H200 + multi-GPU DDP.

---

## Verification Plan

1. **Smoke test** (local, before PACE):
   ```bash
   python scripts/verify_model_loading.py
   # Should print shapes for all 4 variants and pass dummy forward
   ```

2. **Train.py arg parsing** (local):
   ```bash
   python train_convnext.py --help | grep freeze
   # Should show --freeze-backbone and --freeze-backbone-prefix
   ```

3. **Phase 1 validation** (PACE, ~35 min for base):
   - Submit `VARIANT=base sbatch slurm/phase1_train.sbatch`
   - Check `summary.csv` shows eval_top1 > 50% after 5 epochs

4. **Phase 2 validation** (PACE, ~7 hrs for base):
   - Submit Phase 2 with `--initial-checkpoint` from Phase 1 best
   - Target eval_top1 > 70% by epoch 50
   - Compare final summary.csv against ViT (76% top-1)

5. **Embedding extraction test** (PACE):
   ```bash
   python extract_convnext_embeddings.py \
     --scale 5 --model-path /path/to/best.pth \
     --image-dir /path/to/test --output-dir /tmp/test
   # Verify parquet schema matches Exp-7 output
   ```

6. **End-to-end pipeline test** (PACE):
   - Extract ConvNeXt embeddings for test set
   - Run `aggregate_tiles.py` on ConvNeXt parquets
   - Generate submission, compare F1 with ViT baseline (0.40+)

---

## Critical Files to Reference During Implementation

| File | Purpose |
|------|---------|
| `Exp-7/extract_multiscale_embeddings.py` | Template for ConvNeXt extraction script |
| `Exp-9/multiscale_aggregate_knn.py` | Needs emb_dim fix for ConvNeXt integration |
| `Exp-9/build_faiss_index.py` | Template for ConvNeXt FAISS index |
| ViT `args.yaml` (local) | Reference training config to match |
| ViT `summary.csv` (local) | Benchmark results to compare against |
| timm `train.py` (GitHub) | Base script to download and modify |
