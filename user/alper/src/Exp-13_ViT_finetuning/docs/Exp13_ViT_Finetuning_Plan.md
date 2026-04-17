# Exp-13: ViT-Large DINOv2 + DINOv3 Finetuning Plan

## Context

PlantCLEF 2026's current best (F1=0.40100) is an ensemble of the teammate's DINOv2 ViT-Base (F1=0.41754 standalone) + kNN retrieval. Exp-10 adds a ConvNext-Large member. **Exp-13 adds two ViT-Large members** so the final ensemble has architectural diversity across ConvNext, DINOv2 ViT-L, and DINOv3 ViT-L:

- `vit_large_patch14_reg4_dinov2.lvd142m` — ViT-L DINOv2 with 4 register tokens
- `vit_large_patch16_dinov3.lvd1689m` — ViT-L DINOv3 (newer, larger pretraining corpus)

Training follows the two-stage PlantNet reference protocol:

1. **Phase 1** — freeze backbone, train classifier head only (cheap, stable).
2. **Phase 2** — unfreeze everything, finetune end-to-end with layer-wise LR decay (where most F1 gain comes from).

Expected outcome: 2 new ensemble members with F1 ~ 0.43–0.45 standalone (ViT-L typically beats ViT-B by 1–3 F1 pts on this task), boosting the ensemble materially when merged.

Reference configs consulted:

- PlantNet ViT-B P1: `DinoViT_pretrained_models/.../onlyclassifier/args.yaml`
- PlantNet ViT-B P2: `DinoViT_pretrained_models/.../onlyclassifier_then_all/args.yaml`
- Exp-10 ConvNext infrastructure: `user/alper/src/Exp-10_ConvNext_finetuning/`
- Exp-7 ViT extraction pipeline: `user/alper/src/Exp-7_Multiscale_tiling/extract_multiscale_embeddings.py`

---

## Model Comparison

| Attribute | DINOv2 ViT-L reg4 | DINOv3 ViT-L |
|---|---|---|
| timm name | `vit_large_patch14_reg4_dinov2.lvd142m` | `vit_large_patch16_dinov3.lvd1689m` |
| Patch size | 14 | 16 |
| Pretraining data | LVD-142M | LVD-1689M (distilled from ViT-7B) |
| Native input | 518×518 | 256×256 |
| Extra tokens | 4 register | 4 storage |
| Embedding dim | 1024 | 1024 |
| Params | 304M | 303M |

**Critical gotcha**: 518 is not divisible by patch16 (518/16 = 32.375). For DINOv3 we use **512×512** (32×32 = 1024 patches) as the closest divisible-by-16 size to 518. DINOv2 stays at **518×518** (37×37 patches).

---

## Training Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 (head-only)              PHASE 2 (full finetune)                │
│  ~few hours                       ~24–48 hours                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  timm pretrained weights          Phase 1's model_best.pth.tar           │
│         │                                │                               │
│         ▼                                ▼                               │
│  ┌─────────────┐                  ┌─────────────┐                        │
│  │ ViT backbone│ ◀ FROZEN         │ ViT backbone│ ◀ TRAINABLE            │
│  │             │                  │ (layer-wise │   (layer LR decay 0.75)│
│  └─────────────┘                  │  LR decay)  │                        │
│         │                         └─────────────┘                        │
│         ▼                                │                               │
│  ┌─────────────┐                         ▼                               │
│  │    head     │ ◀ TRAINABLE      ┌─────────────┐                        │
│  │  (7806 cls) │                  │    head     │ ◀ TRAINABLE            │
│  └─────────────┘                  └─────────────┘                        │
│                                                                          │
│  opt: Adam     lr: 0.01           opt: AdamW   lr: 5e-5                  │
│  wd: 0         warmup: 0          wd: 0.05    warmup: 5 ep               │
│  epochs: 100                      epochs: 100                            │
│  no layer decay                   layer_decay: 0.75                      │
│                                                                          │
│  train_vit.py --freeze-backbone   train_vit.py --initial-checkpoint ...  │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │ extract_vit_embeddings.py     │
                      │ CLS-token feature extraction  │
                      │ → parquet for ensemble        │
                      └───────────────────────────────┘
```

---

## Recipe Choice (Phase 2)

**Hybrid — Exp-10 refinements (AdamW + wd=0.05 + layer_decay=0.75 + warmup=5).**

The PlantNet reference recipe (Adam, wd=0, no layer decay, no warmup) worked for ViT-Base, but ViT-Large has ~3× more parameters and more layers. Published ViT-Large finetuning recipes (MAE, BEiT, DINOv2 official evaluation) all use:

- **Layer-wise LR decay (0.65–0.85)** — biggest single improvement for ViT finetuning; earlier layers get smaller LR, preserving pretrained features.
- **AdamW + weight decay 0.05** — standard regularization for Large ViTs.
- **Warmup 5 epochs** — stabilizes early gradient flow when the head is already trained but backbone is freshly unfrozen.

Phase 1 keeps reference settings (Adam, lr=0.01, wd=0) — head-only training is simple and the reference values are proven.

---

## Config Field Table

| Field | P1 (both models) | P2 (both models) |
|---|---|---|
| `model` | `vit_large_patch14_reg4_dinov2.lvd142m` / `vit_large_patch16_dinov3.lvd1689m` | same |
| `pretrained` | `true` | omitted (loaded via `--initial-checkpoint`) |
| `freeze_backbone` | `true` | omitted |
| `freeze_backbone_prefix` | `head` | — |
| `input_size` | `[3, 518, 518]` (DINOv2) / `[3, 512, 512]` (DINOv3) | same |
| `num_classes` | 7806 | 7806 |
| `epochs` | 100 | 100 |
| `batch_size` (per-GPU, H200 141 GB) | DINOv2: 96 / DINOv3: 192 | DINOv2: 24 / DINOv3: 48 |
| `opt` | `adam` | `adamw` |
| `lr` | 0.01 | 5e-5 |
| `weight_decay` | 0.0 | 0.05 |
| `layer_decay` | omitted | 0.75 |
| `sched` | cosine | cosine |
| `warmup_epochs` | 0 | 5 |
| `warmup_lr` | 1e-5 | 1e-6 |
| `min_lr` | 0 | 1e-7 |
| `smoothing` | 0.1 | 0.1 |
| `drop_path` | 0.05 | 0.2 |
| `mixup` / `cutmix` | 0.8 / 1.0 | 0.8 / 1.0 |
| `aa` | rand-m9-mstd0.5-inc1 | rand-m9-mstd0.5-inc1 |
| `amp` / `amp_dtype` | true / float16 | true / float16 |
| `channels_last` | true | true |
| `model_ema` / `model_ema_decay` | true / 0.9998 | true / 0.9998 |
| `eval_metric` | top1 | top1 |
| `patience_epochs` | 10 | 10 |
| `seed` | 42 | 42 |

Batch sizes are memory-bound estimates for a single H200 (141 GB); validate with the smoke test before launching full runs.

---

## File Tree

```
user/alper/src/Exp-13_ViT_finetuning/
├── train_vit.py                          # copy of Exp-10/train_convnext.py (model-agnostic)
├── extract_vit_embeddings.py             # mirrors Exp-7 CLS-token extraction (tile_size configurable)
├── configs/
│   ├── phase1_dinov2_large.yaml
│   ├── phase1_dinov3_large.yaml
│   ├── phase2_dinov2_large.yaml
│   └── phase2_dinov3_large.yaml
├── slurm/
│   ├── phase1_train.sbatch               # 4x h200, MODEL env var
│   ├── phase2_train.sbatch               # 4x h200, MODEL env var, chained via afterok
│   ├── extract_embeddings.sbatch         # 1x h200, MODEL env var (tile_size auto)
│   └── submit_chain.sh                   # helper: submits P1 + P2 with dependency
├── scripts/
│   ├── verify_model_loading.py           # smoke test for both ViT-L variants
│   └── export_checkpoint.py              # copied from Exp-10 unchanged
└── docs/
    └── Exp13_ViT_Finetuning_Plan.md      # this doc
```

---

## Key Code Adaptations

### 1. `train_vit.py` — zero logic changes
`Exp-10/train_convnext.py` is fully model-agnostic (uses `timm.create_model(args.model, ...)`), and already supports `--freeze-backbone` with `--freeze-backbone-prefix head`. Copied verbatim; ViT models expose `model.head` as a Linear layer so the prefix match just works.

### 2. `extract_vit_embeddings.py` — mirrors Exp-7
Extraction follows the production pattern from `Exp-7/extract_multiscale_embeddings.py`:

```python
features = model.forward_features(tensors)   # [B, 1+reg+N, D]
cls_tokens = features[:, 0, :]               # [B, 1024]
logits = model.head(cls_tokens)              # [B, 7806]
```

Differences from Exp-7:
- `tile_size` is configurable (518 for DINOv2, 512 for DINOv3) — hardcoded 518 was OK for ViT-B patch14 only.
- `img_size=tile_size` is passed to `timm.create_model` so positional embeddings are interpolated from the pretrained resolution to the inference resolution.
- Output parquet schema identical to Exp-7 (`image_name`, `tile_index`, `embedding_orig_f16`, `teacher_logits_f16`, `max_prob`, `passed_aug_consistency`) so existing aggregation (`aggregate_tiles.py`, `multiscale_aggregate.py`) consumes it unchanged.

### 3. SLURM sbatch files — MODEL env var switches variant
- `phase1_train.sbatch`: `sbatch --export=MODEL=dinov2_large slurm/phase1_train.sbatch`
- `phase2_train.sbatch`: same pattern; looks for `vit_${MODEL}_phase1/model_best.pth.tar`
- `extract_embeddings.sbatch`: auto-selects tile_size (518 vs 512) based on MODEL
- `submit_chain.sh dinov2_large`: submits P1 + P2 with `--dependency=afterok`

---

## Recipe Deviation Notes (vs PlantNet reference)

| Change | Reference (ViT-B) | Exp-13 (ViT-L) | Why |
|---|---|---|---|
| Optimizer P2 | Adam | AdamW | Decoupled WD; standard for Large ViT |
| Weight decay P2 | 0.0 | 0.05 | Regularize larger param count |
| Layer decay P2 | none | 0.75 | MAE/DINOv2 official finetuning uses 0.65–0.85 for ViT-L |
| Warmup P2 | 0 epochs | 5 epochs | Stabilize freshly-unfrozen backbone |
| Drop path P2 | 0.05 | 0.2 | Larger model → more stochastic depth; per MAE ViT-L recipe |
| DINOv3 input | n/a (ViT-B only) | 512 (not 518) | patch16 divisibility |

All other fields match the reference.

---

## Verification Plan

1. **Smoke test** (before any SLURM job):
   ```
   python scripts/verify_model_loading.py
   ```
   Loads both ViT-L variants, runs a forward pass on a (2, 3, 518, 518) and (2, 3, 512, 512) tensor respectively, and prints `forward_features` shape + `forward_head(pre_logits=True)` shape. Expect `[2, 1024]` for both.
2. **Dry-run training** (1 epoch, small subset, 1 GPU):
   ```
   torchrun --nproc_per_node=1 train_vit.py \
       -c configs/phase1_dinov2_large.yaml \
       --epochs 1 --train-num-samples 5000 \
       --experiment dryrun_dinov2
   ```
   Confirm loss decreases, no NaNs, checkpoint saves.
3. **Full Phase 1** (4× H200): should hit peak top-1 within 20–40 epochs (head-only converges fast); early-stop via `patience_epochs: 10`.
4. **Full Phase 2** (4× H200, chained with `--dependency=afterok`): loads P1 best via `--initial-checkpoint`, trains 100 epochs with EMA tracking. Monitor `summary.csv` per-epoch.
5. **Inference extraction** on PlantCLEF test set: `sbatch --export=MODEL=dinov2_large,SCALE=9 slurm/extract_embeddings.sbatch`. Produces `.parquet` embeddings compatible with the existing ensemble pipeline.
6. **Ensemble integration check**: plug extracted embeddings into the existing tile-ensembling script and measure F1 delta vs current best (0.40100).

---

## Open Items / Pre-execution Sanity Checks

- Verify `timm` version in the PACE conda env supports `vit_large_patch16_dinov3.lvd1689m` (requires timm ≥ 1.0.14 per the DINOv3 release). If not, upgrade before running.
- Confirm PlantCLEF 2024 data root on PACE matches the Exp-10 configs (`/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/train/PlantCLEF2024/`).
- Confirm `class_mapping.txt` path and 7806 class count.
- On the first Phase 1 job for each variant, watch nvidia-smi after ~10 steps — if memory is not fully utilised, bump batch size before the main sweep.

---

## Usage Quick Reference

```bash
# 1. Architecture sanity check (run locally or on a small dev GPU)
python scripts/verify_model_loading.py

# 2. Submit chained P1 + P2 for DINOv2 ViT-L
cd user/alper/src/Exp-13_ViT_finetuning
bash slurm/submit_chain.sh dinov2_large

# 3. Submit chained P1 + P2 for DINOv3 ViT-L
bash slurm/submit_chain.sh dinov3_large

# 4. After P2 completes, extract test embeddings at scale 9
sbatch --export=MODEL=dinov2_large,SCALE=9 slurm/extract_embeddings.sbatch
sbatch --export=MODEL=dinov3_large,SCALE=9 slurm/extract_embeddings.sbatch

# 5. (Optional) Export EMA weights as a standalone .pth
python scripts/export_checkpoint.py \
    --checkpoint /path/to/vit_dinov2_large_phase2/model_best.pth.tar \
    --output     /path/to/vit_dinov2_large_plantclef.pth \
    --use-ema
```
