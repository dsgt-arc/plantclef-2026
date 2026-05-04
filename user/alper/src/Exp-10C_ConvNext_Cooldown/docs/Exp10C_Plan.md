# Exp-10C — ConvNeXt-Large cooldown on combined train+val+test

## Goal
Mirror Exp-13 phase2v3 cooldown for the ConvNeXt-Large branch. ViT-L cooldown
delivered +0.011 standalone F1 from short-LR cooldown training on the union
of train + val + test images. Apply the same recipe to ConvNeXt-Large in the
production ensemble.

## Source checkpoint
`aerten3/checkpoints/exp10_convnext/convnext_large_phase2/model_best.pth.tar`
(or `convnext_large_phase2_4gpu/model_best.pth.tar` if that's what's actually
on disk — sbatch supports both via env var override)

## Recipe (mirrors Exp-13 phase2v3)
- Backbone: convnext_large.dinov3_lvd1689m (production ConvNeXt)
- Init: best EMA weights from Phase 2 checkpoint
- Data: `data/raw/train_full` (symlinked combined train+val+test ImageFolder, already exists)
- Epochs: 20 (longer than Exp-13's 10 — softer way to "push more" than raising LR;
  more time under cosine decay tail squeezes additional gain from val/test data)
- LR: 2.0e-5 (low cooldown LR; was 3.0e-5 in Phase 2)
- Layer decay: 0.85 (was 0.8 in Phase 2 — match Exp-13 phase2v3)
- Warmup: 0 (no warmup in cooldown)
- Mixup α=0.8, CutMix α=1.0 (kept on, same as Phase 2)
- drop_path: 0.3 (ConvNeXt's higher value, same as Phase 2)
- EMA: on, decay 0.9998
- Patience: 999 (disable early stopping; val metric on train_full is contaminated)
- Eval metric "best" selection is contaminated (val ⊂ train_full) — use `last.pth.tar`
  as the production checkpoint after run completes

## Expected outcome
| Reference | Expected ConvNeXt cooldown lift | Direct ensemble F1 contribution (w=0.23) |
|---|---|---|
| ViT-L cooldown delta | +0.011 standalone | — |
| ConvNeXt projected | +0.005 to +0.015 standalone | +0.001 to +0.003 |

LB lift expected to be sub-noise (±0.005) directly. Real value comes if the
new ConvNeXt also adds diversity to the ensemble — measurable only by re-running
ensemble sweep with the new ConvNeXt parquets.

## Wallclock + cost
- Cooldown training: ~26-30h on 4×H200 for 20 epochs (Exp-13 phase2v3 took 14h27m for 10 epochs;
  scaling linearly: 20 epochs ≈ 29h)
- Re-extraction at scales 4-7 (K scheme): ~30-60 min H200 array job
- Total: ~28-32h H200 time
- Sbatch walltime set to 30h; resume logic via `last.pth.tar` will pick up
  if walltime is exceeded (just resubmit)

## Order of operations
1. Submit `slurm/phase2v3_train.sbatch` (10h, 4×H200)
2. After completion: submit re-extraction sbatch (Exp-10 `extract_embeddings.sbatch` with new checkpoint path)
3. After re-extraction: re-run Exp-14C ensemble sweep with new ConvNeXt parquets
4. Compare against ensemble best from un-cooldowned ConvNeXt

## What's NOT in this experiment
- No new architecture changes — same ConvNeXt-Large
- No new augmentation — same Mixup/CutMix as Phase 2
- No LoRA / ArcFace — that's a separate (deferred) experiment
- No retraining the head — still 7806 species classifier
