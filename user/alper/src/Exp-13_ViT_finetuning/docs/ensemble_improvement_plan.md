# Ensemble F1 Improvement Plan — Post-ViT-L Phase 2

## Context

Current best ensemble: **F1 = 0.44428** (ConvNeXt-L w=0.23 + Exp-9B ViT-B+LoRA+ArcFace+kNN w=0.77, per `MEMORY.md`).

The newly-trained DINOv2 ViT-L (Exp-13 Phase 2 v2, currently 77.45 EMA top-1, projected plateau ~77.6–77.9) is **not yet integrated** into the production ensemble. This plan stacks ensemble-F1-oriented improvements in priority order.

A 0.78 ensemble F1 target is unrealistic with current architecture/data. Realistic landing zone for this sprint: **F1 = 0.46–0.48** (≈+0.02–0.04 over baseline).

---

## Recommended path (ordered by ensemble F1 ROI)

### Step 1 — Extract ViT-L embeddings and add to ensemble (highest leverage)
- Wait for current Phase 2 v2 run to finish or early-stop, then take best EMA checkpoint.
- Run `extract_vit_embeddings.py` at the standard 77-tile multi-scale config (matches Exp-7 / ConvNeXt extraction).
- Sweep blend weight in the existing aggregation pipeline. Start near `w_vitL = 0.20–0.40`, drop ConvNeXt or Exp-9B weight proportionally.
- **Expected gain: +0.010–0.025 F1**.
- Files involved (read-only context):
  - `user/alper/src/Exp-13_ViT_finetuning/extract_vit_embeddings.py` — extraction entrypoint
  - Exp-11B aggregation/blending script (locate in `Exp-11B_KNN_ArcFace_Conv_Ensemble/`)
- Cost: 1 extraction run (~3–5 GPU-h on H200) + ensemble weight sweep on CPU (~30 min).

### Step 2 — Per-class beta recalibration on the ViT-L outputs
- Same trick that pushed Exp-9B from F1=0.41754 → 0.42267 (+0.005, per `MEMORY.md` Exp-9B findings) — pure post-hoc temperature/beta tune on val, no retraining.
- Apply the same recipe to the new ViT-L logits before ensemble blending.
- **Expected gain: +0.003–0.008 F1** stacked on Step 1.
- Files to read/mirror:
  - Whatever script produced the Exp-9B beta=0.85 recalibration — check `user/alper/src/Exp-9B_*` and `Exp-11B_*`.
- Cost: ~1 hour code + minutes to run.

### Step 3 — EMA checkpoint averaging (weight soup) on ViT-L
- Average state_dicts of last 5–10 EMA checkpoints from `vit_dinov2_large_phase2v2/` before re-extracting.
- Lifts ViT-L standalone by **+0.1–0.3 pt top-1**, propagating to **+0.002–0.006 F1** in the ensemble.
- New file:
  - `user/alper/src/Exp-13_ViT_finetuning/scripts/average_checkpoints.py` — load N `.pth.tar`, average `state_dict_ema` keys, write a new checkpoint. Mirror EMA-loading pattern from `scripts/export_checkpoint.py`.
- Verification: re-extract with the soup checkpoint; expect val top-1 ≥ best-single + 0.1.
- Cost: ~30 min code + 1 GPU-h validation eval + 1 re-extraction (~3–5 GPU-h).

### Step 4 — Horizontal-flip TTA at extraction (stacking lever)

**What it does:** Run each tile through the model on both the original and a horizontally-mirrored version, average the logits, write a single combined embedding/logit row to parquet.

**Why it helps the ensemble:**
- ViT positional embeddings have residual left-right asymmetry that fine-tuning amplifies. Flipping cancels it via averaging.
- Plant images are approximately chirality-symmetric — there is no "true" left/right orientation, so flip TTA gives a more honest estimate of the model's class belief.
- Variance reduction per-member improves ensemble F1 without reducing the member-to-member diversity that ensembles depend on.

**Implementation (recommended option — batch-stacked flip):**
- In `extract_vit_embeddings.py`:
  - Build model input as `torch.cat([tiles, torch.flip(tiles, dims=[-1])], dim=0)` per batch
  - Run one forward, split outputs into original + flipped halves, average element-wise on logits
  - Write a single combined row per tile — no parquet schema change
  - ~15 lines of code change
- Add a `--flip-tta` CLI flag so it's optional.
- Time cost: ~1.6–2.0× extraction wallclock.

**Expected gain:** **+0.001–0.004 F1** in the ensemble (on top of Steps 1–3). Standalone ViT-L gains ~+0.5–1.5 pt top-1 from flip TTA, but most of that information is partially redundant with the existing 77-tile multi-scale.

**Verification:**
- Extract a small val subset with and without `--flip-tta`.
- Compute cosine(emb_raw, emb_flipped) per tile → expect 0.85–0.95.
- Compute val F1 with both parquets through aggregation pipeline → expect flip parquet ≥ non-flip + 0.001 F1.

**Files to modify:**
- `user/alper/src/Exp-13_ViT_finetuning/extract_vit_embeddings.py` — add `--flip-tta` flag, modify the forward call (~15 lines).

**Cost:** 30 min code + 1 extra extraction run (~5–8 GPU-h with flip).

### Step 5 (optional, only if Steps 1–4 underperform) — Apply same stack to existing ConvNeXt member
- The existing ConvNeXt-L extraction may also benefit from EMA averaging + flip TTA.
- Less leverage than ViT-L work because ConvNeXt is already the lower-weight member.
- **Expected gain: +0.001–0.003 F1**.
- Cost: similar to Steps 3+4 applied to a different model.

---

## Realistic outcomes

| After step | Ensemble F1 estimate | Cumulative gain |
|---|---|---|
| Baseline | 0.44428 | — |
| + Step 1 (add ViT-L to blend) | 0.455–0.470 | +0.011–0.026 |
| + Step 2 (beta recal on ViT-L) | 0.458–0.475 | +0.014–0.031 |
| + Step 3 (ViT-L weight soup) | 0.460–0.478 | +0.016–0.034 |
| + Step 4 (flip TTA on ViT-L) | 0.461–0.481 | +0.017–0.037 |
| + Step 5 (same on ConvNeXt) | 0.462–0.483 | +0.018–0.039 |

Realistic landing zone: **F1 = 0.46–0.48**. Reaching 0.50+ would require new architectural diversity (ViT-H, EVA-02, BEiT) or fundamental data improvements — not part of this sprint.

---

## What NOT to do

- **Don't** chase 0.78 ensemble F1 — not reachable with current architecture/data without years of work.
- **Don't** retrain ViT-L with different hyperparameters — current trajectory is healthy and at architecture ceiling.
- **Don't** invest in label-noise filtering or self-distillation as part of this sprint — separate experiments, deferred.
- **Don't** add DINOv3 ViT-L to the ensemble unless empirically validated — predicted marginal contribution +0.002–0.008 F1 per earlier analysis.

---

## Verification (end-to-end)

1. After each step, run the existing val-F1 aggregation script (entrypoint in `Exp-11B_*`) and record the new F1.
2. Stop early at any step where the gain is < expected lower bound — likely indicates a bug.
3. Final check: full val pass with all four members (ConvNeXt + Exp-9B ViT-B + ViT-L + flip TTA), confirm F1 monotonically improved across steps.

---

## Key files referenced

| File | Role | Status |
|---|---|---|
| `user/alper/src/Exp-13_ViT_finetuning/extract_vit_embeddings.py` | ViT-L inference extraction | Modify in Step 4 |
| `user/alper/src/Exp-13_ViT_finetuning/scripts/export_checkpoint.py` | EMA-extraction pattern | Read for averaging utility template |
| `user/alper/src/Exp-13_ViT_finetuning/scripts/average_checkpoints.py` | New SWA utility | Create in Step 3 |
| `Exp-11B_KNN_ArcFace_Conv_Ensemble/...` | Current best ensemble | Locate exact aggregation script |
| `vit_dinov2_large_phase2v2/checkpoint-*.pth.tar` | Source EMA checkpoints | Read in Step 3 |
