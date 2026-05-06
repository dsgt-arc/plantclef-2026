---
experiment: "003-tileq-decoder"
status: in-progress
date: 2026-05-03
last_updated: 2026-05-05
author: Murilo Gustineli
---

# Experiment 003: TileQ-Decoder — Results

## Status: IN PROGRESS

A0 submission-pipeline derisk **complete**. Test-CLS extraction (P3), library
code + sanity checks (P4), and both training runs (P5, Sampler A + Sampler B)
**complete**. Inference + headline submissions A1/A2 (P6) is the next step;
Kaggle LB is the only generalisation signal we have left.

See [proposal.md](proposal.md) for hypothesis framing (H1/H2) and
[spec.md](spec.md) for the technical specification.

## Kaggle Leaderboard Results — A0 baselines

Competition metric: **macro-averaged F1** per image. All scores are public LB.

| # | Method | Avg species/img | Public LB | Expected | Match? |
|---|--------|----------------:|----------:|---------:|--------|
| **A0a** | `max-prob`, top-10, grid_4x4 | 10.0 | **0.22986** | ~0.23 | ✓ |
| **A0b** | `sorted-prob`, top-9, grid_4x4 | 5.0 | **0.30810** | 0.30810 | ✓ exact |

Both submissions generated from
[`experiments/002-classification-methods/classify.py`](../002-classification-methods/classify.py)
against `shared/inference/dinov2-vit-base/grid_4x4.parquet`. No new code.

A0b is the **H1 anchor**: the decoder must clear F1 = 0.30810 to confirm H1.

## Priority 3 — Test-CLS extraction

Job 7966845 on V100, 16 min. Output:
**`shared/inference/dinov2-vit-base/cls_grid_4x4.parquet`** — 49.6 MB,
33,680 rows × `(image_name, tile_idx, cls_embedding[float16, 768])`. Schema
matches spec §3.1; 2,105 unique images × 16 tiles each; joinable to
`grid_4x4.parquet` on `(image_name, tile_idx)`.

**Documented domain shift** (verified during validation):

| | Train CLS | Test CLS |
|---|---:|---:|
| L2 norm mean | 18.6 | **25.4** |
| L2 norm std | 3.2 | 2.8 |
| Source | single-plant photos (1.4 M Pl@ntNet) | vegetation-quadrat patches |

The ~37 % higher test L2 is the empirical signature of the train↔test domain
gap. The decoder's `proj_kv` linear is expected to absorb the scale at training
time (verified post-hoc in P5; both samplers train cleanly without scale
collapse — see "Sanity checks" below).

## Priority 4 — Library code + sanity checks

**Library code** (`user/murilo/src/plantclef/`):
- `asl_loss.py` — `AsymmetricLoss(γ⁻=4, γ⁺=0, m=0.05)` (spec §2.2)
- `tileq_decoder.py` — `TileQDecoder` (4,509,824 parameters; matches spec §2.1)
- `data/samplers.py` — `SamplerA` (naive random-background) and `SamplerB`
  (test-anchored, FAISS-relabelled)

**Build pipeline** (`experiments/003-tileq-decoder/`):
- `build_train_lookup.py` — sorts 1.4 M train CLSs by class_idx; persists
  `train_cls_by_class_idx.npz` + `species_weights.npy`
- `build_faiss_background_map.py` — builds `IndexFlatIP` over normalised train
  CLSs, queries top-1 per test tile, applies threshold τ
- `rethreshold.py` — re-applies τ on cached results without re-running FAISS
- `dataset.py` — `PseudoQuadratDataset` (multi-worker safe `IterableDataset`)
- `sanity_check.py` — shape audit + overfit-one-batch (gates training)
- `train.py` — full training loop with W&B + TensorBoard logging

**Sbatchs** matching `extract_test_cls.sbatch` convention (TMPDIR venv,
cu121-pinned torch): `build_train_lookup.{sh,sbatch}`,
`build_faiss_background_map.{sh,sbatch}`, `train.{sh,sbatch}` (parameterised
by `--sampler {A,B}`).

**Tests**: 80 unit tests covering ASL loss, decoder shape/grad/scale,
samplers A+B (label set bookkeeping is the trickiest property —
overwritten background labels dropped, deduplicated against injected, etc.),
build scripts, dataset wrapper, lr schedule.

### Build artefacts

| Artefact | Job | Wallclock | Size |
|---|---|---:|---:|
| `train_cls_by_class_idx.npz` (sorted-by-class, offsets) | 8029519 | 2:14 | 4.1 GB |
| `species_weights.npy` (1/√count, sums to 1) | (same) | — | 31 KB |
| `sampler_b_lookup.npz` (image_names, test_cls, top1_sim, top1_class_idx, tile_to_class_idx, threshold) | 8035733 | 11:15 | 100 MB |
| `tile_top1_sim_histogram.png` | (same) | — | 38 KB |

### Train CLS lookup — class-frequency stats

Verified against spec §3.3 expectations (median 176, max 823):

| Stat | Spec | Empirical |
|---|---:|---:|
| n_classes | 7,806 | 7,806 ✓ |
| total rows | 1,408,033 | 1,408,033 ✓ |
| rows/species median | 176 | 176 ✓ |
| rows/species max | 823 | 823 ✓ |
| rows/species min | (p10 = 6) | **1** |

Min = 1 (vs spec's p10 = 6) confirms the long tail extends below the 10th
percentile. No zero-count classes, so `species_weights` is well-defined
everywhere.

### Sanity check (overfit-one-batch, login-node 1-CPU, 200 steps batch=32)

| Sampler | Initial loss | Final loss | Reduction | Wallclock |
|---|---:|---:|---:|---:|
| A | 306.66 | 0.0157 | **19,561×** | 44.2 s |
| B (τ=0.65) | 305.10 | 0.0233 | **13,075×** | 46.0 s |

Both reach near-zero loss by step 80; final ratio well above the 5× gate.
**Critically** — both samplers converge at near-identical rates despite
Sampler B's tile L2 ≈ 24 (vs Sampler A's ≈ 19). This empirically validates
the spec §2.3 assumption that `proj_kv` absorbs the scale shift during
training.

## FAISS τ tuning — empirical override of spec default

**Spec default**: τ = 0.85, "pick at the visible shoulder of the histogram."

**Reality** (from `tile_top1_sim_histogram.png`):

```
top-1 cosine similarity (n=33,680):
  min=0.279  max=0.885  mean=0.513
  p50=0.508  p90=0.617  p95=0.655  p99=0.751
```

The histogram is **unimodal centred at 0.51** with a thin right tail to 0.89.
**No shoulder** at 0.85 — only 18 tiles (0.05 %) would get labels there.

| τ | tiles labelled | % of 33,680 |
|---|---:|---:|
| 0.85 (spec default) | 18 | 0.05 % |
| 0.80 | 114 | 0.34 % |
| 0.75 | 342 | 1.02 % |
| **0.65 (chosen)** | **1,838** | **5.46 %** |
| 0.60 | 4,509 | 13.4 % |

**τ = 0.65 chosen** for the final Sampler B. Rationale:
1. **Asymmetric cost**: false-positive labels are toxic (teach decoder wrong
   test-tile↔species mappings); false-negatives are absorbed by ASL γ⁻=4
   margin. Err high.
2. **Spot-check at the high-sim tail confirms biological coherence** —
   top-10 highest-sim tiles all match plausible species:
   - 4 tiles in `GUARDEN-CBNMed-29-*-20240428` quadrats → **Eryngium maritimum**
     (sim 0.87-0.89)
   - 5 tiles in `GUARDEN-CBNMed-{8,9}-*-20240429` consecutive sites →
     **Quercus coccifera** (sim 0.86-0.87)
3. **0.65 is past the bulk of the distribution** (>p95) but still in a regime
   where retrieval looks sensible. ~0.9 surviving FAISS-relabelled tiles per
   pseudo-quadrat on average — a meaningful Sampler B contrast.

Why the spec's 0.85 prior didn't pan out: prior assumed bimodal distribution
where same-species pairs cluster at ~0.85. Empirically the train↔test domain
gap is large enough that even genuine species matches max out at 0.89, and
the bulk of test tiles sit in the 0.4–0.6 cosine range with their nearest
train neighbour. CONTEXT.md flagged the L2 disparity in advance; this is its
directional consequence.

## Priority 5 — Training runs (Sampler A + B)

**Both runs at 80,000 steps, batch 32, AdamW 3e-4 → 1e-5 cosine,
ASL γ⁻=4 γ⁺=0 m=0.05, grad-clip 1.0, fp32 (V100 fallback skipped — RTX 6000
held), seed 42.**

| | Sampler A | Sampler B |
|---|---:|---:|
| SLURM job | 8038028 | 8038496 |
| GPU | RTX 6000 (atl1-1-03-006-25-0) | RTX 6000 (atl1-1-03-006-3-0) |
| Wallclock | **12:05** (9.4 min train) | **11:46** (9.1 min train) |
| Step rate | 142.5 step/s | 146.7 step/s |
| Peak RAM | 24.0 GB | 24.0 GB |
| W&B run | [u5ma36yz](https://wandb.ai/dsgt-arc/plantclef-2026-exp003/runs/u5ma36yz) | [hdcx1p23](https://wandb.ai/dsgt-arc/plantclef-2026-exp003/runs/hdcx1p23) |
| Final checkpoint | `decoder_samplerA.pt` (52 MB) | `decoder_samplerB.pt` (52 MB) |

### Loss + val-acc trajectory (synthetic)

Both samplers; values pulled from `loss_sampler{A,B}.csv` (full curves in
`loss_sampler{A,B}.png`):

| Step | A loss | A val_loss | A acc@10 | B loss | B val_loss | B acc@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 306.72 | — | — | 305.02 | — | — |
| 2,000 | 8.36 | 8.39 | 0.9251 | 5.99 | 6.30 | 0.9931 |
| 4,000 | 7.30 | 7.23 | 0.9831 | 3.95 | 3.70 | 1.0000 |
| 8,000 | 6.45 | 6.33 | 0.9916 | 2.49 | 2.59 | 1.0000 |
| 18,000 | 5.59 | 5.66 | 0.9965 | 1.67 | 1.76 | 1.0000 |
| **80,000** | **4.91** | **4.85** | **0.9965** | **0.93** | **0.92** | **1.0000** |

**Sampler B's final loss is ~5.3× lower than Sampler A's** (0.93 vs 4.85).

### What this offline gap does (and does NOT) tell us

**Important caveat**: Sampler A and Sampler B are validated on
**different synthetic distributions** (each val set is drawn from the same
sampler as training). The lower B loss is **NOT** a generalisation signal.
It tells us only that **B's synthetic distribution is easier to fit**.

The mechanistic reason: A's 11 background tiles are real plant CLSs from
"wrong" species — they sit confidently in train-CLS regions of the embedding
space, and ASL must learn to label them all-zero anyway. B's 11 untouched
tiles are real test-quadrat CLSs which (per the FAISS histogram) live in a
direction *away* from any train cluster — the decoder finds it easier to
output near-zero logits everywhere on those.

This was anticipated in spec §7.3 ("synthetic-distribution overfit (Sampler A)")
as a failure mode for Sampler A specifically. The empirical loss gap is
consistent with that mode having activated.

**The H2 question — does B's training distribution transfer better to the
real test set than A's? — is answerable only by Kaggle.** Synthetic val loss
cannot adjudicate it, by construction.

### Both samplers fit their training distribution cleanly

- **A**: train ↘ val tightly tracked (4.91 vs 4.85). Acc@10 plateaus at
  99.65 %.
- **B**: train ↘ val tightly tracked (0.93 vs 0.92). Acc@10 saturates at
  100 % by step 4,000.
- No overfitting on synthetic-vs-synthetic for either run.
- LR schedule: warmup ended at step 1,000 at peak 3e-4; final lr = 1e-5
  exactly (cosine floor reached).

## Interpretation (so far, offline-only)

1. **Submission pipeline is gas-tight.** A0a/A0b derisk + sanity-check gate
   in the training sbatch + 80/80 unit tests + W&B + TB logging — every
   stage has either an artefact or a test backing it.
2. **Probability magnitude > tile frequency** (re-confirmed at A0a vs A0b).
   Δ = +0.078 F1 on the same per-tile predictions; this is the load-bearing
   prior for the decoder approach.
3. **Train↔test domain gap is large but not training-fatal.** Test L2 ≈ 25.4
   (vs train 18.6) and FAISS top-1 cosine maxes at 0.885 — but the decoder's
   `proj_kv` linear adapts the scale during training, and both samplers
   converge.
4. **Sampler B's much lower training loss does NOT predict higher Kaggle F1.**
   It reflects a fundamentally different (and easier) supervision signal,
   not better generalisation. Kaggle is the only test signal.
5. **n_positives = 5 fixed** for v1 (deferred to future work in spec §8). Both
   runs used identical hyperparameters as the H2 contrast requires.

## What's next (priority-ordered)

- [x] **Priority 0** — A0a/A0b Kaggle pipeline derisk (LB 0.22986, 0.30810)
- [x] **Priority 3** — Test-CLS extraction (`cls_grid_4x4.parquet`)
- [x] **Priority 4** — Library code + sanity checks (80 tests pass; both
      samplers overfit-one-batch ratio > 13,000×)
- [x] **Priority 5** — Train Run 1 (Sampler A) + Run 2 (Sampler B) at 80k
      steps each
- [ ] **Priority 6** — Inference + headline submissions A1, A2:
      `infer.py --checkpoint <ckpt> --topk 9,10 --output-prefix submissions/<tag>`
      → 4 CSVs (A1@k9, A1@k10, A2@k9, A2@k10). Submit all 4 to Kaggle.
- [ ] **Priority 7** — A3 score-blend bonus (5 α values, no retraining)
- [ ] **Priority 8** — Writeup: decide H1, H2 outcomes per `proposal.md`
      success criteria; document failure modes hit (§7); paper-appendix
      wallclock per stage; future-work directions (§8)

## Files

| File | Purpose | Status |
|---|---|---|
| `submissions/A0a_max_prob_grid4.csv` | A0a (LB 0.22986) | done |
| `submissions/A0b_sorted_prob_topk9_grid4.csv` | A0b (LB 0.30810) | done |
| `artifacts/train_cls_by_class_idx.npz` | sorted-by-class train CLS lookup | 4.1 GB |
| `artifacts/species_weights.npy` | 1/√count weights | 31 KB |
| `artifacts/sampler_b_lookup.npz` | image_names, test_cls, top1_sim, top1_class_idx, tile_to_class_idx, threshold (τ=0.65) | 100 MB |
| `artifacts/tile_top1_sim_histogram.png` | FAISS τ decision artefact | committed |
| `artifacts/faiss_threshold.txt` | percentile + coverage table | committed |
| `artifacts/checkpoints/decoder_samplerA.pt` | Final Sampler A weights (step 80k) | 52 MB |
| `artifacts/checkpoints/decoder_samplerB.pt` | Final Sampler B weights (step 80k) | 52 MB |
| `artifacts/loss_sampler{A,B}.csv` + `.png` | Training curves | committed |
| `artifacts/config_sampler{A,B}.yaml` | Run hyperparameters | committed |
| `shared/inference/dinov2-vit-base/cls_grid_4x4.parquet` | Cached test CLSs | 49.6 MB |
| `submissions/SUBMISSIONS.md` | Kaggle CLI message + LB log | source-of-truth |

W&B project: <https://wandb.ai/dsgt-arc/plantclef-2026-exp003>

Branch: `mg/tileq-decoder` (PR #15)
