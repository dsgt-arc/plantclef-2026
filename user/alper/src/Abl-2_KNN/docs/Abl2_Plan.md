# Abl-2 — kNN Knockout Ablation

## Reference

| Run | Private F1 | Public F1 |
|---|---|---|
| B_R250 (full) | 0.44649 | 0.48414 |
| Abl-1 (− ConvNeXt) | 0.45001 | 0.47136 |
| **Abl-2 (− ConvNeXt − kNN)** | TBD | TBD |

Cumulative knockout: peeling kNN on top of Abl-1's already-removed ConvNeXt.

## What this ablation removes

Just the kNN retriever. Concretely, the ViT-L pre-fusion parquet is
rebuilt with **β_match = 1.0** (i.e., pure classifier softmax, no kNN
contribution). The downstream ensemble + habitat-fit + admission stack is
identical to Abl-1.

| Layer | Setting |
|---|---|
| ViT-L per-tile classifier softmax | T = 1.5 |
| **kNN retriever (FAISS, β_match = 0.70)** | **REMOVED** (β_match → 1.0) |
| image_topk per image | 64 |
| Geo mask (SW Europe) | applied |
| ViT controller | `vit_rnnb030.json` (α=0.30 RNNB temporal fusion) |
| ViT aggregator | max, with `{"RNNB": "sim_w"}` per-source override |
| Sim signal | `Exp-14/outputs/signals/visit_similarity.json` |
| ConvNeXt branch | REMOVED (carried over from Abl-1) |
| Threshold | 0.085 |
| Top-k | 10 |
| Habitat-fit β | 0.10 |
| Habitat-fit radius | 250 km |
| Habitat-fit n_near_min | 40 |
| Habitat-fit altitude | on |
| Aquatic blacklist mask | applied |
| Biogeo per-source mask | applied |
| Per-source admission | `{OPTMix: n_floor=2, LISAH: n_floor=2}` |

## Why a fork instead of `make_vitL_prefusion.py --beta-multis 1.00`

`make_vitL_prefusion.py` already supports β=1.0 — line 198 in that script
has `if b < 1.0: blend else: skip`, so at β=1.0 the blending is mathematically
a no-op. **But the script still runs the entire FAISS retrieval pass**
(load index, query every tile, accumulate p_knn) and discards the result.

For ~50 K tiles across 4 scales, that's ~1.5–2 hours of wasted CPU plus a
hard dependency on the FAISS index file existing on PACE.

`Abl-2/scripts/make_vitL_prefusion_no_knn.py` is a stripped fork: classifier
softmax → max_prob → image_topk → geo mask. No FAISS imports, no retriever
construction, no kNN queries.

## Cumulative vs single-knockout

This is the **cumulative** path (Abl-1's ConvNeXt-removed stack with kNN
also removed). The single-knockout alternative — full B_R250 minus *only*
kNN, keeping ConvNeXt — was not chosen because:

1. Abl-1 result (private +0.00352 from removing ConvNeXt) implies the
   final paper pipeline will not include ConvNeXt. The cumulative path
   measures kNN's contribution in the configuration that matters for
   the paper.
2. Reusing `Abl-1/scripts/ensemble_vit_only.py` eliminates code
   duplication.

If reviewers later request the single-knockout (kNN-only removal from
full B_R250), it requires a small extra job: re-run `ensemble_v2.py` with
ConvNeXt included against this no-kNN pre-fusion parquet. ~30 CPU-min + 1 LB.

## Methodology disclaimer (paper caption text)

Fixed-knob ablation. β_match = 0.70 was tuned with kNN present; the
downstream habitat-fit knobs (β=0.10, R=250, n=40) and per-source
admission floors were tuned on top of the kNN-included ensemble. They
are NOT re-tuned for the no-kNN pipeline. Reported ΔF1 is a **lower
bound** on kNN's marginal contribution.

Per project memory (Exp-16 veg-fraction findings, 2026-04-30): kNN was
estimated to contribute **~+0.030 F1 on public LB**. Private contribution
is unknown until this ablation completes. Given the Abl-1 finding that
ConvNeXt's public/private contribution diverged, do not assume kNN's
private contribution matches its public estimate.

## Files in this folder

```
Abl-2_KNN/
├── docs/
│   └── Abl2_Plan.md                                  (this file)
├── scripts/
│   └── make_vitL_prefusion_no_knn.py                 (forked from Exp-14C/make_vitL_prefusion.py)
└── slurm/
    └── ablation_no_knn.sbatch                         (full pipeline: Step A + B + C)
```

Reuses `Abl-1/scripts/ensemble_vit_only.py` for Step B.

## How to run on PACE

```bash
cd /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Abl-2_KNN
sbatch slurm/ablation_no_knn.sbatch
```

Output CSVs land at:
```
${PLANTCLEF_ROOT}/aerten3/submissions/abl2_no_knn/
  abl2_vitL_noknn_w1.00_thr0.085_top10.csv      ← bare CSV (no habitat-fit, informational)
  abl2_s9_B_R250_noknn.csv                       ← HEADLINE submission
```

Submit `abl2_s9_B_R250_noknn.csv` to Kaggle, record both public and private F1.

## Decision rule

Compare against **Abl-1** (the new anchor since ConvNeXt was dropped):

- If private(abl2) − private(abl1) is sub-noise (|Δ| < 0.0015 on ~1873-image
  private set): kNN contribution is negligible on private. **Drop kNN**
  from the paper pipeline. Saves ~1.5 GPU-h per re-extraction and removes
  the FAISS dependency.
- If private(abl2) − private(abl1) is significantly negative (≤ −0.005):
  kNN is load-bearing on private. **Keep kNN.** Worth re-tuning β_match
  for the ConvNeXt-free pipeline in a follow-up (Abl-2b).
- If private(abl2) − private(abl1) is significantly **positive**: kNN
  was hurting private generalization (same public/private divergence
  pattern as ConvNeXt). **Drop kNN** and document the finding — strong
  evidence that public-LB-tuned components don't transfer.

## Expected wall-time and cost

| Phase | Cost |
|---|---|
| Step A (no-kNN pre-fusion build) | ~30–60 min CPU (vs ~1.5–2 h with kNN) |
| Step B (ViT-only fusion + parquet) | ~15–30 min CPU |
| Step C (habitat-fit + admission + CSV) | ~5–10 min CPU |
| GPU | 0 |
| LB submissions | 1 (`abl2_s9_B_R250_noknn.csv`) |
| Total wall-time | < 2 hours |
