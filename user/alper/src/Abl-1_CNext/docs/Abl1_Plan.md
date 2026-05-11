# Abl-1 — ConvNeXt Knockout Ablation

## Reference run

| Item | Value |
|---|---|
| Submission | `exp16c_s9_B_R250.csv` |
| Source script | `Exp-16C_HabitatFit_Retune/slurm/stage9_beta_radius_at_n40.sbatch` (Family B, R=250) |
| Public LB F1 | 0.48414 |
| Private LB F1 | **0.44649** (highest private score across all submissions) |

## What this ablation removes

Just the ConvNeXt branch. The rest of the B_R250 pipeline is held fixed:

| Layer | Setting |
|---|---|
| ViT-L prefusion | `vitL_prefusion_b0.70.parquet` (Exp-14C kNN β=0.70) |
| ViT controller | `vit_rnnb030.json` (α=0.30 RNNB temporal fusion) |
| ViT aggregator | max (default), with `{"RNNB": "sim_w"}` per-source override |
| Sim signal | `Exp-14/outputs/signals/visit_similarity.json` |
| Threshold | 0.085 |
| Top-k | 10 |
| Habitat-fit β | 0.10 |
| Habitat-fit radius | 250 km |
| Habitat-fit n_near_min | 40 |
| Habitat-fit altitude | on |
| Aquatic blacklist mask | applied |
| Biogeo per-source mask | applied |
| Per-source admission | `{OPTMix: n_floor=2, LISAH: n_floor=2}` |
| **ConvNeXt branch** | **REMOVED** (was: scales 5+6+7 blended at w=0.88 ViT / 0.12 ConvNeXt) |

## Why a fork instead of `ensemble_v2.py --weights 1.00`

`Exp-14/scripts/ensemble_v2.py` requires `--convnext-scales` and runs the full
ConvNeXt pipeline (load → aggregate → fuse) even at w=1.0, then multiplies
the result by zero. Numerically equivalent but wastes ~20 min of CPU and
needs the ConvNeXt parquets to exist.

`Abl-1/scripts/ensemble_vit_only.py` is a stripped fork: ViT branch only,
no ConvNeXt code path.

## Methodology disclaimer (paper caption text)

This is a **fixed-knob** ablation. The downstream knobs in the table above
(habitat-fit β/R/n, per-source admission, threshold, topk) were tuned with
ConvNeXt present. They are not re-tuned for the ConvNeXt-removed pipeline.
Therefore the reported ΔF1 is a *lower bound* on the marginal cost of
removing ConvNeXt — a properly re-tuned ViT-only pipeline could close part
of the gap.

Per project memory: ConvNeXt's ensemble contribution was estimated at
sub-noise (~+0.001 to +0.003 F1 at w=0.23 in the older ViT-B baseline; at
W=0.88 in Exp-14C the ConvNeXt fraction is 0.12). Expected ablation cost:
within the ±0.005 LB noise floor. If confirmed, the decision is to drop
ConvNeXt from the final paper pipeline.

## Files in this folder

```
Abl-1_CNext/
├── docs/
│   └── Abl1_Plan.md                    (this file)
├── scripts/
│   └── ensemble_vit_only.py            (forked from Exp-14/scripts/ensemble_v2.py)
└── slurm/
    └── ablation_no_convnext.sbatch     (full pipeline: Step A + Step B + submit)
```

## How to run on PACE

```bash
cd /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Abl-1_CNext
sbatch slurm/ablation_no_convnext.sbatch
```

Output CSVs land at:
```
${PLANTCLEF_ROOT}/aerten3/submissions/abl1_no_cnxt/
  abl1_simw_b0.70_w1.00_thr0.085_top10.csv     ← bare ViT-only, no habitat-fit
  abl1_s9_B_R250.csv                            ← HEADLINE: full stack minus ConvNeXt
```

Submit `abl1_s9_B_R250.csv` to Kaggle, record both public and private F1.

## Decision rule

- If |private F1(abl1) − 0.44649| < 0.003: ConvNeXt contribution is
  sub-noise. **Drop ConvNeXt from the paper pipeline.** Final reported
  configuration becomes ViT-L + kNN + temporal-fusion + masks + habitat-fit.
- If private F1(abl1) is lower by ≥ 0.003: ConvNeXt is load-bearing
  despite small w. Keep it; report the ablation as a non-trivial component.

## Expected wall-time and cost

| Phase | Cost |
|---|---|
| Step A (ViT-only fusion + save parquet) | ~15–30 min CPU |
| Step B (habitat-fit + admission + CSV) | ~5–10 min CPU |
| GPU | 0 |
| LB submissions | 1 (just `abl1_s9_B_R250.csv`) |
| Total wall-time | < 1 hour |
