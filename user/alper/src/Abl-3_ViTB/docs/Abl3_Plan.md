# Abl-3 — Backbone Knockout: ViT-B vs ViT-L

## Reference

| Run | Private F1 | Public F1 |
|---|---|---|
| B_R250 (full, ViT-L + ConvNeXt + kNN) | 0.44649 | 0.48414 |
| Abl-1 (− ConvNeXt) | 0.45001 | 0.47136 |
| Abl-2 bare (− ConvNeXt − kNN, no habitat-fit) | 0.38702 | 0.42181 |
| Abl-2 full (− ConvNeXt − kNN, full downstream) | 0.44053 | 0.44953 |
| **Abl-3 bare (− ConvNeXt − kNN, ViT-B)** | TBD | TBD |
| **Abl-3 full (− ConvNeXt − kNN, ViT-B, full downstream)** | TBD | TBD |

Cumulative knockout: same as Abl-2 (no kNN, no ConvNeXt), but the ViT-L
backbone is swapped for ViT-B. Two CSVs are produced so you can compare at
two depths.

## What this ablation isolates

Just the backbone (ViT-Base vs ViT-Large). Everything downstream of the
per-tile classifier outputs is held fixed: same temporal fusion controller,
same sim_w-on-RNNB aggregator, same masks, same habitat-fit knobs, same
per-source admission. The only thing that changed between Abl-2 and Abl-3
is which model produced the per-tile `teacher_logits_f16`.

| Layer | Setting |
|---|---|
| Backbone tile parquets | **ViT-Base** (Exp-7 extraction), scales 4,5,6,7 |
| ViT-L equivalent (Abl-2) | scales 4,5,6,7 from `exp11c_vitL_lora_arcface/embeddings/test/` |
| kNN retriever | OFF (β_match → 1.0) |
| ConvNeXt branch | REMOVED |
| Temporal fusion | RNNB α=0.30 controller + `{"RNNB":"sim_w"}` |
| Threshold / topk | 0.085 / 10 |
| Aquatic + biogeo masks | applied |
| Habitat-fit | β=0.10, R=250, n=40, altitude on |
| Per-source admission | OPTMix n_floor=2, LISAH n_floor=2 |

## Why no fork is needed

All three Python scripts in the pipeline read only `image_name` and
`teacher_logits_f16` (plus optionally `emb_arcface_f16`, which we don't use
here since kNN is off). Both backbones produce parquets with the same
schema, so:

- `Abl-2/scripts/make_vitL_prefusion_no_knn.py` — backbone-agnostic
- `Abl-1/scripts/ensemble_vit_only.py` — backbone-agnostic
- `Exp-16/scripts/apply_habitat_fit.py` — operates on the ensemble parquet, doesn't know which backbone produced it

Abl-3 is therefore just a different invocation of the same pipeline with
different input paths. No new Python code.

## Methodology disclaimer (paper caption text)

Fixed-knob ablation. The temporal-fusion controller `vit_rnnb030.json`
(α=0.30 for RNNB), the kNN β_match (off, but production β_match=0.85 for
ViT-B vs 0.70 for ViT-L was independently tuned per backbone), the habitat-fit
knobs (β=0.10, R=250, n=40), and the per-source admission floors were all
tuned with the ViT-L backbone. Re-applying them unchanged to ViT-B is a
fixed-knob comparison; the reported ΔF1 is a lower bound on the marginal
cost of using the smaller backbone.

## Files in this folder

```
Abl-3_ViTB/
├── docs/
│   └── Abl3_Plan.md                       (this file)
├── scripts/
│   └── run_local.ps1                       (local PowerShell driver — no SLURM)
└── outputs/                                (created on first run)
    ├── prefusion/
    │   └── abl3_vitB_noknn_prefusion.parquet
    ├── per_src_configs/
    │   └── locked.json
    ├── abl3_vitB_noknn_w1.00_ensemble.parquet
    ├── abl3_vitB_noknn_w1.00_thr0.085_top10.csv   ← BARE → submit
    └── abl3_s9_B_R250_vitB.csv                     ← FULL → submit
```

Reuses `Abl-1/scripts/ensemble_vit_only.py`, `Abl-2/scripts/make_vitL_prefusion_no_knn.py`,
and `Exp-16/scripts/apply_habitat_fit.py` — no new Python scripts.

## How to run (laptop, no SLURM)

```powershell
pwsh .\Abl-3_ViTB\scripts\run_local.ps1
```

Three sequential Python invocations:
- Step A: ViT-B prefusion build (~10–20 min on laptop CPU; ViT-B has fewer tiles than ViT-L at 4×4 due to JPEG path differences — wall-time roughly similar to Abl-2 prefusion)
- Step B: ensemble fusion (~5–10 min)
- Step C: habitat-fit + admission (~5–10 min)

Total wall-time: ~30 min CPU. No GPU. No LB submission until you upload the two CSVs.

## Decision rule

Compare against **Abl-2** (the matched-pipeline ViT-L anchor):

**At the bare level (multi-scale + temporal fusion only):**
- Abl-3 bare vs Abl-2 bare 0.38702 private → isolates the pure backbone-quality effect on raw classifier outputs

**At the full-stack level (with masks + habitat-fit + admission):**
- Abl-3 full vs Abl-2 full 0.44053 private → answers "did the backbone upgrade help the production-style pipeline?"

Expected: ViT-L should beat ViT-B at both depths. Open question is whether
the gap at the full-stack level (where downstream tricks compensate) is
smaller than the gap at the bare level. If yes → "downstream is partly
substituting for backbone quality." If no → "backbone quality is multiplicative
with downstream gains."

## Expected wall-time and cost

| Phase | Cost |
|---|---|
| Step A (no-kNN prefusion from ViT-B tiles) | ~10–20 min laptop CPU |
| Step B (ViT-only fusion + ensemble parquet) | ~5–10 min CPU |
| Step C (habitat-fit + admission + CSV) | ~5–10 min CPU |
| GPU | 0 |
| LB submissions | 2 (bare + full) |
| Total wall-time | < 45 minutes |
