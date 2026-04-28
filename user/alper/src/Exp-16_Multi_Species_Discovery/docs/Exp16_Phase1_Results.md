# Exp-16 Phase 1 — Vote-share rerank (closed: negative result)

**Status:** closed 2026-04-27. Mechanism verified to be real (per-image
species count rises monotonically with β on count-bound sources), but
all three internal proxies regress monotonically. No (β, K) variant
passes the gates. **0 LB submissions made.**

## Wiring sanity

β=0 reproduces the reference baseline byte-for-byte (the reference is
`ensemble_v2.py + apply_biogeo_mask.py` run on the original
`knn_bM0.85_bS1.00_T1.50_prefusion.parquet` with the same args used for
Stage 13). Δ on all proxies = +0.0000. Wiring verified.

> Caveat: the *reference* differs from the LB-locked Stage 13 winner by
> 25 set-membership rows (98.8 % of 2 105 quadrats match exactly). The
> historical args that produced `outputs/stage11_avg/v_ld12RNNBonly_w0.77_ensemble.parquet`
> aren't reproducible from the current scripts; the rest of Phase 1
> compares against my reference run rather than the LB-locked CSV.

## Sweep results (vs reference baseline)

### K = 5 (top-5 species per tile)

| β | RNNB n | LISAH n | OPTMix n | GUARDEN n | Δ rnnb_pair_div | Δ stable_site_inv | Δ density_corr |
|---|---|---|---|---|---|---|---|
| 0.00 | 2.84 | 3.83 | 1.82 | 2.72 | +0.0000 | +0.0000 | +0.0000 |
| 0.25 | 3.55 | 4.22 | 2.17 | 3.00 | −0.0628 | −0.0075 | −0.0082 |
| 0.50 | 4.13 | 4.47 | 2.41 | 3.12 | −0.1279 | −0.0126 | −0.0088 |
| 0.75 | 4.42 | 4.61 | 2.62 | 3.19 | −0.2040 | −0.0156 | −0.0088 |
| 1.00 | 4.49 | 4.64 | 2.74 | 3.19 | −0.2648 | −0.0223 | −0.0114 |

### K = 10 (top-10 species per tile)

| β | RNNB n | LISAH n | OPTMix n | GUARDEN n | Δ rnnb_pair_div | Δ stable_site_inv | Δ density_corr |
|---|---|---|---|---|---|---|---|
| 0.25 | ~3.1 | ~4.1 | ~2.0 | ~2.9 | −0.0700 | −0.0035 | −0.0155 |
| 0.50 | 4.18 | 4.48 | 2.37 | 3.10 | −0.1005 | −0.0106 | −0.0094 |
| 1.00 | 4.59 | 4.77 | 2.76 | 3.17 | −0.1881 | −0.0185 | −0.0135 |

K=10 is slightly less destructive than K=5 at the same β (broader top-K
includes more low-rank species, so the suppression of medium-rank
species is gentler), but still negative on every axis.

## What the mechanism does

Empirically:
1. Per-image species count *rises* monotonically with β on the
   count-bound sources (RNNB, GUARDEN, LISAH, OPTMix). Alpine sources
   are unchanged because they were already saturated at top-10.
2. Pair divergence on RNNB *drops* monotonically (paired visits
   converge on shared species). This is the same homogenization failure
   mode that crashed Exp-14 Stage 1 global sim_w (−0.010 LB).
3. Stable-site invariance on alpine drops slightly (alpine clusters
   become slightly less coherent).

So the rerank surfaces *more* species per image — but the species it
surfaces tend to be **shared across paired visits** rather than
visit-specific. The L1-rescale step amplifies whichever classes have
non-trivial probability *and* non-trivial vote share, which biases
toward species that are recurring across all visits at the same site
(i.e. the location prior's most-favored species).

## Why this doesn't address the user's question

The user's concern is "discover small species visible in the photo but
overpowered by the dominant species." Vote-share rerank can only
redistribute mass *within the model's existing per-tile predictions*.
If the model never assigned non-trivial probability to a small species
on any tile (because the occluded view was too OOD for ViT-B that
trained only on close-up single-species photos), no amount of rerank
will surface it.

Concretely: a small species genuinely visible in the photo but assigned
< 0.001 probability on every one of the 77 tiles will have vote_share
≈ 0 (never in top-K of any tile) AND p_prefusion < 0.001 → rerank
factor ≈ ε^β ≈ 0.1, multiplied by 0.001 = 0.0001 → still far below
threshold 0.100.

The species the rerank *does* boost are the ones the model already
saw on multiple tiles at moderate confidence. These tend to be either
(a) genuine dominant species — already in top-10 — or (b) site-typical
species (in the location prior). Neither is the "small unseen species"
the user wants.

This is a **backbone-level limitation**, not an aggregator-level one.
The fix has to come at the training-data / augmentation level (Phase 3
ViT CutMix re-finetune), not at post-hoc reweighting.

## Decision

- **Phase 1 closes with negative result.** No LB submissions made.
- **Move to Phase 3 (ViT mixup re-finetune)** as the next experiment in
  Exp-16. Skip Phase 2 (per-image ensemble weight schedule) on the same
  reasoning: it can only redistribute existing per-image predictions
  between branches, which doesn't address the root cause (the model
  hasn't learned to recognize occluded small species).

## Files retained

```
Exp-16_Multi_Species_Discovery/outputs/phase1_vote_rerank/
├── _reference_baseline/
│   ├── ref_w0.77_ensemble.parquet
│   └── ref_habitat_biogeo_w0.77_thr0.100_top10.csv
├── vote_share_K5/vote_share.parquet     (re-usable for any future rerank experiment)
├── vote_share_K10/vote_share.parquet
├── K{5,10}_b{0.00,0.25,0.50,0.75,1.00}/
│   ├── vit_prefusion_reranked.parquet
│   ├── K{K}_b{β}_w0.77_ensemble.parquet
│   ├── K{K}_b{β}_habitat_biogeo_w0.77_thr0.100_top10.csv
│   └── proxies.json
└── phase1_summary.json
```
