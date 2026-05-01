# Exp-16 Phase 1 — Vote-share rerank (CONFIRMED NEGATIVE 2026-04-30 after LB re-test)

**Status:** original closure 2026-04-27 (proxy-negative); re-tested
2026-04-30 on LB after N3_r0.5's surprise win raised the question of
whether proxies had under-predicted Phase 1 too. **LB conclusively
disconfirmed any Phase 1 gain: all 4 β variants (0.10, 0.25, 0.50, 1.00)
regress monotonically with β.** Even the most conservative β=0.10 loses
0.019 LB. Compose with N-floor admission also loses 0.024 LB. The
proxy-regime caveat does NOT extend to Phase 1.

## LB results (post-blend formulation, byte-equivalent to Stage 13 at β=0)

| Variant | LB F1 | Δ vs Stage 13 (0.45611) |
|---|---|---|
| postblend_K5_b0.10 | 0.43732 | −0.01879 |
| postblend_K5_b0.25 | 0.41577 | −0.04034 |
| postblend_K5_b0.50 | 0.38836 | −0.06775 |
| postblend_K5_b1.00 | 0.35718 | −0.09893 |
| compose_b0.10_N3r0.5 | 0.43203 | −0.02408 |
| compose_b0.25_N3r0.5 | 0.40770 | −0.04841 |
| compose_b0.50_N3r0.5 | 0.38029 | −0.07582 |

## Why Phase 1 regresses where N3_r0.5 won

Phase 1 vote-share rerank is **global** — it multiplies the entire
ensemble probability vector by `(vote_share + ε)^β`, which reshuffles
the alpine top-10 sets even at β=0.10 (CBN-PdlC: 639 / 816 set-diffs;
CBN-Pla: 542 / 628 set-diffs at β=0.10 alone).

Alpine drives 69 % of test set predictions (1444 / 2105 quadrats). Any
disturbance there costs F1 heavily because the alpine pipeline was
load-bearing for prior gains. The vote-share signal is not a clean
discriminator on alpine quadrats — alpine vote_shares are dominated by
the same site-typical species across paired visits, so multiplying by
vote_share^β homogenizes alpine in a way that hurts the F1 score.

N3_r0.5 won because it was **scoped to count-bound sources only**
(GUARDEN, LISAH, OPTMix, RNNB) — alpine sources are untouched by N-floor
admission (Δ stable_site_invariance was exactly +0.0000 across all
N-floor variants).

## Sharpened proxy-regime caveat

The "proxies under-predict count-bound source uplift" caveat from N3_r0.5
applies only when the mechanism is **scoped to where the proxies are
saturated**. Mechanisms that ALSO affect alpine (where proxies are
well-calibrated) cannot invoke this caveat. Future post-hoc admission
mechanisms must be source-scoped to the count-bound sources to qualify
for the caveat treatment.

Lesson recorded: don't generalize one mechanism's proxy-disconfirmation
to other mechanisms with different source scope.

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
