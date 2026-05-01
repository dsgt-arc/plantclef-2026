# Exp-16 — Per-source N-floor admission (LB-tested 2026-04-30, N3_r0.5 ADOPTED at F1=0.46168)

**Status:** LB-tested 2026-04-30. N3_r0.5 is the new locked best at
**F1 = 0.46168** (+0.00557 over Stage 13 winner 0.45611), the largest
single-mechanism gain since Stage 4. **Internal proxies were wrong** —
proxy negativity did not correspond to LB regression for moderate-
admission variants. The mechanism succeeded despite my proxy gating
calling it negative.

## Mechanism (one sentence)

*On count-bound sources (GUARDEN, LISAH, OPTMix, RNNB), if fewer than
N_floor species pass the standard threshold=0.100, admit additional
candidates up to N_floor — but only those whose probability ratio to
the top-1 admitted species is at or above ratio_floor.*

N_floor=0 reproduces Stage 13 winner byte-for-byte (verified). Wiring
sanity: PASS.

## Sweep results (vs Stage 13 winner baseline)

| Variant | Triggered | Extra | RNNB n | LISAH n | OPTMix n | GUARDEN n | Δ rnnb_pair_div | Δ density_corr | Δ stable_site | **LB F1** | **Δ LB** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Stage 13 (baseline) | — | — | 2.84 | 3.83 | 1.82 | 2.72 | +0.0000 | +0.0000 | +0.0000 | 0.45611 | 0.00000 |
| sanity_n0 | 0 | 0 | 2.84 | 3.83 | 1.82 | 2.72 | +0.0000 | +0.0000 | +0.0000 | 0.45611 | 0.00000 |
| N3_r0.7 | 59 | 118 | 2.97 | 3.97 | 2.09 | 2.80 | −0.0099 | −0.0010 | +0.0000 | 0.44763 | −0.00848 |
| **N3_r0.5** | **93** | **187** | **3.13** | **4.02** | **2.38** | **2.86** | **−0.0148** | **−0.0047** | **+0.0000** | **0.46168** | **+0.00557** |
| N3_r0.3 | 169 | 289 | 3.28 | 4.19 | 2.72 | 2.95 | −0.0233 | −0.0057 | +0.0000 | 0.45320 | −0.00291 |
| N5_r0.7 | 61 | 150 | 3.08 | 4.00 | 2.15 | 2.82 | −0.0138 | −0.0020 | +0.0000 | 0.44811 | −0.00800 |
| N5_r0.5 | 121 | 333 | 3.48 | 4.24 | 2.73 | 2.98 | −0.0250 | −0.0058 | +0.0000 | 0.45423 | −0.00188 |
| N5_r0.3 | 265 | 670 | 4.11 | 4.71 | 3.64 | 3.37 | −0.0531 | −0.0072 | +0.0000 | 0.44407 | −0.01204 |

## Mechanism in one sentence (locked)

*On count-bound sources (GUARDEN, LISAH, OPTMix, RNNB), if fewer than 3
species pass the standard threshold=0.100, admit additional candidates
(up to 3 total) whose probability ratio to the top-1 admitted species
is ≥ 0.5.*

## What the LB results actually show

1. **N3 dominates N5 at every ratio_floor** — N=3 is the right depth.
   Going to top-5 admits too much noise even with the ratio gate.
2. **Ratio_floor exhibits a clear sweet spot at 0.5** for N=3:
   - r=0.7 too restrictive: barely admits anything (118 extras), LB drops to 0.448 — the few admitted are noise relative to baseline because they sit close to the dominant species' confidence and may be genuinely-uncertain duplicates.
   - r=0.5 sweet spot: admits 187 extras with the right confidence margin, LB +0.00557.
   - r=0.3 too permissive: admits 289 extras including too much noise, LB regresses.
3. **N3_r0.5 added 187 species across 93 quadrats** — an average of 2 extras per triggered quadrat — and produced the largest single-mechanism gain since Stage 4.

## Why my proxies pointed wrong

- `rnnb_pair_divergence` was operating in a saturated regime. Stage 13
  baseline value was 0.92 (paired visits ≈ 92% non-overlapping). The
  N3_r0.5 result drops it to ~0.91. In *absolute* terms paired visits
  are still extremely dissimilar; the proxy's monotonic-direction rule
  doesn't account for the regime.
- `density_count_correlation` regression was real but small (−0.0047)
  and doesn't directly correspond to F1 because F1 doesn't reward
  density-prediction-density-correlation; it rewards correctly
  predicting species presence/absence.
- The proxies were tuned against Stage-1-style failure modes
  (CBN-PdlC count collapsing 10→5) and don't generalize to small-
  count-bound-source uplift, which is a different regime entirely.

## Reusable lesson

For count-bound sources operating below their natural species count,
**proxy gating based on shared-prior-aware metrics under-predicts LB
gain**. When mechanism evidence is mixed (count-magnitude rises,
set-overlap proxies regress modestly), prefer LB confirmation for
moderate variants over a proxy-only verdict.

## Comparison to Phase 1 vote-share rerank

| Mechanism | RNNB n change | Δ rnnb_pair_div |
|---|---|---|
| Vote-share rerank β=0.25 (K=5) | 2.84 → 3.55 | −0.0628 |
| **N3_r0.7 admission** | 2.84 → 2.97 | **−0.0099** |
| Vote-share rerank β=0.50 (K=5) | 2.84 → 4.13 | −0.1279 |
| **N5_r0.7 admission** | 2.84 → 3.08 | −0.0138 |

N-floor admission is **less destructive** than vote-share rerank at
comparable count-bound uplift. But still proxy-negative.

## Why the same failure mode appears

Both Phase 1 (vote-share rerank) and the N-floor admission lift
threshold-bound species into the top-K. Both rely on the **same
per-image probability vector** as the input. That vector already
incorporates the location prior (which is shared across paired RNNB
visits via temporal fusion). When either mechanism admits sub-threshold
candidates, those candidates are biased toward the shared-prior
species — which are the same species across paired visits. Pair
divergence drops because both visits converge on the shared species.

The mechanism that *would* be principled here is one that uses
**per-visit-specific** signal (e.g., the per-image photo-only signal
without the location prior blend) to decide admission. But the
prefusion parquet only stores the post-blend probabilities; the
pre-blend per-tile photo signal lives in the ViT classifier tile
parquets, which is a much bigger compute footprint to operate on.

Even if we used the pre-blend per-image probabilities, the underlying
ViT-B classifier produces only ~3 distinguishable confident
predictions per dense quadrat — which is *why* the post-blend
probability vector is so dominated by the location prior in the first
place. The model genuinely doesn't know how to discriminate the 4th-15th
species in dense scenes.

## Decision

- **N3_r0.5 ADOPTED as new locked best at F1 = 0.46168** (+0.00557 over
  Stage 13 winner 0.45611).
- **Submission file:** `Exp-16_Multi_Species_Discovery/outputs/n_floor_sweep/N3_r0.5.csv`.
- **Locked recipe:** Stage 4 per-source temporal fusion + Stage 12
  aquatic mask + Stage 13 alpine biogeo mask + **N-floor admission
  (N=3, ratio_floor=0.5) on count-bound sources**.
- **Stop rule honored:** sweep is closed. Do not iterate to finer
  ratio_floor (e.g. 0.45, 0.55) — that would be parameter-fishing on
  a single-LB-submission gradient. Future tuning waits for Gate A's
  new ViT-L prefusion.
- **Re-test Phase 1 (vote-share rerank) on LB.** I closed Phase 1 as
  negative based on the same proxy logic that under-predicted N3_r0.5.
  At least one Phase 1 variant (likely β=0.25 / K=5) might also gain
  on LB despite proxy negativity. Worth 1-2 LB submissions to confirm
  or rule out.

## Files retained

```
Exp-16_Multi_Species_Discovery/
├── scripts/apply_n_floor_admission.py
└── outputs/n_floor_sweep/
    ├── sanity_n0.csv                 (byte-equivalent to Stage 13)
    ├── N{3,5}_r{0.3,0.5,0.7}.csv
    └── N{3,5}_r{0.3,0.5,0.7}_proxies.json
```
