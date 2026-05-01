# Exp-14 Stage 13 — Per-source biogeographic-impossibility mask

**Status:** LB-tested 2026-04-27. **F1 = 0.45611** (Δ = +0.00039 vs Stage 12 0.45572). Sub-noise on LB; adopted as new best based on stable_site_invariance proxy improvement of +0.0029 on the full 2 105-image cohort. Stop-rule honored: species list stays at 5.
**Mechanism (one sentence):** *On the alpine sources (CBN-PdlC, CBN-Pla), zero five hand-vetted species whose biogeographic distribution excludes alpine summit habitat.*

## Motivation — gap left by Stage 12

The Stage 12 winner (`v_ld12RNNBonly_habitat_w0.77_thr0.100_top10.csv`,
LB = 0.45572) zeroed 91 fully-aquatic species globally (Posidonia,
Zostera, Cymodocea, Lemna, Potamogeton, etc.). Re-auditing the locked
winner CSV against the species_id_to_name list surfaced five species
that were predicted only on CBN-PdlC (no aquatic genus/epithet match,
so they survived Stage 12) but are biogeographically impossible at
alpine summit elevation.

| sid | class_idx | Name | Stage 12 alpine count | Why out of place |
|---|---|---|---|---|
| 1402995 | 5524 | *Axonopus compressus* | 21 | Tropical American C4 carpetgrass; not naturalized in temperate alpine. |
| 1706074 | 6797 | *Equisetum × moorei* | 51 | Coastal/dune horsetail hybrid (E. hyemale × E. ramosissimum). |
| 1391092 | 3192 | *Cerastium tomentosum* | 21 | Mediterranean lowland-to-montane garden escape; native Apennine mid-elevations. |
| 1413349 | 5574 | *Armeria maritima* | 21 | Coastal sea-thrift; alpine taxon is var./subsp. *alpina*, not the species *A. maritima*. |
| 1363741 | 2554 | *Trifolium suffocatum* | 26 | Mediterranean coastal annual on sandy/disturbed soils. |

All five were CBN-PdlC-only in Stage 12 (no occurrences on CBN-Pla,
RNNB, or any other source). The mask is therefore deterministic and
per-source: alpine sources zero these five class indices; all other
sources are unchanged.

## Files

```
Exp-14_Adaptive_Temporal_Fusion/
├── signals/build_biogeo_mask.py             (npz with mask_alpine, mask_default)
├── scripts/apply_biogeo_mask.py             (per-source dispatch on top of geo + aquatic)
├── outputs/signals/biogeo_per_source.npz    (5 alpine-blocked classes)
├── outputs/signals/biogeo_blacklist.json    (audit: sid, name, rationale, blocked_on)
└── outputs/stage13_biogeo_mask/
    ├── v_ld12RNNBonly_habitat_biogeo_w0.77_thr0.100_top10.csv
    └── proxies.json
```

## Diff from Stage 12 winner

| Quantity | Value |
|---|---|
| Filled prediction slots | 16 261 (unchanged from Stage 12) |
| Set-membership changes | 145 quadrats (142 CBN-PdlC + 3 CBN-Pla) |
| Pure-reorder (no set change) | 21 quadrats (argsort tie-flips at sub-threshold) |
| Target species zeroed | 140 slot-removals (21+51+21+21+26) replaced by next-best species |

The 3 CBN-Pla set changes are noise-floor argsort instability when
sub-threshold values change (none of the 5 target species were in any
CBN-Pla top-10 in Stage 12); they swap a single position-10 species
near the threshold cutoff. Negligible expected F1 impact.

## Proxy-gate (vs Stage 12 baseline)

| Metric | Stage 12 | Stage 13 | Δ | Direction |
|---|---|---|---|---|
| `rnnb_pair_divergence` | 0.9207 | 0.9207 | +0.0000 | unchanged (mask doesn't touch RNNB) |
| `stable_site_invariance` | 0.2483 | 0.2512 | **+0.0029** | improved (alpine within-site coherence up) |
| `density_count_correlation` | 0.2603 | 0.2603 | +0.0000 | unchanged (count distribution preserved) |
| Per-source mean count | identical | identical | — | CBN-PdlC: 10.00, CBN-Pla: 9.77, RNNB: 2.84 |

No regression on any axis; alpine within-site Jaccard improves by
+0.003 (the mask removes inconsistent species from CBN-PdlC clusters,
which makes paired/repeated visits more coherent). Count-magnitude
proxies all stable — passes the binding count-magnitude requirement
from Exp-14 lessons.

## Wiring sanity check

`apply_biogeo_mask.py` was run with an all-True (empty) biogeo mask
against the same Stage 11 ensemble parquet; output CSV is byte-for-byte
identical to the locked Stage 12 winner. Graceful-fallback verified:
turning the mechanism off recovers the prior winner exactly.

## Why the mechanism is defensible (anti-LB-overfit)

1. **Mechanism in one sentence** — yes (above).
2. **Hand-vetted species list, not pattern-fishing** — five species
   with explicit biogeographic rationale; not derived by sweeping
   thresholds or per-source parameter tuning.
3. **Per-source restriction is principled** — the same species' alpine
   prediction is wrong; their RNNB/coastal prediction *might* be right
   (e.g., *Armeria maritima* on RNNB), so a global blacklist would
   over-correct. The mask blocks only on CBN-PdlC and CBN-Pla.
4. **Count-magnitude metrics preserved** — per-source mean species/quadrat
   identical to Stage 12; the failure mode that crashed Stage 1 global
   sim_w (alpine count collapsing 10 → 5) does not occur here.
5. **Graceful fallback** — empty mask reproduces Stage 12 byte-for-byte.

## What this is not

- Not an iterative tune. List size 5 is fixed; if Stage 13 misses LB,
  we do not expand the list. (Going to 10/15 species would be the LB-fitting
  failure mode the binding feedback prohibits.)
- Not a model change. Pure post-processing on a saved probability parquet.
- Not a temporal-fusion or aggregator change. The Stage 4 + Stage 12
  mechanisms (per-source sim_w on RNNB, ViT α=0.30 on RNNB, aquatic
  blacklist) remain in force.

## LB outcome

**F1 = 0.45611, Δ = +0.00039 vs Stage 12 0.45572.** Sub-noise on the
232-image public LB sample (noise floor ±0.005), so LB alone cannot
confirm a real gain. The pre-submission mechanism prediction was
Δ ∈ [+0.000, +0.005]; the realized lift is on the lower edge of that
band, consistent with the hypothesis being weakly correct (most of the
target species removed *were* wrong predictions, but not all).

The +0.00039 LB delta + the +0.0029 stable_site_invariance improvement
on the full 2 105-image cohort + zero regression on any other proxy
collectively justify adoption of Stage 13 as the new best. The full-cohort
proxy is higher-resolution evidence than the 11 % LB sample.

**Stop rule honored:** species list stays at 5. No expansion to 10/15.
Any future biogeographic mask additions need an independent mechanism
(e.g., RNNB-source list with its own rationale), not enlargement of
the alpine list.

## Reproduction

```bash
cd Exp-14_Adaptive_Temporal_Fusion

# 1. Build the per-source biogeo masks (deterministic, ~1 sec)
python signals/build_biogeo_mask.py \
    --species-map pace_downloads/class_to_species_id.json \
    --out-npz outputs/signals/biogeo_per_source.npz \
    --out-list-json outputs/signals/biogeo_blacklist.json

# 2. Apply geo + aquatic + per-source biogeo, emit Stage 13 CSV
python scripts/apply_biogeo_mask.py \
    --ensemble-parquet outputs/stage11_avg/v_ld12RNNBonly_w0.77_ensemble.parquet \
    --species-map pace_downloads/class_to_species_id.json \
    --geo-mask pace_downloads/sw_europe_mask_bool.npy \
    --habitat-mask outputs/signals/aquatic_blacklist_mask.npy \
    --biogeo-npz outputs/signals/biogeo_per_source.npz \
    --out-csv outputs/stage13_biogeo_mask/v_ld12RNNBonly_habitat_biogeo_w0.77_thr0.100_top10.csv \
    --threshold 0.100 --topk 10

# 3. Proxy gate vs Stage 12
python diagnostics/proxy_metrics.py \
    --csv outputs/stage13_biogeo_mask/v_ld12RNNBonly_habitat_biogeo_w0.77_thr0.100_top10.csv \
    --baseline-csv outputs/stage12_habitat_mask/v_ld12RNNBonly_habitat_w0.77_thr0.100_top10.csv \
    --density outputs/signals/density.parquet \
    --out-json outputs/stage13_biogeo_mask/proxies.json
```
