# Exp-16 — Habitat-fit demotion (final results, 2026-04-30)

**Final F1 = 0.48529** with `hab_refined_R200_b0.25_n30_alt.csv`.
+0.02361 over Stage 13 + N3_r0.5 baseline (F1 = 0.46168). Single largest
improvement in the project. β=0.15 + n=30 sits at 0.48705 (+0.00176)
but adjacent to a structural cliff at β=0.10 (0.47891); locked at
β=0.25 as the more robust mid-plateau pick.

The doc below captures only the four experiments that actually moved
F1 — the dead-end mechanisms (predicted-habitat, single-visit prior,
genus dedup, veg-filter) are documented in their own files.

## Mechanism (one sentence)

For each test image, multiply each candidate species' probability by a
habitat-fit factor in [β, 1] derived from how well the species'
PlantCLEF2024 training-data geographic distribution overlaps with the
test image's source's known regional locale, then run the existing
N-floor admission on the demoted probabilities.

The factor uses external information — training-data lat/lon/altitude —
that the model never saw during inference. Cosmopolitan species are not
penalized; range-restricted species observed only in distant regions
are.

## Step-by-step recipe (the winner)

1. **Build per-species geographic profile** from PlantCLEF2024 training
   metadata (`PlantCLEF2024singleplanttrainingdata.csv`, 1.4 M rows).
   For each species_id, store: median lat/lon/altitude, n_obs, and per-
   source `n_near_<source>_refined_R200` = count of training observations
   within 200 km of the source's refined centroid.
   Code: `diagnostics/build_species_geo_profile.py`,
         `diagnostics/refine_source_centroids.py`,
         `diagnostics/add_radius_columns.py`.

2. **Refined source centroids.** For each of the 7 test sources, take
   the top 30 most-frequently-predicted species in the locked Stage 13
   + N3_r0.5 winner; drop any species also in the top-30 of ≥3 other
   sources (cosmopolitans). Compute the median (lat, lon, alt) of the
   remaining anchor-species' SW-Europe-bounded training observations.
   These replace hand-set centroids. Stored in
   `outputs/habitat_classifier/refined_source_ref.json`.

   | Source | Refined centroid | Alt range (p10–p90) |
   |---|---|---|
   | RNNB | 43.21°N, 2.36°E | 4–453 m |
   | CBN-PdlC | 45.21°N, 6.52°E | 415–2632 m |
   | CBN-Pla | 45.16°N, 6.60°E | 924–2753 m |
   | GUARDEN | 43.35°N, 2.89°E | 7–698 m |
   | LISAH | 45.00°N, 2.67°E | 34–697 m |
   | OPTMix | 45.78°N, 2.81°E | 45–1277 m |
   | CBN-can | 45.73°N, 5.97°E | 82–2367 m |
   | 2024-CEV3 | 46.23°N, 5.40°E | 66–1777 m |

3. **Per (test image × candidate species) habitat-fit factor.** For
   each test image, look up source from filename. For each candidate
   species, compute:

   ```
   geo_score = β + (1 − β) × min(1.0, n_near_<source>_refined_R200 / n_near_min)
   alt_score = 1.0 if species' median altitude is within source's
               [alt_lo − margin, alt_hi + margin] band
               (margin = 0.5 × (alt_hi − alt_lo))
             else max(β, 1 − distance / 1000m)
   habitat_fit = geo_score × alt_score
   ```

   With locked params β = 0.25, n_near_min = 30:
   - A species with ≥30 training obs within 200 km of source centroid →
     geo_score = 1.0 (no penalty).
   - A species with 0 nearby training obs → geo_score = 0.25 (75 %
     reduction).
   - Linear interpolation between.
   - Altitude axis adds an analogous penalty for species whose
     training-data median altitude is outside the source's expected band.
   - Species with no training lat/lon (n_obs = 0): habitat_fit = 1.0
     (neutral, no info).

4. **Demote and re-admit.** Multiply per-class probability by
   habitat_fit. Apply geo + aquatic + per-source biogeographic mask.
   Run N-floor admission with N_floor = 3, ratio_floor = 0.5 on count-
   bound sources. Sort by quadrat_id, write CSV.

   Code: `scripts/apply_habitat_fit.py` with
   `--refined-ref-json refined_source_ref.json --radius-km 200
    --beta 0.25 --n-near-min 30 --use-altitude`.

## What worked — sweep summary

The mechanism was tuned across 4 axes. Each line below is the LB
result that motivated the next axis. Negative results within each
sweep are recorded only at the bracket edges where they help confirm
the peak.

### 1. Initial mechanism + parameter location

The first habitat-fit sweep used hand-set centroids and proximity
radius 250 km. The relevant winners on the way to the locked best:

| β | n_near_min | altitude | LB F1 |
|---|---|---|---|
| 1.0 (parity) | — | — | 0.46168 |
| 0.85 | 5 | off | 0.46444 |
| 0.7 | 5 | off | 0.46790 |
| 0.5 | 5 | off | 0.46877 |
| 0.5 | 10 | off | 0.47212 |
| 0.5 | 20 | on | 0.47383 |
| 0.5 | 25 | on | 0.47438 |
| 0.4 | 25 | on | **0.47592** |

Mechanism is real (β=0.85 → 0.5 monotone with stronger demotion;
n_near_min=5 → 10 → 20 → 25 monotone with stricter inclusion).
Adding altitude lifts ~+0.0004 at the peak.

### 2. Refined source centroids (Test 2)

Replacing hand-set lat/lon centroids with anchor-species-derived
centroids shifted the n optimum and lifted the peak.

| Centroids | β | n_near_min | LB F1 |
|---|---|---|---|
| Hand-set | 0.4 | 30 | 0.47292 |
| Refined | 0.4 | 30 | **0.47951** (+0.00659) |

The proper head-to-head at b0.4_n30_alt: refined wins by +0.00659,
well above noise. Most striking centroid shift: CBN-PdlC moved from
hand-set Pyrenees (42.7, 0.5) to refined Alps (45.21, 6.52) because
anchor species like Salix herbacea have Alps-heavy training data
despite PdlC being a Pyrenees source.

### 3. Proximity radius (Test 3)

At locked refined β=0.4, n=30, alt:

| Radius | LB F1 |
|---|---|
| 150 km | 0.46413 (too tight) |
| **200 km** | **0.48210** (+0.00259) |
| 250 km | 0.47951 (anchor) |
| 300 km | 0.47441 |
| 500 km | 0.47180 (too lax) |

Clean inverted-U with sharp peak at 200 km — roughly the scale of a
French biogeographic region (Pyrénées-Orientales, Mediterranean
coastal strip, Alps massif). Tighter cuts into legit species; looser
admits out-of-region species into the "in-region" set.

### 4. β re-tuning at R=200

The radius change shifted the β optimum lower because tighter R puts
more species in the demotion regime, where stronger demotion has more
leverage.

| β | n_near_min | LB F1 | Note |
|---|---|---|---|
| 0.50 | 30 | 0.48021 | |
| 0.40 | 30 | 0.48210 | (anchor at the new R) |
| 0.30 | 30 | 0.48497 | |
| **0.25** | **30** | **0.48529** | ← LOCKED |
| 0.20 | 30 | 0.48371 | |
| 0.15 | 30 | 0.48705 | (cliff-adjacent) |
| 0.10 | 30 | 0.47891 | structural cliff |

The β = 0.10 cliff (0.008 drop from β = 0.15) confirms that out-of-
region species need *some* probability mass — over-demoting removes
legitimate predictions.

β = 0.15 sits at the cliff edge with the highest LB but the largest
private-LB regression risk. β = 0.25 is the mid-plateau pick: well-
bracketed on both sides, captures most of the lift, dominates β=0.4
across n=20-35 (real structural evidence).

### 5. Sanity — n bracket at the locked β

At locked β = 0.25, R = 200, alt:

| n_near_min | LB F1 |
|---|---|
| 20 | 0.47382 |
| 25 | 0.47974 |
| **30** | **0.48529** ← locked |
| 35 | 0.48047 |
| 40 | 0.47832 |
| 50 | 0.47548 |

Sharp peak at n = 30, consistent with the n optimum at β = 0.4.

## Why this worked where prior post-hoc mechanisms didn't

This was the first post-hoc mechanism to substantially beat the locked
Stage 13 + N3_r0.5 baseline (+0.024 vs the previous best post-hoc gain
of +0.006 from N3_r0.5). The reason it worked where previous
mechanisms failed is **structural**, not parametric:

- **Predicted-habitat (Test 1, closed negative):** classified each
  image's habitat from its top predicted species, then applied
  demotion against the predicted habitat. Failed monotonically
  (predicted-only: 0.45547; source-fallback c3: 0.47309). The model's
  per-source predictions were already coherent enough that classifying
  habitat from them added correlated noise instead of new information.
  The source name was already the right signal.

- **Habitat-fit (this doc, won):** uses external information (training
  data lat/lon/altitude per training observation) and tilts the
  underlying probabilities BEFORE admission. Cosmopolitan species
  aren't touched; range-restricted out-of-region species get demoted
  on the sources they don't belong to. The F1-near-optimal admission
  machinery operates on improved probabilities rather than fighting
  reshaped output.

This validates the meta-lesson recorded after the genus-dedup negative:
post-hoc reshape of an already-near-optimum prediction list is
exhausted; gains must come from structurally-different sources, and
external training-data geography is one such source.

## Files

- Code: `scripts/apply_habitat_fit.py`
- Profile builders: `diagnostics/build_species_geo_profile.py`,
  `diagnostics/refine_source_centroids.py`,
  `diagnostics/add_radius_columns.py`
- Profile data: `outputs/habitat_classifier/species_geo_profile.parquet`
- Refined centroids: `outputs/habitat_classifier/refined_source_ref.json`
- Locked-best CSV: `outputs/habitat_classifier/hab_refined_R200_b0.25_n30_alt.csv`

## Open follow-ups

1. **β=0.15 vs β=0.25 head-to-head — incomplete.** β=0.15 wins at
   n=25 and n=30; β=0.25 wins at n=20; n=35 and n=40 at β=0.15 are
   on disk but not yet submitted. If β=0.15 dominates ≥3 of 4 n
   values, switch lock to β=0.15 + n=30 = 0.48705.
2. **Test 4 — EIVE / Ellenberg ecological-indicator composition.**
   Plan documented at `docs/Exp16_Habitat_Classifier_Plan.md`.
   Requires downloading EIVE 1.0 (~12 MB) and species-name matching;
   on hold pending decision.
3. **Gate A.5 — ViT-L + ArcFace + LoRA retriever.** Habitat-fit
   composes multiplicatively at the per-image probability level, so
   it stacks orthogonally with any backbone improvement.
