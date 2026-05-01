# Exp-16 — Per-genus dedup (closed sub-noise 2026-04-30)

**Status:** closed. 5 LB submissions. Best variant (`dedup_all_r0.7`)
landed +0.00011 above the locked best — well below the ±0.005 noise
floor. Stricter dedup variants regressed monotonically.

## Mechanism (one sentence)

*For each quadrat's prediction list, group predicted species by genus
and drop sibling species below `dedup_ratio × p_top-of-genus`.*

## Audit motivation

Winner-audit (2026-04-30) showed alpine sources predict Salix on
114-181% of quadrats (S. herbacea + S. serpillifolia + S. retusa
stacked) and RNNB predicts Helichrysum on 55% of quadrats with 4
species per quadrat. The hypothesis: model is genus-confident,
species-uncertain on prostrate dwarf willows and Mediterranean
everlastings; dropping siblings should reduce FPs.

## LB results

| Variant | LB F1 | Δ vs locked best (0.46168) |
|---|---|---|
| sanity_r0 (parity) | 0.46168 | 0.00000 |
| dedup_alpine_r1.0 | 0.45993 | −0.00175 |
| dedup_alpine_r0.85 | 0.46167 | −0.00001 |
| dedup_all_r1.0 | 0.45441 | −0.00727 |
| dedup_all_r0.85 | 0.45882 | −0.00286 |
| dedup_all_r0.7 | 0.46179 | +0.00011 (sub-noise) |

Monotone trend toward harm with stricter dedup:
- r=0.7 ≈ baseline
- r=0.85 = −0.003
- r=1.0 = −0.007

## Why the audit's intuition was wrong

The F1 metric is multi-label and rewards predicting all true species.
When ground-truth labels for an alpine quadrat genuinely include
multiple co-occurring dwarf willow species (which is biologically
common — alpine snowbed communities really do harbor S. herbacea +
S. serpillifolia + S. retusa together), the "Salix triplet" the
audit flagged as over-prediction is in fact correct multi-label
coverage.

Dropping siblings costs TPs faster than it gains by removing FPs.
Same for Helichrysum on RNNB — co-occurring everlasting species
are biologically real on Mediterranean dunes.

## Why this matters for future work

This is the **second consecutive negative on the "fix the prediction
shape" axis** (single-visit prior, genus-dedup). Both audits inspected
the per-source prediction *shape* and tried to clean it up; both
regressed.

The signal: the prediction shape produced by Stage 13 + N3_r0.5 is
roughly correct for the F1 metric. Adjustments at this layer are
zero-sum — every TP gained by widening costs as much as the FP
removed by tightening, on average.

**Implication:** post-hoc reshaping of an already-near-optimum
prediction list is exhausted. Future gains must come from:

1. **Improving the underlying probability vector** (better backbone,
   better kNN retriever — Gate A.5 with ViT-L + ArcFace LoRA).
2. **External information** — habitat classifier with externally-
   sourced species-habitat maps (planned, see
   `Exp16_Habitat_Classifier_Plan.md`). This is structurally
   different from genus-dedup and single-visit-prior because it
   imposes information NOT already in the model's probability
   vector.
3. **Source-specific within-source refinement** — habitat clustering
   inside large sources (alpine 1474 quadrats, currently treated
   uniformly).

## Decision

- **Mechanism: closed sub-noise.** Do not iterate to per-genus
  thresholds, per-source dedup ratios, or genus-specific lists.
  The structural lesson (model already has the right shape) means
  finer parameter tuning will not break out of the noise floor.
- **Locked best stays at F1=0.46168 (Stage 13 + N3_r0.5).**
- **Next:** the habitat-classifier plan (Variant A hard-prune with
  externally-sourced habitat lists) becomes the natural follow-up
  if/when we want another post-hoc attempt.
