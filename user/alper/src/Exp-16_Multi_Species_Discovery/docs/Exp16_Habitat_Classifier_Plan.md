# Exp-16 — Habitat-classifier post-processing (planned 2026-04-30)

User's idea, recorded: classify each quadrat into a habitat type from
the predicted species composition, then either (a) hard-prune species
that don't belong to the chosen habitat, or (b) re-score by demoting
out-of-habitat species softly.

## Mechanism (one sentence)

*Use the top-K predicted species as a fingerprint for the quadrat's
habitat type, classify the habitat, and either drop or down-weight any
predicted species whose habitat affinity disagrees with the chosen
habitat.*

## Why this is plausible (winner-audit evidence, 2026-04-30)

The audit confirmed that source-level biogeography is real and clean:
- Alpine sources (CBN-PdlC, CBN-Pla, CBN-can) predict pure alpine flora
  (Salix herbacea, Gentiana alpina, Geum montanum, Carex curvula).
- Coastal sources (GUARDEN, RNNB) predict Mediterranean halophytes
  (Medicago marina, Salicornia, Helichrysum stoechas).
- LISAH predicts ruderal/agricultural weeds.
- OPTMix predicts forest understory.
- Cross-source Jaccard correctly clusters: alpine pairs 0.08-0.41,
  coastal pairs 0.18, off-cluster ≈0.

The mask machinery already does this *coarsely* via the source prefix
and biogeo mask. But:
1. The mask is binary (allowed / blocked) and global per source.
2. Within-source heterogeneity exists. E.g. some LISAH quadrats are
   ruderal field margins, others are riparian, others are vineyard
   understory. A single-source mask treats them identically.
3. The current biogeo mask only handles one axis (alpine impossibility);
   it doesn't distinguish coastal-dune vs salt-marsh vs garrigue within
   GUARDEN/RNNB.

A habitat classifier built from the *prediction fingerprint itself*
could refine within-source.

## Concrete design (sketch)

### Variant A — Hard prune

1. Build habitat → species-list map from the training corpus or from
   per-source training metadata. Habitats: alpine_meadow,
   alpine_snowbed, mediterranean_dune, mediterranean_saltmarsh,
   mediterranean_garrigue, mediterranean_ruderal,
   atlantic_forest_understory, etc.
2. For each test quadrat, take the top-20 predicted species and score
   each candidate habitat by Σ p_species over species-in-habitat.
3. Pick the highest-scoring habitat (if margin over second > δ;
   otherwise abstain).
4. Prune any predicted species not in the chosen habitat from the
   final admission.

### Variant B — Soft demotion

Same up to step 3, then multiply p_species by a habitat-fit factor
(1.0 if in chosen habitat, β if out) and re-run admission. β small
(e.g. 0.5) gives a soft tilt; β = 0 reproduces variant A.

### Variant C — Use *images* not labels for habitat

Instead of building habitat lists from species lists, cluster the test
images themselves (CLS embeddings or ViT-LoRA embeddings, both already
extracted). Each cluster gets a habitat ID. Apply demotion within
clusters: if a species is rare in cluster K, demote it on K's images.

## Risks and binding caveats

1. **Anti-LB-overfit principle (binding):** see
   `feedback_anti_lb_overfit.md`. The LB is 11% / 232 images, noise
   floor ±0.005. Any habitat classifier with > 5 free parameters needs
   internal-proxy validation before LB.
2. **Within-source vs. across-source overlap:** GUARDEN ↔ RNNB share
   18% of top-50 species (correct — both Mediterranean coastal). A
   habitat classifier must not penalize species that legitimately
   appear in both.
3. **The biogeo mask already does the across-source axis well.** The
   gain from this experiment is on the *within-source* axis, which
   is smaller in scale.
4. **Training-data habitat labels may not exist.** PlantCLEF training
   data is image-level species labels; habitat labels would have to
   be inferred from external biogeographic databases (e.g. EIVE,
   Ellenberg values) or learned from co-occurrence in test set
   itself (risk: circular).
5. **Same failure mode as single-visit prior?** That experiment
   failed because "borrow from siblings" supplied correlated bias
   not new info. Habitat classifier is structurally different
   because it imposes EXTERNAL information (which species belong
   in which habitat), not intra-model self-consistency. So it
   should not have the same failure mode — but watch for the
   variant where cluster IDs come from the model's own embeddings
   (Variant C), which IS intra-model and could fail similarly.

## Pre-requisite reading

- Genus-dedup result (running 2026-04-30): if genus-dedup wins,
  habitat classifier is harder to motivate (the species over-
  prediction problem is partly solved). If genus-dedup loses,
  habitat classifier remains attractive.
- Veg-filter doc (Exp16_VegFilter_Results.md): note the kNN-
  contributes-0.03 finding. Any post-hoc that bypasses kNN starts
  -0.03 below current best.

## Decision

**Hold this experiment until genus-dedup LB results are in.**
- If genus-dedup wins (≥ +0.005), iterate on it first; come back to
  habitat classifier later.
- If genus-dedup loses or is sub-noise, this is the next experiment
  to pursue, starting with Variant A (hard prune) on alpine vs
  coastal vs ruderal vs forest cohorts (the 4 cleanest clusters
  from the cross-source Jaccard).

## Status: planned (not started)
