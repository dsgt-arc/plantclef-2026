# Exp-16 — Per-tile vegetation-fraction filter (closed negative 2026-04-30)

**Status:** closed negative after 4 LB submissions. Mechanism hurts
monotonically: every veg_threshold > 0 loses ~0.018-0.020 F1 vs the
no-filter anchor. **Bigger finding from this experiment is the kNN
leverage in the production pipeline (~0.03 F1, not ~0.005 as I had
estimated).**

## Mechanism (one sentence)

*Drop tiles whose ExG (excess-green) vegetation fraction is below a
threshold before max_prob aggregation, so substrate / ruler / wood-
frame tiles don't contribute high-confidence non-plant predictions to
the per-class maximum.*

## LB results

| Variant | LB F1 | Δ vs anchor | Δ vs locked best (0.46168) |
|---|---|---|---|
| veg0.00 (anchor, no filter, no kNN) | 0.42715 | 0.00000 | −0.03453 |
| veg0.02 | 0.40933 | −0.01782 | −0.05235 |
| veg0.05 | 0.40685 | −0.02030 | −0.05483 |
| veg0.10 | 0.40667 | −0.02048 | −0.05501 |

All variants used β=1.0 (no kNN) on the ViT branch (FAISS index is
PACE-only). N3_r0.5 admission applied to all.

## Two findings

### 1. The filter hurts monotonically

Even at threshold 0.02 (only 1-3% of tiles dropped on alpine sources,
~10-30% on count-bound sources), F1 drops by 0.018. The fall-off
saturates quickly — going to 0.10 (more aggressive) only loses ~0.003
more. The filter is dead.

**Why it failed:** the ExG > 20 threshold is too crude — it treats
brown dormant vegetation, dry grasses, and senescent leaves on RNNB
winter visits / OPTMix forest floor as "non-vegetation," excluding
real plant signal. The audit-derived per-source asymmetry was real
(50% of OPTMix tiles below 0.05) but the "low ExG" tiles are not
exclusively non-vegetation noise.

### 2. kNN contributes ~0.03 F1, not ~0.005

The β=1.0 (no-kNN) anchor scored 0.42715 vs the production locked
best of 0.46168. **The kNN component contributes ~0.03 F1** — roughly
6× the amount the Exp-9B summary suggested. This is the more
important finding from this experiment.

**Implications:**

- Any future post-hoc experiment that bypasses kNN starts ~0.03 below
  the current best. A non-kNN mechanism would need to gain ~0.03 just
  to break even. Realistic post-hoc ceiling is much lower than that.
- kNN is load-bearing for the production F1. The pending ViT-L +
  ArcFace + LoRA retriever (Gate A.5) is therefore higher-leverage
  than I'd been valuing — a better retriever directly affects this
  ~0.03 F1 contribution.
- Future post-hoc admission/rerank experiments should be tested on
  PACE with kNN included rather than approximated locally, because
  the no-kNN approximation distorts the absolute LB numbers by
  ~0.03 in addition to whatever the mechanism does.

## Decision

- **Filter mechanism: closed negative.** Do not iterate to finer
  thresholds or per-source thresholds.
- **Locked best stays at 0.46168 (N3_r0.5 + Stage 13).**
- **Memory updated** with the kNN-leverage finding so future
  experiments are scoped against the right baseline.
