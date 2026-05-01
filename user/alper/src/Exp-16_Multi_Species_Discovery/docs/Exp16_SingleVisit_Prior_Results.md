# Exp-16 — Single-visit source-prior blend (closed negative 2026-04-30)

**Status:** closed negative after 4 LB submissions. Mechanism hurts
monotonically through α=0.20; α=0.30 only partly recovers.

## Mechanism (one sentence)

*For quadrats whose site has only 1 photo in the test set (no temporal
prior), blend the per-image probability vector with the empirical
per-source mean prior at α; multi-visit quadrats are unchanged.*

## Audit motivation (round-2 audit, 2026-04-30)

- 509/2105 quadrats are single-visit. Temporal fusion is a no-op for
  them by construction.
- Per-source single-visit fractions: GUARDEN 100%, OPTMix 100%,
  CBN-can 100%, LISAH 84%, RNNB 16%, alpine 0%.
- Hypothesis: single-visit images can borrow strength from same-source
  siblings. By construction this never touches alpine (CBN-PdlC,
  CBN-Pla are 0% single-visit), so unlike Stage 1 sim_w it should not
  collapse alpine accuracy.

## LB results

| Variant | LB F1 | Δ vs locked best (0.46168) |
|---|---|---|
| sanity_alpha0 (parity) | 0.46168 | 0.00000 |
| svp_alpha0.10 | 0.45477 | −0.00691 |
| svp_alpha0.20 | 0.44919 | −0.01249 |
| svp_alpha0.30 | 0.45008 | −0.01160 |

Diffs concentrated 100% on single-visit quadrats (104 / 179 / 230 rows
changed at α=0.10/0.20/0.30 respectively). Alpine never touched.

## Why it failed

The per-source mean prior on count-bound sources is dominated by the
model's *current* top-1 confusion classes:
- GUARDEN: top1 = sid 1359344 at p=0.055
- LISAH: top1 = sid 1356013 at p=0.039
- OPTMix: top1 = sid 1356404 at p=0.086
- RNNB: top1 = sid 1359344 at p=0.078

Blending toward those means *amplifying the model's source-level
systematic bias* rather than borrowing real plant signal. The
siblings share the same systematic errors as the target image, so
averaging them just sharpens those errors.

## Decision

- **Mechanism: closed negative.** Do not iterate to finer α or
  multi_visit_only mode — the failure mode (averaging shared model
  bias) is structural, not parametric.
- **Locked best stays at F1=0.46168 (Stage 13 + N3_r0.5).**
- **Lesson:** "borrow from same-source siblings" only helps if the
  siblings carry information the target image lacks. When siblings
  share systematic backbone errors with the target, the prior is
  *the bias itself*, not corrective signal. A useful prior would
  need to come from a structurally different model, source, or
  ground-truth reference, not from the same model's same-source
  predictions.
