# Exp-16C — All LB Results (2026-05-04 → 2026-05-06)

Final pipeline lock: **F1 = 0.51096** (Stage 14C `RNNB_b0.55`).
Final ensemble lock: **F1 = 0.52739** (set-level augment of A=0.51096 ∪ B=0.49842, capped at 10).
Beats prior best 0.49842 by **+0.02897** total, target 0.49896 by **+0.02843**.

Held config (where unchanged): alt-on, R=200, k=10, alpha defaults (n_floor=3, ratio_floor=0.5).

## Stage 4 — combine Stage 3 winners (anchor for everything below)
β=0.10, n_near=30, thr=0.085

| Variant | F1 | Notes |
|---|---|---|
| C3 (OPTMix n=2 + LISAH n=2) | 0.49207 | Best of Stage 4; anchor for Stage 5+ |

## Stage 5 — threshold + topk + n_near sweep at Stage 4 anchor

| Variant | F1 | Δ |
|---|---|---|
| A_thr0.060 | (not LB) | — |
| A_thr0.075 | 0.48900 | −0.00307 |
| A_thr0.080 | 0.48034 | −0.01173 |
| A_thr0.090 | 0.49760 | +0.00553 |
| A_thr0.100 | 0.48761 | (vs 0.49207) |
| A_thr0.110 | 0.48311 | — |
| B_k12 | 0.48904 | k=12 hurts |
| B_k15 | 0.48308 | k=15 hurts |
| C_n20 | 0.47722 | tighter n hurts |
| **C_n40** | **0.50304** | **+0.01097** ← peak |
| C_n50 | 0.49997 | rolloff |
| D_alt_off | 0.49207 | alt-on neutral here |

## Stage 7 — n_near refinement around 40

| Variant | F1 |
|---|---|
| n25 | (not LB) |
| n30 | 0.49207 (anchor) |
| n35 | 0.49702 |
| **n40** | **0.50304** ← peak |
| n45 | 0.50067 |
| (n50 from Stage 5) | 0.49997 |

## Stage 8 — thr + topk re-sweep at n=40

| Variant | F1 | Δ vs 0.50304 |
|---|---|---|
| E_anchor (thr=0.085, k=10) | 0.50304 | 0 |
| A_thr0.075 | 0.48900 | −0.01404 |
| A_thr0.080 | 0.49718 | −0.00586 |
| A_thr0.090 | 0.49760 | −0.00544 |
| B_k12 | 0.50044 | −0.00260 |
| B_k15 | 0.49174 | −0.01130 |

thr=0.085 and k=10 still optimal at new n=40 anchor.

## Stage 9 — β × R sweep at n=40

| Variant | F1 | Notes |
|---|---|---|
| anchor (β=0.10, R=200) | 0.50304 | |
| A_b0.075 | 0.50137 | |
| A_b0.15 | **0.50488** | new peak |
| A_b0.20 | 0.50088 | |
| B_R150 | 0.47551 | R locked hard |
| B_R250 | 0.48414 | |
| B_R300 | 0.47816 | |
| C_b0.15_R150 | 0.47556 | β doesn't pull R |
| C_b0.15_R300 | 0.48090 | β doesn't pull R |

R=200 is a sharp narrow peak, β=0.15 is the new global β.

## Stage 10 — lock confirmation at β=0.15

| Variant | F1 |
|---|---|
| E_anchor (β=0.15, n=40, thr=0.085, k=10) | 0.50488 |
| A_n35 | 0.49757 |
| A_n45 | 0.50335 |
| B_thr0.080 | 0.49594 |
| B_thr0.090 | 0.50042 |

All side-points lose. Lock confirmed.

## Stage 11 — per-source admission re-tune at β=0.15

| Variant | F1 | Δ vs 0.50488 |
|---|---|---|
| F_anchor | 0.50507 | +0.00019 (sanity ✓) |
| A_OPTMix_n1 | (not LB) | — |
| A_OPTMix_n3 | (not LB) | — |
| **E_RNNB_n2** | **0.50507** | **+0.00019** (only positive lift in Family E) |
| E_GUARDEN_n2 | 0.50488 | 0 |
| C_LISAH_r0.6_n2 | 0.50488 | 0 (LISAH ratio_floor dead at n=2) |
| C_LISAH_r0.7_n2 | 0.50488 | 0 |
| C_LISAH_r0.8_n2 | 0.50488 | 0 |
| B_LISAH_n3 | 0.50352 | −0.00136 (n=2 still wins) |
| B_LISAH_n4 | 0.50482 | −0.00006 |
| D_OPTMix_r0.3_n2 | 0.50327 | −0.00161 (partially rehabilitated, still neg) |
| D_OPTMix_r0.4_n2 | 0.50327 | −0.00161 |
| D_OPTMix_r0.6_n2 | 0.50004 | −0.00484 |

New tentative anchor: 0.50507 with RNNB n=2 added.

## Stage 12 — per-source β (NEW MECHANISM) at 0.50507 anchor

| Variant | F1 | Δ vs 0.50507 |
|---|---|---|
| F_anchor (no per-source β) | 0.50507 | 0 (code path validated ✓) |
| A_OPTMix_b0.05 | 0.50507 | 0 (no-op — see below) |
| A_OPTMix_b0.10 | 0.50507 | 0 (no-op) |
| B_alpine_b0.05 | 0.50516 | +0.00009 |
| B_alpine_b0.10 | 0.50515 | +0.00008 |
| **C_alpine_b0.30** | **0.50551** | +0.00044 |
| **C_alpine_b0.50** | **0.50558** | +0.00051 |
| C_alpine_b1.00 | 0.49106 | −0.01401 (catastrophic) |
| D_OPTMix_b0.05_alpine_b0.10 | 0.50515 | +0.00008 |
| D_OPTMix_b0.10_alpine_b0.50 | 0.50558 | +0.00051 (= alpine alone) |

Findings: OPTMix β-axis dead. Alpine wants LOOSER demotion (β≈0.50). β=1.00 catastrophic.

## Stage 12B — alpine β refinement around 0.50

| alpine β | F1 | Δ vs 0.50558 |
|---|---|---|
| 0.30 (Stage 12) | 0.50551 | −0.00007 |
| 0.40 | 0.50590 | +0.00032 |
| **0.45** | **0.50623** | **+0.00065** ← FINAL LOCK |
| 0.50 (anchor) | 0.50558 | 0 |
| 0.55 | (not LB) | — |
| 0.60 | 0.50477 | −0.00081 |
| 0.70 | (not LB) | — |
| 1.00 (Stage 12) | 0.49106 | catastrophic |

## Stage 14 — per-source β at alpine_b0.45 anchor (0.50623)

| Variant | F1 | Δ |
|---|---|---|
| anchor (no extra override) | 0.50623 | 0 |
| **RNNB_b0.50** | **0.50876** | **+0.00253** ← biggest single change |
| RNNB_b0.05 | 0.50647 | +0.00024 |
| OPTMix_b0.05 | 0.50623 | 0 (no-op) |
| OPTMix_b0.30 | 0.50407 | −0.00216 |
| OPTMix_b0.50 | 0.49924 | −0.00699 |
| LISAH_b0.30 | 0.50282 | −0.00341 |
| LISAH_b0.50 | 0.49683 | −0.00940 |
| GUARDEN_b0.30 | 0.50166 | −0.00457 |
| GUARDEN_b0.50 | 0.49831 | −0.00792 |
| CBN-can b0.05/0.30/0.50 | 0.50623 | 0 (no-op, all 3) |
| alpine_asym_PdlC0.30_Pla0.45 | 0.50634 | +0.00011 |
| alpine_asym_PdlC0.45_Pla0.30 | 0.50540 | −0.00083 |

Findings: RNNB wants LOOSE β (like alpine). OPTMix/CBN-can β are no-ops. LISAH/GUARDEN want global β=0.15.

## Stage 14B — RNNB β refinement around 0.50

| RNNB β | F1 |
|---|---|
| 0.30 | 0.50745 |
| 0.40 | 0.50551 |
| 0.50 (anchor) | 0.50876 |
| **0.60** | **0.50991** ← new peak |
| 0.70 | 0.50783 |

## Stage 14C — RNNB β refinement around 0.60

| RNNB β | F1 |
|---|---|
| 0.50 | 0.50876 |
| **0.55** | **0.51096** ← FINAL peak |
| 0.60 | 0.50991 |
| 0.65 | 0.50876 |
| 0.70 | 0.50783 |

## Final lock (2026-05-05)

```
β_global=0.15, R=200, n_near_min=40, alt-on, thr=0.085, k=10
Per-source admission: OPTMix n=2 + LISAH n=2 + RNNB n=2
Per-source β: CBN-PdlC=0.45, CBN-Pla=0.45, RNNB=0.55
F1 = 0.51096
```

Submission CSV: `exp16c_s14C_RNNB_b0.55.csv`.

## Stage 13 — ConvNeXt cooldown comparison (IN PROGRESS, regression)

| Variant | F1 |
|---|---|
| production scheme P thr=0.085 (standalone) | 0.35610 |
| cooldown_last scheme P thr=0.085 | 0.26867 (−0.087, broken) |
| cooldown_last scheme P thr=0.100 | 0.27581 |
| cooldown_last scheme P thr=0.125 | 0.29072 |
| cooldown_last scheme K thr=0.100 | 0.27471 |

Diagnostic in progress: extract scale 5 with `model_best.pth.tar` (vs `last.pth.tar`) to test whether 20-epoch cooldown drifted past peak.

## LB submissions used in Exp-16C
~110 across Stages 4-14C.

## Stage 15 — pseudo-ensemble at submission level (2026-05-06)

Idea: combine A=`WINNER_exp16c_s14C_RNNB_b0.55.csv` (0.51096, ViT-L+per-source-β pipeline)
with B=`OLD_WINNER_D_OPTMix_k5.csv` (0.49842, ViT-B+per-source-admission pipeline).
Per-source overlap: RNNB 37%, CBN-Pla 47%, OPTMix 45% — high disagreement = real diversity.

Merge schemes tested at cap=10:

| Mode | F1 | Δ vs A=0.51096 |
|---|---|---|
| **augment_A (= union here)** | **0.52739** | **+0.01643** ← lock |
| persrc_OPTMixRRF_alpineAonly | 0.52739 | +0.01643 (different content, 132 rows differ) |
| interleave | 0.52387 | +0.01291 |
| rrf | 0.52385 | +0.01289 |
| borda | 0.52375 | +0.01279 |
| persrc_OPTMixB_alpineINT | 0.51870 | +0.00774 |
| intersection | 0.47787 | −0.03309 |

Findings:
- A-priority asymmetric merge wins (A is genuinely better, symmetric schemes dilute its edge)
- augment_A and union are byte-identical for cap=10 with A's rankings preserved
- Symmetric merges (rrf/borda/interleave) all plateau at ~0.524
- Intersection collapses (recall starvation)
- Merge-scheme axis fully explored — 0.52739 is the ceiling for this CSV pair

## Final submission selection (5 slots)

Diversity-weighted across project axes:

1. **`ENS_union_*.csv`** (0.52739) — primary, A+B augment merge
2. **`ENS_persrc_OPTMixRRF_alpineAonly_*.csv`** (0.52739) — same public score, 132 rows differ
3. **`WINNER_exp16c_s14C_RNNB_b0.55.csv`** (0.51096) — A pipeline standalone (ViT-L + per-source β)
4. **`OLD_WINNER_D_OPTMix_k5.csv`** (0.49842) — B pipeline standalone (ViT-B + per-source admission ratio_floor)
5. **`ENS_interleave_*.csv`** (0.52387) — symmetric merge fallback

Three independent diversity axes covered:
- Merge math: picks 1, 2, 5
- A pipeline (ViT-L stack): picks 1-3 use it
- B pipeline (ViT-B stack): picks 1, 2, 4, 5 use it; pick 4 standalone

ConvNeXt cooldown integration FAILED (model_best=0.30, last=0.27 vs production=0.34 standalone).
Production ConvNeXt phase2 retained in both A and B pipelines.
