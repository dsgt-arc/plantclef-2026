# Exp-16C — All LB Results (2026-05-04 → 2026-05-05)

Final lock: **F1 = 0.50623** (Stage 12B `alpine_b0.45`).
Beats prior best 0.49842 by **+0.00781**, target 0.49896 by **+0.00727**.

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

## Final lock (2026-05-05)

```
β_global=0.15, R=200, n_near_min=40, alt-on, thr=0.085, k=10
Per-source admission: OPTMix n=2 + LISAH n=2 + RNNB n=2
Per-source β: CBN-PdlC=0.45, CBN-Pla=0.45
F1 = 0.50623
```

Submission CSV: `exp16c_s12B_alpine_b0.45.csv`.

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
~80 across Stages 4-13.
