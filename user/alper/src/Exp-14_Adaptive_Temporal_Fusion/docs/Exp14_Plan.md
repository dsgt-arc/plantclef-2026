# Exp-14 — Adaptive Temporal Fusion (Plan)

## Context

The current Exp-11B submission (F1=0.44428 public LB) applies static-α temporal fusion per branch: ViT α=0.15 (85% prior weight), ConvNeXt α=0.30 (70% prior weight). The location prior is built by element-wise max across all visits to the same plot (`Exp-8/temporal_fusion.py:116-132`, `np.max`). For paired-visit sites where vegetation density changes dramatically year-to-year — e.g., RNNB-3-1, RNNB-3-2, RNNB-2-7, RNNB-1-2 (May 2023 lush → January 2024 bare sand) — this combination assigns the lush visit's species set to the barren visit. Inspection of `ens_knn_w0.77_thr0.100_top10.csv` confirms this: RNNB-2-7 has byte-identical predictions across both dates; RNNB-3-1/3-2/1-2 differ only by reordering. The barren image's own visual evidence is being drowned by the prior.

**Goal:** introduce per-image adaptive fusion that down-weights the prior when the current image's visual content does not match the historical visits. Target: +0.01 to +0.03 F1 on the full 2,105-image test set, primarily by recovering lost precision on multi-visit barren follow-ups (462 two-visit sites + high-cadence CBN sites).

**Hard constraint — anti-LB-overfit.** The public LB scores ~280 images (~13% of the 2,105 test set). Optimizing parameters directly on LB feedback is statistical noise-fitting. Decisions in this plan are gated on **internal proxies** computed against all 2,105 images; LB submissions are used only as coarse confirmation, not as the optimization target.

## Anti-overfit principles (binding)

1. **Internal proxies are primary.** Every controller variant is scored against four internal metrics computed on all 2,105 images (no labels needed):
   - **`rnnb_pair_divergence`** — `1 - jaccard(top10_a, top10_b)` averaged over RNNB pair sites. Currently ~0.0–0.1 (near-identical). Target ≥0.4.
   - **`stable_site_invariance`** — Jaccard(top10) within CBN-PdlC site clusters (alpine, season-stable). Should *not* drop below baseline; if a strategy moves these >10%, it is breaking working cases.
   - **`density_count_correlation`** — Spearman ρ between vegetation density (Phase 2 signal) and #species predicted per image. Currently near zero. Should become positive.
   - **`photo_faithfulness`** — fraction of final top-10 species that also appear in P_photo top-20. Currently low for barren images; should rise.
2. **Coarse parameter precision.** Sweep α at {0.15, 0.30, 0.50, 0.70} grain, not {0.18, 0.22, 0.26, 0.30}. Forces real-mechanism wins.
3. **Maximum 8 LB submissions across the entire experiment.** Each one is a single-axis perturbation off a locally-validated config. Bracket-confirm direction once.
4. **Mechanism must be explicable in one sentence.** If the winning config requires a paragraph to justify, it's overfitted.

## Architecture — new folder

```
user/alper/src/Exp-14_Adaptive_Temporal_Fusion/
├── docs/
│   ├── Exp14_Plan.md                    # this plan, copied + extended
│   ├── Exp14_Diagnostic_Findings.md     # filled at end of Phase 1
│   └── Exp14_Results_Summary.md         # filled at end of Phase 5
├── diagnostics/
│   ├── viz_per_image.py                 # local — produces side-by-side comparisons
│   ├── proxy_metrics.py                 # local — computes the 4 internal proxies
│   └── prefusion_sanity_check.py        # Phase 0 failure-mode-3 check
├── signals/
│   ├── compute_density.py               # local — ExG/HSV vegetation density per JPEG
│   ├── compute_confidence.py            # local — c_max, c_top5, c_entropy from parquets
│   ├── compute_visit_similarity.py      # PACE — pairwise CLS-cosine within each location
│   └── compute_season_weights.py        # local — date-pair seasonal alignment
├── controller/
│   ├── alpha_controller.py              # the unified per-image α function
│   └── prior_aggregators.py             # max | mean | top2_mean | sim_w | conf_w | season_w
├── scripts/
│   ├── fuse_v2.py                       # extends Exp-8 fuse_all with controller + aggregators
│   ├── ensemble_v2.py                   # extends Exp-11/ensemble_with_knn.py
│   └── stage_runner.py                  # orchestrates the staged sweep
└── slurm/
    ├── slurm_signals.sbatch             # PACE — compute_visit_similarity
    └── slurm_stage_sweep.sbatch         # PACE — runs one stage of the sweep
```

`temporal_fusion.py` (Exp-8) gets minimally extended with two new params (`aggregator`, `alpha_map`) on its existing functions — no rewrite. Exp-11/`ensemble_with_knn.py` gets a sibling `ensemble_v2.py` that imports the new pieces; the original is untouched.

## Phase 0 — Pre-flight (mandatory, ~30 min, zero LB cost)

**0.1 Pull pre-fusion parquets from PACE to local**
- `submissions/exp9b_knn_diff_beta_phase2a/knn_bM0.85_bS1.00_T1.50_prefusion.parquet` (ViT branch, pre-fusion)
- `parquet/exp11_ensemble/convnext_large/scale_{4x4,5x5,6x6}.parquet` (ConvNeXt tiles)
- `submissions/exp9b_knn_diff_beta_phase2a/knn_bM0.85_bS1.00_T1.50_fused_a0.15.parquet` (post-fusion reference for diff)
- Test JPEGs are already local at `C:\Users\alper\Documents\PlantCLEF_2026_Local\PlantCLEF2025test\images\`.

**0.2 Failure-mode-3 sanity check** (`diagnostics/prefusion_sanity_check.py`):
- Load pre-fusion ViT parquet rows for `RNNB-2-7-20230512` and `RNNB-2-7-20240117`.
- Compute `np.max(np.abs(P_a - P_b))`. Must be > 0.05.
- Repeat for the other 3 RNNB pairs the user flagged.
- **If max-diff is ~0 for any pair: STOP.** The bug is upstream of fusion; an extractor or dedupe step is reading the same image twice. The α plan is invalid until this is fixed.
- Log the per-pair diff stats to `docs/Exp14_Diagnostic_Findings.md`.

**0.3 Confirm public-LB sample size** with user (currently ~280 / 2,105 = 13%, user-asserted ~11%). This sets the LB noise floor and the maximum-submissions budget downstream.

## Phase 1 — Diagnostic notebook (local, ~half day)

**1.1** `diagnostics/viz_per_image.py` — for any `quadrat_id`, render a single PDF page:
- Original image (resized).
- Optional vegetation-density heatmap overlay (computed via Phase 2.1).
- P_photo top-10 species: ID + name + probability bar.
- P_loc (current max-aggregated prior) top-10 species: ID + name + bar.
- P_final (current α=0.15 blend) top-10: bar, with species color-coded by source (photo-only / prior-only / both).
- Sister visits at the same location: thumbnails + their P_photo top-10.

Render a multi-page PDF for: the 8 user-flagged RNNB images + 30 randomly sampled multi-visit pairs across all 5 source prefixes (RNNB / CBN-PdlC / CBN-Pla / GUARDEN / LISAH / OPTMix).

**1.2** `diagnostics/proxy_metrics.py` — implements the 4 internal proxies and computes them on the **current Exp-11B submission** to establish the baseline numbers all subsequent variants are scored against.

**1.3** Write `docs/Exp14_Diagnostic_Findings.md` capturing:
- Phase 0.2 sanity-check results.
- Baseline proxy metrics.
- Visual confirmation of the over-prediction pattern: how many of the 30 random multi-visit pairs show the lush-prior-clobbers-barren-image pattern? Quantify before fixing.

**Decision gate after Phase 1:** if the diagnostic data refutes the hypothesis (e.g., RNNB pairs are not actually misclassified, or the prior contribution is small), stop and reassess. Don't proceed to Phase 2 on autopilot.

## Phase 2 — Signal computation

**2.1 Vegetation density** — `signals/compute_density.py` (local, ~10 min CPU, parallel):
For each test JPEG: resize → 512×512 → central 384×384 crop → ExG = `2G − R − B` → Otsu mask → HSV green-range mask → `density = 0.5 * exg_mask.mean() + 0.5 * hsv_mask.mean()` ∈ [0,1]. Write `density.parquet` keyed by `quadrat_id`. Validate: density(May lush RNNB) ≥ 0.4, density(January bare RNNB) ≤ 0.1.

**2.2 Photo confidence** — `signals/compute_confidence.py` (local, seconds):
From the pre-fusion parquets, per image compute `c_max`, `c_top5`, `c_entropy`, normalized to [0,1] via dataset percentiles. Write `confidence.parquet` per branch (ViT and ConvNeXt separately).

**2.3 Visual visit similarity** — `signals/compute_visit_similarity.py` (PACE, ~few min on CPU node):
Load `cls_embedding` column from the existing per-image-aggregated DINOv2 parquet (already produced by `Exp-9B/project_embeddings.py`). Per location, compute pairwise cosine-similarity matrix `sim[loc] : [N_v, N_v]`. Write `visit_similarity.npz`. **This is the highest-leverage signal:** it lets the prior aggregator weight each historical visit by how visually similar it is to the current image, capturing the year-to-year vegetation variation the user flagged.

**2.4 Seasonal weights** — `signals/compute_season_weights.py` (local, seconds):
Parse dates from filenames; compute cyclical month distance per visit pair: `season_w_iv = exp(-((Δmonths mod 12 clipped to [0,6])^2) / (2·σ²))`, σ=2. Write `season_weights.npz`. Lower-priority signal per user; included for completeness so we can quantify whether seasonality matters at all.

**2.5 Source labels** — `signals/__init__.py` exposes `parse_source(qid) → str` returning one of `{CBN-PdlC, CBN-Pla, RNNB, GUARDEN, LISAH, OPTMix, OTHER}`.

## Phase 3 — Controller + new aggregators

**3.1 New prior aggregators** — `controller/prior_aggregators.py`:
```python
def aggregate_prior(quadrat_probs, location_groups, mode, *,
                    sim_matrix=None, season_weights=None,
                    confidences=None) -> Dict[loc, P_loc]:
    # mode ∈ {"max" (baseline), "mean", "top2_mean",
    #         "sim_w", "conf_w", "season_w", "sim_season_w"}
```
For `sim_w`, the prior is **per-(location, current_image)**, not per-location:
```
P_loc_for_i(s) = Σ_v sim_iv · P_v(s)  /  Σ_v sim_iv
```
This means `aggregate_prior` returns a callable `f(qid) → P_loc` rather than a dict for similarity-weighted modes. Cleanest fix to the user's stated mental model.

**3.2 α controller** — `controller/alpha_controller.py`:
```python
α_i = clip(α_src(source_i)
           * (1 + λ_c · (conf_i − 0.5))
           * (1 + λ_d · (0.5 − dens_i))
           * (1 + λ_s · (1 − season_w_max_i)),
           α_min, α_max)
```
Defaults that recover Exp-11B exactly: `λ_c=λ_d=λ_s=0`, `α_src ≡ 0.15` (ViT) or `0.30` (ConvNeXt), `α_min=0.05`, `α_max=0.95`. By construction, the controller cannot underperform baseline if all λ stay at 0.

**3.3 `temporal_fusion.py` extension** — minimal, additive:
- `build_location_profiles(..., aggregator="max")` accepts new modes (delegates to `prior_aggregators.aggregate_prior`).
- `apply_prior_blend(p_q, p_loc, alpha)` unchanged.
- `fuse_all(..., alpha_map: Optional[Dict[qid, float]] = None)` — when supplied, uses per-qid α; falls back to scalar `params["alpha"]` otherwise.

**3.4 `ensemble_v2.py`** — sibling of `ensemble_with_knn.py`:
- New CLI flags: `--prior-aggregator`, `--alpha-controller-config <json>`.
- Loads density/confidence/sim/season parquets if present; computes `alpha_map` via the controller; passes through to `fuse_all`.
- Uses the **pre-fusion** ViT parquet (input change vs Exp-11B which loads post-fusion).

## Phase 4 — Staged sweep with internal-proxy gating (PACE, ~5 days)

Each stage:
1. Run controller variant locally on cached parquets → produces a candidate submission CSV in seconds.
2. Score against the 4 internal proxies.
3. **Pass criterion** to advance: `rnnb_pair_divergence` ≥ +0.15 over baseline AND `stable_site_invariance` within −0.05 of baseline AND `photo_faithfulness` ≥ +0.05.
4. Only stage *winners* (by internal proxy) get an LB submission, max 1 per stage. Total LB budget: 8.

| Stage | Variable | Holds | LB? |
|---|---|---|---|
| 0 | Re-emit Exp-11B baseline through new code path | n/a | 1 (sanity — must reproduce 0.44428 ±0.001) |
| 1 | Prior aggregator: `max` vs `mean` vs `top2_mean` vs `sim_w` vs `conf_w` | λ=0, α_src baseline | 1 (best of 5 by proxy) |
| 2 | `α_src[RNNB]` ∈ {0.15, 0.30, 0.50, 0.70} | aggregator from Stage 1 | 1 (best by proxy) |
| 3 | `λ_d` (density gate) ∈ {0, 0.4, 0.8, 1.2}, applied broadly | α_src from Stage 2 | 1 (best by proxy) |
| 4 | `λ_c` (confidence gate) ∈ {0, 0.4, 0.8} | accumulated config | 1 (best by proxy) |
| 5 | `λ_s` (season gate) ∈ {0, 0.4, 0.8} + season-weighted aggregator | accumulated | 1 (only if Stage 4 paid off; user flagged seasonality as low-priority) |
| 6 | Re-tune ensemble weight `w` ∈ {0.73, 0.77, 0.81} | best controller | 1 |
| 7 | Bracket-confirm winner: ±1 grid step on the most-impactful axis | n/a | 1 |

**Kill rule:** if a stage's best variant doesn't pass the internal-proxy criterion, *skip* the LB submission for that stage and continue to the next axis with the previous-stage config locked.

**Per-branch sweep:** ViT and ConvNeXt branches are fused independently. Stages 1–5 run separately on each, with the proxy metrics computed against the *ensemble* output (since that's what gets submitted). This is twice the local work but no extra LB submissions.

## Phase 5 — Result documentation

**5.1** `docs/Exp14_Results_Summary.md` — same template as `Exp11B_Results_Summary.md`. Tables: per-stage proxy metrics, per-stage LB results, final config.

**5.2** Update memory:
- New file `current-best-config.md` if F1 improved.
- New `feedback_anti_lb_overfit.md` capturing the internal-proxy-first approach as a durable principle.
- New `exp14-findings.md` per the project-memory pattern (lead with rule, **Why:**, **How to apply:**).

**5.3** If F1 ≥ 0.46, generate the final submission and copy to `C:\Users\alper\Desktop\exp14_adaptive_fusion\`. Otherwise document why and propose Exp-14b (e.g., SAM-based density for the residual barren cases).

## Critical files

**Modify (additive only — no behavior change at default params):**
- `user/alper/src/Exp-8_Quantile_Aggregation_SS/temporal_fusion.py` — add `aggregator` param to `build_location_profiles` (line 116); add `alpha_map` param to `fuse_all` (line 191).

**Create (new Exp-14 folder):**
- `user/alper/src/Exp-14_Adaptive_Temporal_Fusion/` — full tree per the Architecture section above. ~14 new files, ~1,200 LOC total.

**Reuse without modification:**
- `Exp-9B_Improved_KNN_Retrieval/multiscale_aggregate_knn.py` — only the pre-fusion parquet output is read.
- `Exp-9B_Improved_KNN_Retrieval/project_embeddings.py` — `cls_embedding` column already exists.
- `Exp-8_Quantile_Aggregation_SS/generate_submission.py` — submission CSV writer.
- `Exp-11_Conv_ViT_Ensemble/ensemble_with_knn.py` — left untouched as the F1=0.44428 baseline reference; `ensemble_v2.py` sits alongside.
- `Exp-4_Enhanced_Tile_Aggregation/docs/test_dataset_location_analysis/PlantCLEF2025_test.csv` — date metadata source.

## Verification

**Stage 0 (smoke test, must pass before any sweep):**
- `ensemble_v2.py` with controller defaults (λ=0, α_src=0.15/0.30, aggregator="max") must reproduce `ens_knn_w0.77_thr0.100_top10.csv` byte-for-byte (or to ≤0.001 numerical tolerance) when fed the same inputs as Exp-11B. If not, the controller wiring has a bug; do not submit.

**Per-stage:**
- Run `proxy_metrics.py` against each candidate; log to `docs/proxy_results.csv`.
- If `stable_site_invariance` drops by >0.10 on CBN-PdlC, the strategy is breaking the working case — reject regardless of other proxies.

**End-to-end:**
- Final submission verified by visual spot-check: re-render `viz_per_image.py` for the 8 user-flagged RNNB images. Predictions on barren visits should now differ meaningfully from lush-visit predictions, with the photo-only species rising and the prior-only species falling.
- LB submission for the final config; report public-LB delta with explicit caveat that ~280 images is noisy.

## Out of scope

- SAM / GroundingDINO masking (Next_Steps_Literature_Review B2). Higher cost, separate experiment if Exp-14 leaves residual barren-image errors.
- Retraining any backbone.
- Per-class beta recalibration. The Exp-9B β=0.85 is held fixed; this experiment touches only temporal-fusion α and prior aggregation.
- New ensemble members (Exp-13 ViT-L). That's `ensemble_improvement_plan.md`'s lane.

## Risks (rank-ordered)

1. **Failure-mode-3 (upstream bug):** Phase 0.2 may show RNNB-2-7's pre-fusion vectors are byte-identical, indicating an extractor bug. The α plan is then irrelevant until that is fixed. **Detection: Phase 0.2.**
2. **Internal proxies don't track LB:** strategies that improve `rnnb_pair_divergence` may still hurt LB if the test-set ground truth actually expects persistent perennials predicted across visits. **Mitigation: Stage 0 LB sanity check; if Stage-1 winner *drops* LB, the proxy is misaligned and we revert + reassess.**
3. **Per-source α overfits to the 5 visible source prefixes** with degenerate transfer to private LB. **Mitigation: coarse {0.15, 0.30, 0.50, 0.70} grain, mechanism-explicable defaults, single-axis stages.**
4. **Density signal saturates on RNNB:** bordering wood frame may dominate ExG even on bare-sand quadrats. **Mitigation: Phase 2.1 validation — RNNB barren density must be ≥2σ below RNNB lush density; if not, drop the density signal entirely.**
