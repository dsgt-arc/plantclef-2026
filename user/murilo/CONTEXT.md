# Session Context

Session handoff document for Murilo's PlantCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

**Last updated**: 2026-05-02 PM. Exp 003 (TileQ-Decoder) scaffolded —
`proposal.md` + `spec.md` + `tasks.md` + `pyproject.toml` committed on
`mg/tileq-decoder` and opened as PR #15. **No source files written yet** —
the spec is under review before any implementation begins. Competition ends
**2026-05-07** (5 calendar days from this update).

---

## Working principles (read every session)

Adapted from
[Karpathy CLAUDE.md](https://github.com/forrestchang/andrej-karpathy-skills/blob/main/CLAUDE.md).
These are the operating contract for any Claude session in this codebase —
they apply *before* any specific task.

1. **Think before coding.** Surface assumptions and confusion explicitly. If
   uncertain about scope, intent, or the right approach, **ask** rather than
   proceed silently. When multiple plausible interpretations exist, list them
   and ask Murilo to pick. The cost of one clarifying question is far below
   the cost of a wrong direction taken in the last 5 days of a competition.
2. **Simplicity first.** Write the minimum viable code that achieves the
   goal. Skip speculative features, premature abstractions, retry loops for
   internal-only code, defensive validation against bugs that can't actually
   happen. The quality test: *"would a senior engineer say this is
   overcomplicated?"* Three similar lines beat a premature abstraction.
3. **Surgical changes.** Modify only what's necessary for the task. **Match
   existing style**, even if a different approach would be your preference —
   this codebase has conventions across `experiments/NNN-*`, `src/plantclef/`,
   and the proposal/spec/tasks/results triplet. Don't rewrite working code
   while fixing an unrelated bug. Remove only dead code your changes
   created, not pre-existing dead code.
4. **Goal-driven execution.** For *experiments*, define a falsifiable
   hypothesis + Kaggle-measurable success criteria before writing the
   training loop. For *bug fixes*, write the failing reproduction first;
   "fix the bug" → "the test that currently fails passes". For *new features*,
   state what observable behaviour proves the feature works before writing
   it. Avoid commits whose verification is "looks right".
5. **Independence boundary.** This workspace's experiments must NOT extend
   Alper's pipeline. Do not consume artefacts from `aerten3/` (FAISS index,
   ConvNeXt weights, geo mask, Exp-7 multiscale parquets, temporal-fusion
   code). Use only Murilo's cached parquets, the team-shared train CLS
   embeddings, the public ViT-B checkpoint, raw test/train images, and
   public metadata. The independence boundary is a paper-contribution
   requirement, not a technical one — see `proposal.md` for each experiment
   for the rationale.

When following these guidelines would harm a trivial task, exercise judgment
— a one-line typo fix doesn't need a hypothesis statement. But the default
posture is the principles above, not "ship fast".

---

## Read these files first

Read at session start to get full context. Priority matters — read the top
three fully, the rest can be skimmed.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/experiments/003-tileq-decoder/spec.md` | **Active experiment ground truth.** Full technical spec for TileQ-Decoder: architecture (4 M-param label-as-query decoder over 16×768 ViT-B CLS tokens), Asymmetric Loss, Sampler A vs Sampler B (CLS-domain pseudo-quadrats), FAISS background-cleaning, training config, sanity checks, ablation table (A0a/A0b/A1/A2/A3), repo structure, failure modes, absolute path reference. 11 sections, dependency-ordered. | Read fully |
| `user/murilo/experiments/003-tileq-decoder/proposal.md` | **Hypothesis framing.** H1 (decoder beats `max_prob`?) + H2 (test-anchored Sampler B beats naive Sampler A?). Success/reject criteria are explicit Kaggle F1 deltas. A null result on H2 is publishable. | Read fully |
| `user/murilo/experiments/003-tileq-decoder/tasks.md` | **Implementation checklist**, dependency-ordered (not day-ordered). A0a/A0b derisk-first. | Read fully |
| `user/murilo/notes/deep-research-briefing.md` | **Comprehensive briefing.** Map of cached processed data (test prob parquets, 1.4 M train CLS embeddings, ConvNeXt parquets, metadata), Alper's F1=0.4379 pipeline summary, already-rejected techniques, candidate angles. Originally written for the Claude deep-research session that produced exp 003. | Skim |
| `user/alper/src/Final_pipeline_methods_Summary/Final_Pipeline_Methods_Summary.md` | **Team-level current best (F1=0.4379)** — Alper's ConvNeXt-L DINOv3 + ViT-B DINOv2 ensemble. Pipeline reference for what's already been tried. | Skim — reference only |
| `user/murilo/experiments/002-classification-methods/results.md` | **Validated baseline F1=0.30810** — sorted-prob top-k=9 on grid_4x4, 99.9 % exact match to the 2025 reference CSV. The classifier `classify.py` is the source for A0a/A0b in exp 003 with no new code. | Skim |
| `user/murilo/experiments/001-dinov2-grid-ablation/results.md` | Per-tile probability parquets at `shared/inference/dinov2-vit-base/{no_grid, grid_2x2…grid_10x10}.parquet` (10 grids, ~12 GB). My code, my schema, fully reusable. | Skim |
| `user/murilo/README.md` | Workspace conventions — per-experiment `proposal.md/tasks.md/results.md/pyproject.toml`, `artifacts/`, `src` symlink. | Skim |
| `CLAUDE.md` (project root) | OpenSpec/OPSX workflow, doc slash commands, PACE Phoenix conventions (scratch, venv, XDG_CACHE_HOME). | Reference |

---

## Current state

**Team-level best**: **F1 = 0.4379** (Alper's exp 11, ConvNeXt-L DINOv3 +
ViT-B DINOv2 probability-level ensemble with multiscale 4+5+6 tiles, ViT-only
FAISS kNN, temporal `prior_blend`, geo mask, threshold 0.100, topK 10).

**My validated best**: **F1 = 0.30810** (sorted-prob top-k=9 on grid_4x4,
exp 002) — exact-match reproduction of the 2025 reference CSV.

**My target for the paper**: independent methodological contribution in
F1 ∈ [0.20, 0.35]; Kaggle score is **not the bar**, ablation cleanliness is.
A null result on H2 of exp 003 is publishable.

**Competition deadline**: 2026-05-07 (5 days from last update).

**Active branch**: `mg/tileq-decoder` — PR #15 open, scaffold only, no source
files yet. Spec under review.

**Recent merges to main**:

- PR #14 → Alper's Exp-10 ConvNeXt finetuning
- PR #7 → exps 001 + 002 (Murilo, `mg/dinov2-grid-ablation`) — merged 2026-04-18-ish
- (current) PR #15 → exp 003 scaffolding — open, awaiting review

| Submission | F1 | Status |
|---|---|---|
| **Alper exp 11 final ensemble** | **0.4379** | Public LB, 2026-04-17 |
| Sorted-prob top-9, grid_4x4 (Murilo exp 002 — 2025 reproduction) | 0.30810 | validated 2026-03-13 |
| Expanded-list top-5 max-20, grid_4x4 (exp 002) | 0.23 | |
| max-prob top-10 (organisers' approach) | ~0.23 | reference, not yet submitted by Murilo |
| Union top-5 (exp 002) | 0.21313 | |
| No-grid baseline (exp 001) | 0.14506 | |

---

## What happened recently

### 2026-05-02 PM — exp 003 scaffolded; PR #15 opened

Outcome of a multi-turn deep-research session with Claude that produced a
TileQ-Decoder design. Sequence of decisions:

1. **Framing pivot**: from "implement Q2L" → "let deep research find the best
   attention-based independent experiment that addresses single→multi-label
   gap and global/local representation". Explicitly excluded already-tried
   techniques (max_prob, multiscale, FAISS-kNN, temporal prior_blend, ConvNeXt
   ensemble, KL distillation, JPEG round-trip, border crop, hard geo mask,
   hflip/rot TTA).
2. **Independence boundary settled**: must NOT consume Alper's artefacts
   (`aerten3/...` FAISS index, ConvNeXt parquets, multiscale Exp-7 parquets,
   geo mask, temporal-fusion code). Murilo's own `grid_NxN.parquet` (probs
   only) + the team-shared train CLS embeddings (1.4 M × 768-d) + public
   ViT-B checkpoint are fair game. Fresh per-tile CLS extraction (~2 GPU-h)
   is in budget; fresh 1.4 M-image train extraction is not.
3. **Venue settled**: CLEF 2026 working notes — pragmatic engineering OK,
   ablation table required, no cross-task novelty claims needed.
4. **Goal settled**: independent methodological contribution. Leaderboard
   delta is explicitly *not* the bar; an F1 ∈ [0.20, 0.35] result with a
   clean ablation is the deliverable.
5. **Output of deep research**: TileQ-Decoder design — small (~4 M-param)
   label-as-query transformer decoder over per-tile CLS embeddings, trained
   on synthetic CLS-domain pseudo-quadrats with Asymmetric Loss. Headline
   ablation: **Sampler A** (random-background pseudo-quadrats) vs
   **Sampler B** (test-anchored, FAISS-relabelled backgrounds).
6. **Repo adaptation**: deep-research output suggested `docs/` and a flat
   `src/` layout; rewrote the spec to fit Murilo's existing
   `experiments/NNN-*/` + `user/murilo/src/plantclef/` library convention.
   Test CLS extraction adds a `--save-cls` flag to exp-001's `inference.py`
   (reuses `TileDataset`, `load_model`, GPU-monitor); cached test CLS lands
   at `shared/inference/dinov2-vit-base/cls_grid_4x4.parquet` alongside the
   existing 10 prob parquets.
7. **A0 baselines free**: A0a (`max_prob` top-10) + A0b (sorted-prob top-9,
   F1=0.30810 reference) both producible from existing exp-002
   `classify.py` with no new code.
8. **Geo mask**: recommended SKIPPED for v1 to keep the comparison clean and
   avoid Alper-dependence. Independent rebuild from
   `metadata_lucas/LUCAS_metadata_SWE_Wide_77k.csv` is documented in §2.6
   of the spec for v2.

PR #15 contains: `proposal.md`, `spec.md`, `tasks.md`, `pyproject.toml`,
empty `artifacts/`, empty `sbatch/`, `src` symlink → `../../src`.

### 2026-05-02 AM — deep-research briefing written

Built `user/murilo/notes/deep-research-briefing.md` (249 lines, committed in
`cc66fc5`). Inventories cached processed data with concrete schemas:

- **`shared/inference/dinov2-vit-base/`** — Murilo's exp-001 output: 10
  parquets (`no_grid` + `grid_2x2 … grid_10x10`), ~12 GB total, schema
  `image_name (str), tile_idx (int32), probs (binary float16[7806])`, probs
  only.
- **`shared/embeddings/train/sample_id={0..19}/`** — 4.39 GB,
  **1,408,033 rows**, 7,806 unique species, schema
  `image_name, species_id, cls_embedding (list<float32>[768])`, **raw CLS,
  not L2-normalised** (mean L2 ≈ 18.6, std 3.2 — must normalise before
  cosine retrieval). Hash-partitioned by `image_name`, so all 20 partitions
  must be read for any species-level aggregation.
- **Alper's parquets** at `aerten3/parquet/.../scale_NxN.parquet` — different
  schema (embeddings + logits + max_prob + passed_aug_consistency), Lanczos
  + JPEG-q85 preproc. **Off-limits** under the independence boundary.
- Class distribution: median 176 images/species, max 823, **609 species
  (7.8 %) have <5 images** — the rare-species long tail that motivates
  group-FFN over per-class queries in TileQ-Decoder.

Used as the input briefing for the deep-research session.

### Earlier — exp 002 reproduces 2025 baseline at F1=0.30810 (2026-03-13)

Self-contained `classify.py` implements `sorted-prob` (default), `union`,
`expanded-list`, `max-prob`, with optional `--quadrat-agg`. **Sorted-prob
top-k=9 on grid_4x4 = F1 0.30810** — 99.9 % exact match (2103/2105 images
identical) to the 2025 reference CSV `dsgt_run_topk_9_grid_4x4.csv`. This
establishes that the 2025 method was sorted-prob despite committed code
implying union or `SubmissionTask`. Full results table:

| Method | top-k | max_species | Avg / image | F1 |
|---|---|---|---|---|
| Sorted-prob | 9 | — | 5.0 | **0.30810** |
| Expanded-list | 5 | 20 | ~10 | 0.23 |
| Union | 5 | any | 5–37 | 0.19–0.21 |
| Quadrat agg on union | 5 | 10 | 10 | 0.17103 (worse) |

**Key finding**: probability magnitude beats tile frequency. Quadrat-level
aggregation hurts at this aggregation strength.

### Earlier — exp 001 grid sweep + parquet pipeline (2026-03-10 → 03-12)

Built `user/murilo/src/plantclef/` shared library (`model.py`, `tiling.py`,
`inference/dataset.py`, `gpu_monitor.py`). Two-stage pipeline:
`inference.py` (GPU → parquet) + `classify.py` (CPU → CSV). Ran all 10 grid
sizes; cached at `shared/inference/dinov2-vit-base/`. GPU profiling: 78.4 %
GPU util at 3×3, **99.6 % data-loading bottleneck** (NFS reads). No-grid
baseline F1 = 0.14506.

---

## What needs to happen next

### Priority 1 — Spec review (no code)

**Decision needed before any implementation**: is exp 003 spec correct as
written, or does it need revision? Specifically:

- Is the Sampler A / Sampler B contrast the right ablation? Or should A be
  replaced by a stronger naive baseline (e.g. real-train-images with no
  multi-label structure, single-positive-per-quadrat)?
- Is `n_positives = 5` the right fixed value, or should it be sampled from a
  distribution matching the inferred 2025 species-per-image (avg 5, but with
  visible spread)?
- Should A0b (sorted-prob top-9) be the comparison anchor instead of A0a
  (max_prob)? A0b is the validated F1=0.30810 baseline; A0a is the
  organizers' notebook approach (~0.23 from Alper's notes).
- Is geo mask off (v1) the right call, or should the independent
  LUCAS-rebuild go in v1?

If the spec is approved, move to Priority 2.

### Priority 2 — Derisk submission pipeline (A0a + A0b, no training)

Free F1 baselines that validate the Kaggle submission format **before**
spending 4–8 GPU-h on decoder training. Both producible from existing
`experiments/002-classification-methods/classify.py`:

- A0a: `python classify.py --grid-size 4 --method max-prob --output-dir submissions/`
- A0b: `python classify.py --grid-size 4 --method sorted-prob --top-k 9 --output-dir submissions/`

Submit both; confirm Kaggle accepts them. Reproduces F1=0.30810 for A0b.
**Block training until this passes.**

### Priority 3 — Test-CLS extraction (~2 GPU-h)

Add `--save-cls` flag to `experiments/001-dinov2-grid-ablation/inference.py`.
Use `model.forward_features(...)` + `model.forward_head(..., pre_logits=True)`
to extract the 768-d CLS instead of (or in addition to) probs. Run for
grid_4×4 only on 2,105 test images → 33,680 rows → output at
`shared/inference/dinov2-vit-base/cls_grid_4x4.parquet`.

Validate row count = 33,680 and per-image groups = 16 before training.

### Priority 4 — Library code + sanity checks

- `user/murilo/src/plantclef/tileq_decoder.py` — `TileQDecoder` class
  (spec §2.1).
- `user/murilo/src/plantclef/asl_loss.py` — Asymmetric Loss (spec §2.2).
- `user/murilo/src/plantclef/data/samplers.py` — Sampler A + B
  (spec §2.3).
- `experiments/003-tileq-decoder/{build_train_lookup, build_faiss_background_map}.py`
  — preprocessing.
- Sanity checks (overfit-one-batch, shape audit, sampler unit tests)
  **must pass** before launching long training (spec §4.3).

### Priority 5 — Two training runs (~8 GPU-h total)

Train Run 1 (Sampler A, 80 k steps, ~4 h on RTX 6000) and Run 2 (Sampler B,
same config) — sequentially or in parallel on two GPUs. Save checkpoints
every 10 k steps. Loss curves to `artifacts/`.

### Priority 6 — Inference + headline submissions

A1 (Sampler A) + A2 (Sampler B) → 2 Kaggle submissions. Record scores in
`results.md` ablation table.

### Priority 7 — A3 score-blend bonus (only if time permits)

α-blend decoder probs with cached `max_prob` baseline for
α ∈ {0, 0.25, 0.5, 0.75, 1.0}. 5 CSVs, no retraining.

### Priority 8 — Results writeup

`user/murilo/experiments/003-tileq-decoder/results.md` filled out per
proposal.md success criteria. Decide H1, H2 outcomes. Note wall-clock per
stage for paper appendix. Note any failure modes hit (§7 of spec).

### Deferred / out of scope (paper Discussion section)

- DINOv2 attention-rollout for background tile selection.
- LUCAS land-cover Bayesian prior.
- Abstain query (K+1 logit absorbing background tiles).
- Per-class F1-optimal thresholding.
- Multiscale tile inputs (3×3, 5×5, 6×6).
- Variation of `n_positives` per pseudo-quadrat.
- End-to-end fine-tuning (decoder + backbone).

---

## Key patterns (cumulative findings worth remembering across sessions)

- **Probability magnitude > tile frequency** (exp 002, 2026-03-13). Sorted-prob
  (rank by softmax probability) F1 0.30810 vs union (rank by tile count)
  F1 0.19–0.21 on the same per-tile predictions. This is the load-bearing
  finding for any decoder over per-tile probs: confidence matters more
  than agreement.
- **Quadrat aggregation hurts at low aggregation strength** (exp 002).
  Grouping per-image lists by physical plot and unioning made F1 drop
  0.21 → 0.17. The per-image lists are too noisy to benefit from cross-visit
  union; would need to retry on top of a stronger per-image classifier.
- **Data-loading is the inference bottleneck**, not GPU compute (exp 001).
  ~99 % of step time on NFS reads at grid_3×3, GPU util ~78 %. Any future
  inference experiment should consider local-scratch staging or webdataset.
- **Train CLS embeddings are NOT L2-normalised** (verified 2026-05-02). Mean
  L2 ≈ 18.6, std 3.2, range [11.8, 29.5]. **Must normalise before cosine
  retrieval / `IndexFlatIP`**. Raw scale is what TileQ-Decoder's input
  projection sees during training (no need to normalise there — let the
  LayerNorm + linear adapt).
- **Train data is hash-partitioned by `image_name`** (verified 2026-05-02).
  Every species spread across all 20 `sample_id={0..19}` partitions. Must
  read all 20 for species-level aggregation (centroids, prototypes, FAISS).
- **Long-tail species are 7.8 %** (verified 2026-05-02). 609 of 7806 species
  have <5 images. 1,062 (13.6 %) have <10. This is why TileQ-Decoder uses
  group-FFN (64 groups × 122 classes) instead of 7806 individual queries —
  per-class queries would gradient-starve.
- **Kaggle is the only test signal**. No public ground-truth labels for the
  2,105 test images. Synthetic-only val loss in training only confirms
  fitting the training distribution. Each Kaggle submission costs a slot
  (~5/day, 25 over 5 days). Plan submissions, don't burn slots.
- **Independence boundary is paper-driven, not technical**. Alper's
  artefacts work and would lift the score; the rule against using them is
  to keep the methodological contribution clean for the working notes
  paper. Re-evaluate after the paper is drafted if a follow-up wants to
  ensemble back in.

---

## Key files for current work

| File | Purpose |
|---|---|
| `user/murilo/experiments/003-tileq-decoder/spec.md` | **Active experiment ground truth.** Architecture, samplers, training, ablation table, failure modes, paths. |
| `user/murilo/experiments/003-tileq-decoder/proposal.md` | H1/H2, success/reject criteria. |
| `user/murilo/experiments/003-tileq-decoder/tasks.md` | Implementation checklist. |
| `user/murilo/experiments/003-tileq-decoder/pyproject.toml` | Adds `torch`, `timm`, `faiss-cpu`, `pyyaml` on top of exp-001 deps. |
| `user/murilo/experiments/001-dinov2-grid-ablation/inference.py` | **To be extended** with `--save-cls` flag for test-CLS extraction (Priority 3). |
| `user/murilo/experiments/002-classification-methods/classify.py` | **A0a/A0b producer** — no new code needed for baselines. |
| `user/murilo/src/plantclef/{model.py, tiling.py, inference/dataset.py, gpu_monitor.py}` | Shared library — extend with `tileq_decoder.py`, `asl_loss.py`, `data/samplers.py`. |
| `user/murilo/notes/deep-research-briefing.md` | Cached-data inventory (paths, schemas, stats, candidate angles). |
| `user/alper/src/Final_pipeline_methods_Summary/Final_Pipeline_Methods_Summary.md` | Team best F1=0.4379 reference. Read but **do not consume** the artefacts it points to. |
| `shared/inference/dinov2-vit-base/grid_NxN.parquet` | Murilo's cached test probs, 10 grids. Schema `(image_name, tile_idx, probs)`, float16 bytes. |
| `shared/inference/dinov2-vit-base/cls_grid_4x4.parquet` | **TO BE CREATED** (Priority 3). Test CLS embeddings, schema `(image_name, tile_idx, cls_embedding)` float16[768]. |
| `shared/embeddings/train/sample_id={0..19}/` | Team-shared 1.4 M train CLS embeddings, 4.39 GB. Schema `(image_name, species_id, cls_embedding)`. |
| `shared/data/raw/class_id_to_species_mapping/{class_mapping.txt, class_to_species_id.json, species_id_to_name.txt}` | Index ↔ species_id ↔ Latin name. |
| `shared/data/raw/PlanCLEF_test_data/{images/, PlantCLEF2025_test.csv}` | 2,105 test JPGs + quadrat metadata. |
| `shared/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth` | Frozen backbone for the decoder. |
| `CLAUDE.md` (project root) | OPSX workflow, doc slash commands, PACE Phoenix conventions. |
