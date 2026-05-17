# Session Context

Session handoff document for Murilo's PlantCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

**Last updated**: 2026-05-17. **Exp 003 complete; PR #15 squash-merged
to main** as commit `9208c07` on 2026-05-17; `mg/tileq-decoder` branch
deleted (both remote and local). **Headlines**: A1@k9 = 0.12180
(H1 anchor), A1@k10 = 0.11640, A2@k9 = 0.09060 (H2 contrast),
A2@k10 = 0.08445.
**A3 α-curve monotone decreasing**: α=0 (max_prob only) = 0.22144;
α=1 (decoder only) = 0.11640. **No α > 0 improves over α=0** — decoder
provides zero complementary signal.
**H1 rejected** decisively. **H2 rejected, opposite direction predicted**
(B 0.031 *below* A). **A3** confirms decoder is purely additive noise
(stronger result than spec's complementary-vs-redundant framing). New
Sampler B failure mode (FAISS-undersampling under-confidence) documented
in `results.md`. **P8 paper writeup handed off**: 1,159-line
`paper-briefing.md` synthesizes all four exp-003 docs into a single
narrative-ordered input for Claude Research to draft the CLEF 2026
working-notes paper. Competition closed **2026-05-07**; **no active
PlantCLEF work planned**.

---

## Working principles (read every session)

Adapted from
[Karpathy CLAUDE.md](https://github.com/forrestchang/andrej-karpathy-skills/blob/main/CLAUDE.md).
These are the operating contract for any Claude session in this codebase —
they apply *before* any specific task. **Tradeoff**: bias toward caution
over speed. For trivial tasks, use judgment.

### 1. Think before coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, **ask**.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. **Push back when warranted.**
- If something is unclear, stop. Name what's confusing. Ask.

In this 5-day-deadline context: the cost of one clarifying question is far
below the cost of a wrong direction.

### 2. Simplicity first

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- **If you write 200 lines and it could be 50, rewrite it.**

Quality test: *"Would a senior engineer say this is overcomplicated?"* If
yes, simplify.

### 3. Surgical changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently. This codebase has
  conventions across `experiments/NNN-*`, `src/plantclef/`, and the
  `proposal.md`/`spec.md`/`tasks.md`/`results.md` quartet — follow them.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that **YOUR changes** made unused.
- Don't remove pre-existing dead code unless asked.

**The test**: every changed line should trace directly to the user's
request. If a line in your diff can't be tied to what was asked, drop it.

### 4. Goal-driven execution

**Define success criteria. Loop until verified.**

Transform vague tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"
- "Run experiment 003" → "Hypothesis H1/H2 defined; A0a/A0b submitted;
  A1/A2 F1 deltas vs A0b recorded in `results.md`"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make
it work") require constant clarification.

### 5. Independence boundary (plantclef-specific)

This workspace's experiments must **NOT** extend Alper's pipeline. Do not
consume artefacts from `aerten3/` (FAISS index, ConvNeXt weights, geo mask,
Exp-7 multiscale parquets, temporal-fusion code). Use only Murilo's cached
parquets, the team-shared train CLS embeddings, the public ViT-B checkpoint,
raw test/train images, and public metadata. The independence boundary is a
paper-contribution requirement, not a technical one — see `proposal.md` for
each experiment for the rationale.

---

**These guidelines are working if**: fewer unnecessary changes in diffs,
fewer rewrites due to overcomplication, and clarifying questions come
**before** implementation rather than after mistakes.

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

**Competition deadline**: 2026-05-07 (~2 days remaining).

**Active branch**: none — `mg/tileq-decoder` deleted after PR #15 was
squash-merged to main on 2026-05-17 as commit `9208c07` ("exp 003
(TileQ-Decoder): full experiment + paper briefing"). 52 files changed,
+8,663 / −32 in the squash. Pre-merge milestone commits (now collapsed):
P4 library `ad898bb`, P5 training `9febc5f`, P6 inference/headlines
`3d88553` + `eab81f3`, P7 A3 α-blend `27779f5`, paper-briefing `8ba24ea`.

**Recent merges to main**:

- **PR #15 → exp 003 (TileQ-Decoder full experiment + paper briefing) — merged 2026-05-17 (squash commit `9208c07`)**
- PR #14 → Alper's Exp-10 ConvNeXt finetuning
- PR #7 → exps 001 + 002 (Murilo, `mg/dinov2-grid-ablation`) — merged 2026-04-18-ish

| Submission | F1 | Status |
|---|---|---|
| **Alper exp 11 final ensemble** | **0.4379** | Public LB, 2026-04-17 |
| **A0b** sorted-prob top-9, grid_4x4 (exp 003 derisk) | **0.30810** | Public LB, 2026-05-03 (exact reproduction of 2025 reference) |
| Sorted-prob top-9, grid_4x4 (Murilo exp 002 — 2025 reproduction) | 0.30810 | validated 2026-03-13 |
| Expanded-list top-5 max-20, grid_4x4 (exp 002) | 0.23 | |
| **A0a** max-prob top-10, grid_4x4 (exp 003 derisk) | **0.22986** | Public LB, 2026-05-03 |
| Union top-5 (exp 002) | 0.21313 | |
| No-grid baseline (exp 001) | 0.14506 | |
| **A1@k9** TileQ-Decoder Sampler A top-9 (exp 003 H1 anchor) | **0.12180** | Public LB, 2026-05-05 — **H1 rejected** |
| **A1@k10** TileQ-Decoder Sampler A top-10 | **0.11640** | Public LB, 2026-05-05 |
| **A2@k9** TileQ-Decoder Sampler B (τ=0.65) top-9 (exp 003 H2 contrast) | **0.09060** | Public LB, 2026-05-05 — **H2 rejected, opposite direction** |
| **A2@k10** TileQ-Decoder Sampler B (τ=0.65) top-10 | **0.08445** | Public LB, 2026-05-05 |
| **A3 α=0.00** (pure max_prob, A0a sanity dup) | **0.22144** | Public LB, 2026-05-06 — best of A3 sweep |
| A3 α=0.10 | 0.21952 | Public LB, 2026-05-06 |
| A3 α=0.20 | 0.21827 | |
| A3 α=0.30 | 0.21001 | |
| A3 α=0.40 | 0.19954 | |
| A3 α=0.50 | 0.18931 | |
| A3 α=0.60 | 0.17292 | |
| A3 α=0.70 | 0.15277 | |
| A3 α=0.80 | 0.14562 | |
| A3 α=0.90 | 0.13154 | |
| **A3 α=1.00** (pure decoder, = A1@k10) | **0.11640** | Public LB, 2026-05-06 |

---

## What happened recently

### 2026-05-17 — PR #15 squash-merged; paper briefing handed off to Claude Research

Drafted `user/murilo/experiments/003-tileq-decoder/paper-briefing.md`
(1,159 lines, commit `8ba24ea` on `mg/tileq-decoder` before merge):
self-contained narrative-ordered synthesis of proposal.md / spec.md /
tasks.md / results.md, intended as the single input artefact for a
Claude Research session to draft the CLEF 2026 working-notes paper.
Every number reproduced inline; no internal repo links to follow.
Thirteen sections following the paper's natural arc — TL;DR →
context → hypotheses → method → setup → training (synthetic) →
Kaggle results → mechanism analysis → contributions synthesis →
auxiliary findings → discussion/future work → reproducibility →
references → suggested paper structure. Mirrors the
`deep-research-briefing.md` pattern that originally seeded exp 003.

PR #15 squash-merged to main as commit `9208c07` on 2026-05-17 with a
commit message summarising the full PR arc (verdict, new failure mode
characterised, all deliverables). 52 files / +8,663 / −32 squashed.
`mg/tileq-decoder` branch deleted both remote and local; stale
remote-tracking refs pruned. **No active PlantCLEF work planned**
beyond running the paper briefing through Claude Research externally.

### 2026-05-06 early morning — A3 α-blend done; decoder is purely additive noise

11-point α sweep over A1 (Sampler A, the better-LB decoder source at
0.12180) blended with cached `max_prob` baseline at fixed top-k=10.
Generated all CSVs locally via `blend.py` (no decoder forward — the
per-image probs were already cached from `infer.py`), submitted all 11
to Kaggle in a single loop (~3 s sleep between each).

**α-curve is monotonically decreasing**: every increase of α reduces LB.

| α | LB |
|---:|---:|
| **0.00** (pure max_prob) | **0.22144** |
| 0.10 | 0.21952 |
| 0.20 | 0.21827 |
| 0.30 | 0.21001 |
| 0.40 | 0.19954 |
| 0.50 | 0.18931 |
| 0.60 | 0.17292 |
| 0.70 | 0.15277 |
| 0.80 | 0.14562 |
| 0.90 | 0.13154 |
| **1.00** (pure decoder) | **0.11640** |

Sanity checks:
- α=1.00 LB exactly matches A1@k10 (0.11640). Blend math confirmed
  byte-correct at the boundary.
- α=0.00 LB = 0.22144 vs A0a's 0.22986 — 0.008 gap from tie-breaking
  differences (numpy `argpartition+argsort` vs pandas `sort` on species
  with near-equal softmax probabilities; ~80 of 2105 rows have positions
  8/9 or 9/10 swapped, occasionally a different boundary species at
  rank 10 in the strictest cases). Same top-10 sets in 96 % of rows.
  Relative A3 curve internally consistent (all 11 use same tie-breaking).

**Stronger than expected**: spec §2.5 framed the question as
"complementary vs redundant" with the implicit assumption that some
intermediate α might lift performance. Reality is a third outcome:
**best α is exactly 0**, with monotone-worse blends — i.e., decoder is
not even partially redundant; it actively destroys F1 at any non-zero
weight. This is the strongest possible "decoder is noise" result.

### 2026-05-05 — P4-P6 landed; H1 rejected, H2 rejected (opposite direction)

**P4 (library code + sanity checks) done.** Library: `asl_loss.py`
(`AsymmetricLoss(γ⁻=4, γ⁺=0, m=0.05)` per spec §2.2),
`tileq_decoder.py` (4,509,824 params; matches spec §2.1 exactly),
`data/samplers.py` (`SamplerA` naive + `SamplerB` test-anchored).
Build pipeline: `build_train_lookup.py` (1.4 M train CLS sorted-by-class
into a flat npz with offsets, 4.1 GB), `build_faiss_background_map.py`
(`IndexFlatIP` over 1.4 M, 33,680 queries, ~10 min CPU),
`rethreshold.py` (re-apply τ on cached FAISS results in seconds),
`dataset.py` (multi-worker-safe `IterableDataset`),
`sanity_check.py` (shape audit + overfit-one-batch gate).
**88 unit tests all passing.** Sanity check on real artefacts:
both samplers reduce loss > 13,000× in 200 steps, ratio well above the
5× gate. Commits `ad898bb`, `9febc5f`, `3d88553`.

**FAISS τ tuned to 0.65 from histogram, overriding spec's 0.85 default.**
The cosine-similarity distribution between test tiles and their nearest
train neighbour is **unimodal centred at 0.51, max 0.89** — *no shoulder*
at the spec's 0.85 prior (would have labelled only 18 of 33,680 tiles).
At τ = 0.65: 1,838 tiles labelled (5.46 %, ~0.9 surviving bg labels per
pseudo-quadrat avg). Spot-check at the high-sim tail confirmed
biological coherence — *Eryngium maritimum* × 4 tiles in one quadrat,
*Quercus coccifera* × 5 across consecutive CBNMed sites. Histogram +
percentile/coverage table committed as decision artefacts in
`artifacts/`.

**P5 (training Sampler A + Sampler B) done — 12 min wallclock each.**
Both runs at 80,000 steps × batch 32 = 2.56 M pseudo-quadrats trained on,
matching spec §4.1 hyperparameters exactly. RTX 6000 partition held
(no V100 silent reroute this time). Sampler A job 8038028 wallclock
12:05 (9.4 min training + venv install), 142.5 step/s, final train/val
loss 4.91 / 4.85, val acc@10 = 99.65 %. Sampler B job 8038496 wallclock
11:46, 146.7 step/s, final loss 0.93 / 0.92, val acc@10 = 100 %.
Both checkpoints (52 MB each) saved at
`artifacts/checkpoints/decoder_sampler{A,B}.pt`. W&B project
`plantclef-2026-exp003`, runs `u5ma36yz` (A) and `hdcx1p23` (B).
Loss curves committed as PNG decision artefacts.

**Notable: B's 5.3× lower synthetic val loss did NOT predict higher LB.**
At training time we observed B's loss collapsing to ~0.93 vs A's ~4.85,
and per-image prediction distributions had B much sharper (median 0
classes > 0.5) and more diverse (3,737 unique top-9 classes vs A's
1,874). We read this then as encouraging for H2; the LB now reveals
it as the warning sign — B was learning calibrated under-confidence
on test-shaped inputs because most of those tiles had no FAISS label.

**P6 (inference + 4 headline submissions) done.** Single `infer.py`:
load checkpoint, forward over the 2,105 test quadrats (7-8 s on
login-node CPU), sigmoid → top-k decode at multiple k from one pass
(default `--topk 5,7,9,10,12,15`). Per-image probability vector also
persisted to `probs_<tag>_seed42.npy` (66 MB) so `blend.py` can α-mix
without re-running inference. **CSV format bug**: initial submissions
errored "ID column quadrat_id not found" because the writer used `;`
delimiter (matched the spec text but not exp 002's actual
`pandas.to_csv(sep=",", quoting=csv.QUOTE_ALL)` format). Fixed in
`eab81f3`; re-generated all 12 CSVs.

**Headline LB results (2026-05-05)**: A1@k9 = **0.12180**,
A1@k10 = 0.11640, A2@k9 = **0.09060**, A2@k10 = 0.08445.

- **H1 rejected decisively**: A1=0.12180 is **0.166 below** the spec's
  reject threshold of 0.288, **0.186 below** the A0b anchor (0.30810).
  Decoder underperforms even the no-grid full-image baseline (0.14506
  from exp 001) — i.e., the decoder destroys signal that's already in
  the cached per-tile probs.
- **H2 rejected, opposite direction predicted**: A2 − A1 = −0.0312;
  Sampler B significantly *worse* than Sampler A, not better. The
  proposal didn't enumerate this third state — only "≥ +0.01 confirm"
  vs "|.| ≤ 0.005 reject". Worth flagging in the paper.

**New failure mode discovered (worth a Discussion paragraph)**: at
τ=0.65 only 5.5 % of test tiles get FAISS labels, so 94.5 % of B's
pseudo-quadrats have 11 unlabelled-test-tile slots → decoder learns
"test-shaped tile → no species in label" → outputs uniformly low
logits at inference → forced top-k picks noise. Spec §7.3
(synthetic-distribution overfit) was about Sampler A and DID hold
empirically, but it was NOT the binding failure mode for our LB —
A actually held up better than B did. The new mode (FAISS-undersampling
under-confidence) is documented in `results.md` § "Mechanism analysis".

**Kaggle daily cap correction**: the spec §4.2 + earlier CONTEXT.md
note of "~5/day" was wrong. Verified with user 2026-05-05: cap is
**100/day**. Plenty of headroom for the planned A3 sweep
(11 submissions) plus k-sweep extras (8 ready in `submissions/`).
Memory updated.

### 2026-05-04 late evening — Priority 1 closed; Priority 3 done; sbatch convention finalized

**Priority 1 (spec review) closed.** H1 success criteria in `proposal.md`
explicitly tightened to **A0b = 0.30810** as the anchor (decoder must clear
this). A0a (0.22986) is reference, not anchor — beating it is necessary
but not sufficient. Inference produces **both k=9** (apples-to-apples vs
A0b) **and k=10** (organizers' convention) from one decoder forward pass —
4 decoder CSVs (A1@k9, A1@k10, A2@k9, A2@k10) instead of 2. Other
questions ratified (no spec change): keep Sampler A as the naive contrast,
fix `n_positives = 5` for v1, geo mask off for v1.
`submissions/SUBMISSIONS.md` log added — kaggle CLI message convention +
A0a/A0b backfilled. Commit `462134e`.

**Priority 3 (test-CLS extraction) done.** `--save-cls` flag added to
exp 001's `inference.py` (`forward_features` + `forward_head(pre_logits=True)`
→ 768-d CLS, sibling parquet `cls_<grid_label>.parquet`). Sbatch
`experiments/003-tileq-decoder/sbatch/extract_test_cls.{sh,sbatch}` runs
the extraction (V100, 16 min, 0.46 s/image, data-loading-bound as in
exp 001). Output:
**`shared/inference/dinov2-vit-base/cls_grid_4x4.parquet` — 49.6 MB,
33,680 rows × float16[768]**. External validation passed all checks: schema
matches spec, 2,105 unique images each with exactly 16 tiles, joinable to
`grid_4x4.parquet` on `(image_name, tile_idx)`.
Commits `1811f94`, `442bc01`, `76ea227`, `8b61690`.

**Test CLS L2 distribution observed** (sampled 2,000 tiles): mean **25.4**,
std 2.8, range [13.3, 30.0], p10/p50/p90 = 21.3 / 26.1 / 28.3. **Higher
than train CLS mean ~18.6** (vegetation-scene tiles vs single-plant
images — different framing, content). Not a problem for the decoder
(`proj_kv` linear adapts the absolute scale during training), but worth
recording for FAISS τ tuning in §3.4 — when querying L2-normalized test
tiles against L2-normalized train CLS, the cosine-sim histogram should
still produce a clean shoulder for τ selection.

**Sbatch convention finalized**: per-task `<task>.sh` (launcher) +
`<task>.sbatch` (sbatch with `#SBATCH` directives + venv setup + job).
Iterated through several layouts (single sbatch → birdclef-style
`_job.sh+run.sh` → per-task explicit names). Final user-facing command:
`bash sbatch/<task>.sh [--fast-dev-run|--gpu h100|--dry-run]`. Direct
sbatch invocation (`sbatch sbatch/<task>.sbatch`) is also supported via
positional-arg parsing in the sbatch. Future tasks (build-FAISS,
train-A/B, infer) copy `extract_test_cls.{sh,sbatch}` and modify.

**Two practical gotchas discovered:**
1. **`paceship-clef2026_plantclef` account silently routes to `gpu-v100`**
   even when `--partition=gpu-rtx6000` is set. Not blocking — V100 16 GB
   is enough for ViT-B inference, and exp 001's data-loading bottleneck
   means GPU choice doesn't move the needle. If RTX 6000 access is needed
   later, check `sacctmgr show assoc account=paceship-clef2026_plantclef`.
2. **Default-PyPI torch installs `2.11.0+cu130`**, but PACE driver supports
   max CUDA 12.9 → silent CPU fallback. Fix in
   `extract_test_cls.sbatch`: pin `"torch>=2.1.0,<2.11"` with
   `--index-url cu121`. Resolves to `torch==2.5.1+cu121`. Bake this same
   pin into all future exp 003 sbatchs.

### 2026-05-03 → 2026-05-04 — A0 derisk submitted; venv restructured to mirror birdclef

**A0 baselines submitted to Kaggle**, both via existing
`experiments/002-classification-methods/classify.py` against the cached
`shared/inference/dinov2-vit-base/grid_4x4.parquet` — no new code:

- **A0a** (`max-prob` top-10, `--max-species 10`): LB **0.22986**, matches
  expected ~0.23 from organizers' notebook approach.
- **A0b** (`sorted-prob` top-9): LB **0.30810**, exact-match reproduction of
  the validated 2025 baseline. Avg 5.0 species/image as documented.

CSV format accepted by Kaggle on first submission. Submission pipeline
**derisked** for the headline runs (A1/A2). A0b is the comparison anchor
for H1 in `proposal.md` (decoder must clear 0.30810 to confirm the
hypothesis). Results in `experiments/003-tileq-decoder/results.md`.

**Workspace venv restructured** to mirror birdclef's pattern. Driver was
needing `kaggle` CLI in the project venv:

- Old layout: root `pyproject.toml` declared a uv workspace with
  `members = ["user/*/experiments/*"]`. Per-experiment pyprojects auto-
  discovered. Root `.venv` had no umbrella source of truth.
- New layout: root `pyproject.toml` + `uv.lock` **deleted**. New
  `user/murilo/pyproject.toml` is the umbrella (torch, timm, faiss-cpu,
  pillow, pandas, pyarrow, kaggle, tqdm, matplotlib, pyyaml, nvidia-ml-py +
  dev extras `pytest`/`ruff`/`ipython`). Single `.venv` at
  `clef/plantclef-2026/.venv` populated via `uv pip install -e user/murilo`.
- Per-experiment `pyproject.toml` files (001, 002, 003) **retained
  unchanged** — they remain the source of truth for `/tmp` sbatch venvs
  built by `_job.sh` via `uv pip install -e "$EXPERIMENT_DIR"`. Independent
  of the umbrella.
- Activation pattern: `source .venv/bin/activate` from project root, then
  `cd user/murilo`. Identical to birdclef.
- Caveat: `uv sync --package <experiment>` from project root is gone (no
  workspace). To install an experiment's deps, `cd` into its directory and
  use its pyproject. Sbatch already works that way.

`uv.lock` files are gitignored (line 36 of `.gitignore`); CSVs are
gitignored (line 115). Submission CSVs are reproducible from `classify.py`,
so not in repo.

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

### Priority 1 — Spec review (no code) — DONE 2026-05-04

Closed. Outcomes:

- **Sampler A as spec'd kept** — replacing it would conflate H1 and H2.
- **`n_positives = 5` fixed** for v1 — variable distribution deferred to §8.
- **A0b = 0.30810 confirmed as the H1 anchor** — `proposal.md` success
  criteria tightened to drop A0a/A0b ambiguity. Reject H1 if A1 < 0.288.
- **Geo mask off for v1** — keep ablation clean; LUCAS rebuild deferred
  to v2.

Bonus side-effects:
- Inference now produces both k=9 and k=10 from one forward pass
  (4 decoder CSVs from 2 checkpoints).
- `submissions/SUBMISSIONS.md` is the source-of-truth log for kaggle CLI
  message convention + per-submission LB. A0a / A0b backfilled.

### Priority 2 — Derisk submission pipeline (A0a + A0b, no training) — DONE 2026-05-03

A0a (LB 0.22986) and A0b (LB 0.30810) both submitted and accepted. Pipeline
clean. Details in `experiments/003-tileq-decoder/results.md`.

### Priority 3 — Test-CLS extraction (~2 GPU-h) — DONE 2026-05-04

Job 7966845 on V100, 16 min. Output:
`shared/inference/dinov2-vit-base/cls_grid_4x4.parquet` — 49.6 MB,
33,680 rows × float16[768]. Schema as spec'd. Validation passed (2,105
images, 16 tiles each, joinable to `grid_4x4.parquet`).

### Priority 4 — Library code + sanity checks — DONE 2026-05-05

Full library shipped: `tileq_decoder.py` (4.5 M params), `asl_loss.py`,
`data/samplers.py` (A + B), `build_train_lookup.py`,
`build_faiss_background_map.py`, `rethreshold.py`, `dataset.py`,
`sanity_check.py`. **88 unit tests passing.** Both samplers' overfit-
one-batch ratio > 13,000× in 200 steps. FAISS τ tuned empirically to
0.65 (spec's 0.85 default produced near-zero coverage). Commits
`ad898bb`, `9febc5f`.

### Priority 5 — Two training runs — DONE 2026-05-05

Sampler A job 8038028 (RTX 6000, 12:05) and Sampler B job 8038496
(RTX 6000, 11:46), both at spec's 80 k steps × batch 32 = 2.56 M
pseudo-quadrats. Final losses: A 4.91 / 4.85 (train/val), B 0.93 / 0.92.
Checkpoints `decoder_sampler{A,B}.pt` (52 MB each).
W&B runs: A `u5ma36yz`, B `hdcx1p23`.

### Priority 6 — Inference + headline submissions — DONE 2026-05-05

`infer.py` runs in seconds on login-node CPU; persists per-image probs
for `blend.py` to reuse. **All 4 headlines submitted to Kaggle**:
A1@k9 = **0.12180** (H1 anchor — 0.186 below A0b = 0.30810,
**H1 rejected**), A1@k10 = 0.11640, A2@k9 = **0.09060**
(H2 contrast — 0.031 *below* A1, **H2 rejected, opposite direction**),
A2@k10 = 0.08445. Both findings publishable per proposal.md.

8 additional k-sweep CSVs (k ∈ {5, 7, 12, 15} per sampler) ready in
`submissions/` for free additional ablation rows; not yet submitted.

### Priority 7 — A3 score-blend — DONE 2026-05-06

11 α-blend submissions all complete. **α-curve monotone decreasing**;
α=0 (pure max_prob) wins at 0.22144; α=1 (pure decoder) at 0.11640
exactly matches A1@k10 (sanity check). No α > 0 improves over α=0.
Decoder provides zero complementary signal — purely additive noise.

### Priority 8 — Results writeup — DONE 2026-05-17

`results.md` carries the H1/H2 verdicts, full ablation tables, A3
α-curve, mechanism analysis (including the new
FAISS-undersampling-under-confidence failure mode for Sampler B),
wall-clock per stage, and future-work directions. The paper-writing
hand-off is `paper-briefing.md` (1,159 lines, narrative-ordered,
self-contained, no internal repo links) — intended as the single
input for a Claude Research session to draft the CLEF 2026
working notes. PR #15 squash-merged 2026-05-17 as commit `9208c07`.

### Deferred / out of scope (paper Discussion section)

- DINOv2 attention-rollout for background tile selection (would address
  the FAISS-undersampling under-confidence failure mode discovered in B).
- LUCAS land-cover Bayesian prior.
- Abstain query (K+1 logit absorbing background tiles) — explicitly
  predict "no species" instead of low-confidence everywhere.
- Per-class F1-optimal thresholding.
- Multiscale tile inputs (3×3, 5×5, 6×6).
- Variation of `n_positives` per pseudo-quadrat.
- End-to-end fine-tuning (decoder + backbone).

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
- **Test CLS L2 is higher than train** (verified 2026-05-04). Mean ≈ 25.4,
  std 2.8, range [13.3, 30.0] across 2,000 sampled tiles. ~37 % higher
  than train mean — vegetation-scene tiles vs single-plant images. The
  `proj_kv` linear in TileQ-Decoder absorbs this scale shift during
  training. After L2-normalization both distributions live on the unit
  sphere; FAISS `IndexFlatIP` retrieval works as designed.
- **Train data is hash-partitioned by `image_name`** (verified 2026-05-02).
  Every species spread across all 20 `sample_id={0..19}` partitions. Must
  read all 20 for species-level aggregation (centroids, prototypes, FAISS).
- **Long-tail species are 7.8 %** (verified 2026-05-02). 609 of 7806 species
  have <5 images. 1,062 (13.6 %) have <10. This is why TileQ-Decoder uses
  group-FFN (64 groups × 122 classes) instead of 7806 individual queries —
  per-class queries would gradient-starve.
- **Kaggle is the only test signal**. No public ground-truth labels for the
  2,105 test images. Synthetic-only val loss in training only confirms
  fitting the training distribution. **Daily cap is 100 submissions/day**
  (verified 2026-05-05; spec §4.2 + earlier CONTEXT.md said ~5/day, that
  was wrong). Generous headroom — exp 003's full plan (4 headlines + 12
  k-sweep + 11 A3 = 27 submissions) fits inside one day.
- **Independence boundary is paper-driven, not technical**. Alper's
  artefacts work and would lift the score; the rule against using them is
  to keep the methodological contribution clean for the working notes
  paper. Re-evaluate after the paper is drafted if a follow-up wants to
  ensemble back in.
- **Synthetic val loss is uncorrelated with LB F1** (verified 2026-05-05,
  exp 003 P5/P6). Sampler B trained to ~5× lower synthetic val loss than
  Sampler A (0.93 vs 4.85), with sharper and more diverse top-k
  predictions on the test set — but on Kaggle B underperformed A by
  0.031 F1 (0.0906 vs 0.1218 at k=9). The synthetic distribution is
  different enough from the real test distribution that fitting it
  better doesn't help. **Train monitoring is a pipeline-health check,
  not a generalisation proxy** — Kaggle is the only signal that decides
  H1/H2.
- **FAISS-undersampling under-confidence** (new failure mode, 2026-05-05).
  At τ=0.65 only 5.5 % of test tiles get FAISS-discovered species
  labels. So 94.5 % of Sampler B's pseudo-quadrats have 11 unlabelled
  test-tile slots → decoder learns "test-shaped tile → no species in
  the multi-hot label" → outputs uniformly low logits at inference →
  forced top-k decode picks noise. This is **NOT** the same as spec
  §7.3's synthetic-distribution overfit (which was about A's confidently-
  wrong negative labels). Beyond the spec's enumerated failure modes;
  worth a Discussion paragraph in the paper. Mitigation directions
  in spec §8: attention-rollout tile selection, abstain query.

---

## Key files for current work

| File | Purpose |
|---|---|
| `user/murilo/experiments/003-tileq-decoder/spec.md` | **Active experiment ground truth.** Architecture, samplers, training, ablation table, failure modes, paths. |
| `user/murilo/experiments/003-tileq-decoder/proposal.md` | H1/H2, success/reject criteria. |
| `user/murilo/experiments/003-tileq-decoder/tasks.md` | Implementation checklist. |
| `user/murilo/experiments/003-tileq-decoder/paper-briefing.md` | **Paper writeup hand-off** — 1,159-line narrative-ordered synthesis of proposal/spec/tasks/results for a Claude Research session to draft the CLEF 2026 working-notes paper. Self-contained; every number reproduced inline. |
| `user/murilo/experiments/003-tileq-decoder/pyproject.toml` | Adds `torch`, `timm`, `faiss-cpu`, `pyyaml` on top of exp-001 deps. |
| `user/murilo/experiments/001-dinov2-grid-ablation/inference.py` | Extended with `--save-cls` flag (commit `1811f94`). Default behavior unchanged. |
| `user/murilo/experiments/003-tileq-decoder/sbatch/extract_test_cls.{sh,sbatch}` | Per-task launcher + sbatch (template for future build_faiss / train / infer pairs). Pin `torch>=2.1.0,<2.11` cu121. |
| `user/murilo/experiments/003-tileq-decoder/submissions/SUBMISSIONS.md` | Source-of-truth log for kaggle CLI message convention + per-submission LB. |
| `user/murilo/experiments/002-classification-methods/classify.py` | **A0a/A0b producer** — no new code needed for baselines. |
| `user/murilo/src/plantclef/{model.py, tiling.py, inference/dataset.py, gpu_monitor.py}` | Shared library — extend with `tileq_decoder.py`, `asl_loss.py`, `data/samplers.py`. |
| `user/murilo/notes/deep-research-briefing.md` | Cached-data inventory (paths, schemas, stats, candidate angles). |
| `user/alper/src/Final_pipeline_methods_Summary/Final_Pipeline_Methods_Summary.md` | Team best F1=0.4379 reference. Read but **do not consume** the artefacts it points to. |
| `shared/inference/dinov2-vit-base/grid_NxN.parquet` | Murilo's cached test probs, 10 grids. Schema `(image_name, tile_idx, probs)`, float16 bytes. |
| `shared/inference/dinov2-vit-base/cls_grid_4x4.parquet` | ✓ **Created 2026-05-04** (49.6 MB, 33,680 rows). Schema `(image_name, tile_idx, cls_embedding (binary float16[768]))`. Decode via `np.frombuffer(blob, dtype=np.float16)`. Joinable to `grid_4x4.parquet` on `(image_name, tile_idx)`. |
| `shared/embeddings/train/sample_id={0..19}/` | Team-shared 1.4 M train CLS embeddings, 4.39 GB. Schema `(image_name, species_id, cls_embedding)`. |
| `shared/data/raw/class_id_to_species_mapping/{class_mapping.txt, class_to_species_id.json, species_id_to_name.txt}` | Index ↔ species_id ↔ Latin name. |
| `shared/data/raw/PlanCLEF_test_data/{images/, PlantCLEF2025_test.csv}` | 2,105 test JPGs + quadrat metadata. |
| `shared/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth` | Frozen backbone for the decoder. |
| `CLAUDE.md` (project root) | OPSX workflow, doc slash commands, PACE Phoenix conventions. |
