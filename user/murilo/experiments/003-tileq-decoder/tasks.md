# Tasks

Reference: `spec.md` for technical detail; `proposal.md` for hypothesis framing.
Order is by **logical dependency**, not by day. Sections that don't depend on
each other can be parallelised across compute nodes / sessions.

## 0. Submission pipeline derisk (do FIRST, no training required)
- [x] 0.1 Generate **A0a** (`max_prob` top-10 on `grid_4x4.parquet`) via
      `experiments/002-classification-methods/classify.py --grid-size 4 --method max-prob --max-species 10`
      → `submissions/A0a_max_prob_grid4.csv`. Validate Kaggle accepts the format.
      **DONE 2026-05-03** — Kaggle LB **F1 = 0.22986** (matches expected ~0.23).
- [x] 0.2 Generate **A0b** (sorted-prob top-9 on `grid_4x4.parquet`) via
      `classify.py --grid-size 4 --method sorted-prob --top-k 9`
      → `submissions/A0b_sorted_prob_topk9_grid4.csv`. Confirm reproduces F1=0.30810.
      **DONE 2026-05-03** — Kaggle LB **F1 = 0.30810** (exact match).

## 1. Environment + scaffolding
- [x] 1.1 Workspace venv now driven by `user/murilo/pyproject.toml` umbrella
      (mirrors birdclef pattern; root `.venv` at `clef/plantclef-2026/.venv`).
      Includes `torch`, `timm`, `faiss-cpu`, `pandas`, `pyarrow`, `numpy`,
      `tqdm`, `matplotlib`, `pyyaml`, `pillow`, `nvidia-ml-py`, `kaggle`.
      Per-experiment `pyproject.toml` files retained for sbatch `/tmp` venvs
      (used by `_job.sh` via `uv pip install -e "$EXPERIMENT_DIR"`).
- [x] 1.2 `src` symlink → `../../src` resolves; `from plantclef.model import load_model`
      verified inside root `.venv`.
- [x] 1.3 GPU access on PACE confirmed (user 2026-05-04). Allocation pattern
      lives in each `sbatch/*.sbatch` (`gpu-rtx6000` partition, qinferno).

## 2. Test-side CLS extraction (§3.1)
- [x] 2.1 Added `--save-cls` flag to `experiments/001-dinov2-grid-ablation/inference.py`
      (commit on `mg/tileq-decoder`). Uses `forward_features` +
      `forward_head(..., pre_logits=True)`; writes
      `cls_<grid_label>.parquet` next to the existing probs parquet.
- [x] 2.2 Smoke test (5 images, job 7965650): GPU active (V100, cu121),
      shape=(16, 768), L2_norm ≈ 24-27. Identical CPU/GPU output (deterministic).
- [x] 2.3 Full run (job 7966845, V100, 16 min): output at
      `shared/inference/dinov2-vit-base/cls_grid_4x4.parquet`, 49.6 MB,
      33,680 rows × float16[768] CLS embeddings.
- [x] 2.4 Validation passed: 33,680 rows, 2,105 unique images, 16 tiles each
      (tile_idx 0..15 each at 2,105). Joinable to grid_4x4.parquet on
      (image_name, tile_idx). L2 norm distribution: mean 25.4, std 2.8,
      range [13.3, 30.0] — higher than train mean ~18.6 (vegetation scenes
      vs single plants); decoder `proj_kv` adapts away. Note: in-job tail
      crashed on pyarrow API bug (`StringArray.length()` → fixed to
      `len(...)` in commit); validation re-run manually.

## 3. Train-side data preparation
- [x] 3.1 Build per-class-idx lookup (§3.2): `build_train_lookup.py` reads 1.4 M
      train CLS via `pyarrow.dataset`, remaps species_id → class_idx via
      `class_mapping.txt`, sorts rows by class_idx and persists as a flat
      `(N, 768)` array + `(n_classes+1,)` offsets in
      `artifacts/train_cls_by_class_idx.npz`. **DONE 2026-05-05** — job 8029519,
      2:14 wallclock, 4.1 GB output. All 7806 classes have ≥ 1 row
      (min=1 > spec's p10=6, confirms long tail extends below p10).
- [x] 3.2 Species sampling weights `w_s ∝ 1/√count[s]` (§3.3) persisted to
      `artifacts/species_weights.npy`. Verified sum=1.000000, all positive.
- [ ] 3.3 *(Optional)* SW-Europe geo mask (§2.6). **Skipped for v1** per spec
      recommendation; deferred to future work.

## 4. FAISS background-cleaning map (§3.4, Sampler B prerequisite)
- [x] 4.1-4.4 `build_faiss_background_map.py`: L2-normalises all 1.4 M train
      CLS in place, builds `faiss.IndexFlatIP`, L2-normalises 33,680 test
      tiles, queries top-1. **DONE 2026-05-05** — job 8035733, 11:15 wallclock
      (586 s on the FAISS query alone, 47 G IP dot products).
- [x] 4.5 Top-1 sim histogram saved to `artifacts/tile_top1_sim_histogram.png`.
      **No shoulder at the spec's 0.85 default**: distribution is unimodal at
      0.51, max 0.89. Percentile + coverage table in `artifacts/faiss_threshold.txt`.
- [x] 4.6 **τ = 0.65 chosen** (override of spec default 0.85, documented in
      `results.md` § "FAISS τ tuning"). 1,838 / 33,680 tiles labelled (5.46 %).
      `tile_to_class_idx` persisted in `artifacts/sampler_b_lookup.npz`
      (folded together with `image_names`, `test_cls`, `top1_sim`,
      `top1_class_idx`, `threshold` — single npz so the sampler can't get
      ordering wrong). `rethreshold.py` re-applies τ in seconds without
      re-running FAISS, using the cached `top1_sim` + `top1_class_idx`.
- [x] 4.7 Sanity check passed: top-10 highest-sim tiles all match plausible
      species — Eryngium maritimum × 4 tiles in one GUARDEN-CBNMed-29 quadrat
      (sim 0.87-0.89), Quercus coccifera × 5 across consecutive CBNMed-{8,9}
      sites (sim 0.86-0.87).

## 5. Library code
- [x] 5.1 `user/murilo/src/plantclef/tileq_decoder.py` — `TileQDecoder`
      (4,509,824 parameters; matches spec §2.1).
- [x] 5.2 `user/murilo/src/plantclef/asl_loss.py` — `AsymmetricLoss(γ⁻=4,
      γ⁺=0, m=0.05)` (matches spec §2.2; reduces to BCE when γ=m=0,
      verified by test).
- [x] 5.3 `user/murilo/src/plantclef/data/samplers.py` — `SamplerA` (naive)
      and `SamplerB` (test-anchored, FAISS-relabelled). Both class_idx-keyed
      throughout (we standardised on class_idx internally; species_id only
      at I/O).
- [x] 5.4 `experiments/003-tileq-decoder/dataset.py` — `PseudoQuadratDataset`
      `IterableDataset` with multi-worker-safe per-worker RNG reseeding.

## 6. Sanity checks (§4.3)
- [x] 6.1 Sampler A unit tests (12): shapes, label cardinality = n_positives,
      positives distinct, tiles drawn from correct per-class pool, tile order
      shuffled, weights respected over many draws.
- [x] 6.2 Sampler B unit tests (12): shapes, label includes injections + any
      surviving FAISS-discovered backgrounds, **overwritten background labels
      strictly dropped** (controlled deterministic test), label dedupes when
      injection coincides with bg label, untouched slots match `test_cls[q,t]`
      exactly.
- [x] 6.3 Shape audit (`sanity_check.py`): all tensor shapes printed through
      one forward + backward; 17/17 parameters receive gradient. Output
      shape `(B, 7806)` confirmed.
- [x] 6.4 Overfit-one-batch (200 steps, batch 32) on real artefacts:
      Sampler A 306.66 → 0.0157 (**19,561× reduction**), Sampler B 305.10 →
      0.0233 (**13,075× reduction**). Both pass the 5× gate; near-zero
      reached by step 80. `sanity_check.py` is wired as a gate at the top of
      `train.sbatch` — training does not start unless these pass.
- [x] 6.5 Full pytest suite: **80/80 passing** in 12.7 s.

## 7. Train Run 1 — Sampler A (§4.4)
- [x] 7.1 `train.py --sampler A --seed 42 --steps 80000`. Saves checkpoints
      every 10k steps + final. AdamW + warmup→cosine + ASL + grad-clip 1.0
      + bf16 auto-fallback to fp32. W&B + TensorBoard logging matching
      birdclef pattern (entity dsgt-arc, project plantclef-2026-exp003).
- [x] 7.2 Submitted `sbatch/train.sh --sampler A` (job **8038028**).
      Wallclock **12:05** (9.4 min training + ~3 min venv install) on
      RTX 6000. **Step rate 142.5 step/s** — much faster than the spec's
      ~6 h estimate (4.5 M decoder is GPU-cheap per step).
- [x] 7.3 Per-step train loss + sparse val_loss / acc@10 logged to
      `artifacts/loss_samplerA.csv` (2.3 MB, 80,000 rows). Loss curve
      `artifacts/loss_samplerA.png`. W&B run: `u5ma36yz`.
- [x] 7.4 Final checkpoint at `artifacts/checkpoints/decoder_samplerA.pt`
      (52 MB). Final train/val loss = 4.91 / 4.85, val acc@10 = 0.9965.

## 8. Train Run 2 — Sampler B (§4.4)
- [x] 8.1-8.3 `train.py --sampler B --seed 42 --steps 80000` (same hyperparams
      as Sampler A run). Job **8038496**, wallclock **11:46** on RTX 6000,
      step rate 146.7 step/s. Final checkpoint at
      `artifacts/checkpoints/decoder_samplerB.pt` (52 MB). Final train/val
      loss = 0.93 / 0.92, val acc@10 = 1.0000. W&B run: `hdcx1p23`.

      **Note**: Sampler B's much lower synthetic val loss (0.92 vs A's 4.85,
      ~5.3× lower) does **not** predict higher Kaggle F1. The two samplers
      validate on different synthetic distributions; B's distribution is
      easier to fit because untouched test tiles don't strongly activate any
      class. The H2 question is decidable only by Kaggle. Documented in
      `results.md` § "What this offline gap does (and does NOT) tell us".

## 9. Inference + headline submissions
- [ ] 9.1 `infer.py --checkpoint artifacts/checkpoints/decoder_samplerA.pt
      --tile-cls shared/inference/dinov2-vit-base/cls_grid_4x4.parquet
      --topk 9,10 --output-prefix submissions/A1_samplerA_seed42`
      → produces both `A1_samplerA_topk9_seed42.csv` (H1 anchor) and
      `A1_samplerA_topk10_seed42.csv` from a single forward pass.
- [ ] 9.2 Same for Sampler B →
      `submissions/A2_samplerB_topk{9,10}_seed42.csv`.
- [ ] 9.3 *(If §3.3 mask was built)* re-run with `--geo-mask artifacts/sw_europe_mask.npy`
      and add `_geo` suffix to filenames.
- [ ] 9.4 Submit all four CSVs (A1@k9, A1@k10, A2@k9, A2@k10) to Kaggle.
      Record messages + LB scores in `submissions/SUBMISSIONS.md`.

## 10. A3 score-blend (bonus, §2.5)
- [ ] 10.1 Pick best of {A1, A2} as decoder source. Load decoder probs (per-image)
      and cached `max_prob` probs (per-image, from `grid_4x4.parquet`).
- [ ] 10.2 `blend.py --alpha {0,0.25,0.5,0.75,1.0}` → 5 CSVs in `submissions/`.
- [ ] 10.3 Submit each. Record α vs F1 curve.

## 11. Analysis
- [ ] 11.1 Compile ablation table in `results.md` (rows A0a, A0b, A1, A2, A3-α*).
- [ ] 11.2 Document any failure modes hit (§7) and how they were resolved.
- [ ] 11.3 Decide H1 / H2 outcomes per `proposal.md` success criteria.
- [ ] 11.4 Note wall-clock per stage for paper appendix.
- [ ] 11.5 Note any future-work directions confirmed by results (§8).
