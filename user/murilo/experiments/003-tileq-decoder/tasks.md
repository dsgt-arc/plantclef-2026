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
- [ ] 1.3 Confirm GPU access on PACE (RTX 6000 or H100); record allocation
      command in `sbatch/`.

## 2. Test-side CLS extraction (§3.1)
- [ ] 2.1 Add `--save-cls` flag to `experiments/001-dinov2-grid-ablation/inference.py`.
      Use `forward_features` + `forward_head(..., pre_logits=True)` to extract CLS.
- [ ] 2.2 Smoke test on 5 images (`--fast-dev-run`) — verify output shape (5, 16, 768) for grid 4×4.
- [ ] 2.3 Full run: grid 4×4 only, 2,105 images. Output:
      `shared/inference/dinov2-vit-base/cls_grid_4x4.parquet`. Schema:
      `image_name, tile_idx, cls_embedding (binary float16[768])`. Expected
      ~50 MB, ~2 GPU-h on RTX 6000.
- [ ] 2.4 Validate row count = 33,680 and per-image-name groups = 16.

## 3. Train-side data preparation
- [ ] 3.1 Build per-class-idx lookup (§3.2): load 1.4 M train CLS via
      `pyarrow.dataset`, remap species_id → class_idx via `class_mapping.txt`,
      group into `dict[class_idx] -> ndarray (n_i, 768)`. Persist to
      `artifacts/train_cls_by_class_idx.npz`.
- [ ] 3.2 Build species sampling weights `w_s ∝ 1/√count[s]` (§3.3). Persist
      to `artifacts/species_weights.npy`. Sanity-check sum=1, nonzero everywhere.
- [ ] 3.3 *(Optional)* Build SW-Europe geo mask (§2.6). Persist to
      `artifacts/sw_europe_mask.npy`. Decision: probably skip for v1.

## 4. FAISS background-cleaning map (§3.4, Sampler B prerequisite)
- [ ] 4.1 L2-normalise all 1.4 M train CLS embeddings.
- [ ] 4.2 Build `faiss.IndexFlatIP` over normalised train vectors.
- [ ] 4.3 L2-normalise all 33,680 test tile CLS embeddings.
- [ ] 4.4 Query top-1 per test tile; collect `(quadrat_idx, tile_idx, sim, train_species_id)`.
- [ ] 4.5 Plot top-1 sim histogram → `artifacts/tile_top1_sim_histogram.png`.
      Pick τ at the visible shoulder; record in `artifacts/faiss_threshold.txt`.
- [ ] 4.6 Persist `tile_to_species: dict[(quadrat_idx, tile_idx)] -> species_id | None`
      to `artifacts/tile_to_species.pkl`.
- [ ] 4.7 Sanity check: spot-compare 5 high-sim test tiles to their top-1 train
      neighbours (organ + species name visually plausible).

## 5. Library code
- [ ] 5.1 `user/murilo/src/plantclef/tileq_decoder.py` — `TileQDecoder` class
      from §2.1.
- [ ] 5.2 `user/murilo/src/plantclef/asl_loss.py` — `AsymmetricLoss` from §2.2.
- [ ] 5.3 `user/murilo/src/plantclef/data/samplers.py` — `sample_pseudo_quadrat_A`
      and `sample_pseudo_quadrat_B` from §2.3.
- [ ] 5.4 `experiments/003-tileq-decoder/dataset.py` — PyTorch
      `IterableDataset` wrapping the samplers. Yields `(tile_cls (16,768), label (7806,))`.

## 6. Sanity checks (§4.3)
- [ ] 6.1 Sampler unit test: A and B both produce correct shapes; injected
      species in correct slots; multi-hot label matches species set.
- [ ] 6.2 Sampler B unit test: synthetic test where a known-species CLS is
      retrieved → label set augmented as expected.
- [ ] 6.3 Shape audit: print every tensor shape through forward + backward at startup.
- [ ] 6.4 Overfit-one-batch: train on 32 fixed pseudo-quadrats for 1,000
      steps; loss → near zero. Block long training runs until this passes.

## 7. Train Run 1 — Sampler A (§4.4)
- [ ] 7.1 `train.py` with `--sampler A --seed 42 --steps 80000`. Save
      checkpoint every 10k steps + final.
- [ ] 7.2 Submit `sbatch/train_samplerA.sbatch` (1× RTX 6000, ~6 h).
- [ ] 7.3 Monitor val loss + any-positive-found acc every 2k steps. Save
      loss curves to `artifacts/loss_samplerA.csv` + `.png`.
- [ ] 7.4 Final checkpoint → `artifacts/checkpoints/decoder_samplerA.pt`.

## 8. Train Run 2 — Sampler B (§4.4)
- [ ] 8.1 `train.py` with `--sampler B --seed 42 --steps 80000`. Same hyperparams.
- [ ] 8.2 Submit `sbatch/train_samplerB.sbatch`. Run in parallel with Run 1
      if 2 GPUs available; otherwise sequentially.
- [ ] 8.3 Final checkpoint → `artifacts/checkpoints/decoder_samplerB.pt`.

## 9. Inference + headline submissions
- [ ] 9.1 `infer.py --checkpoint artifacts/checkpoints/decoder_samplerA.pt
      --tile-cls shared/inference/dinov2-vit-base/cls_grid_4x4.parquet
      --topk 10 --output submissions/A1_samplerA_seed42.csv`.
- [ ] 9.2 Same for Sampler B → `submissions/A2_samplerB_seed42.csv`.
- [ ] 9.3 *(If §3.3 mask was built)* re-run with `--geo-mask artifacts/sw_europe_mask.npy`
      and add `_geo` suffix to filenames.
- [ ] 9.4 Submit A1 + A2 to Kaggle. Record scores.

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
