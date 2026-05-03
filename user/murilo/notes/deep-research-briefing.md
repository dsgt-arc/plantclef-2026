# PlantCLEF 2026 — Deep-Research Briefing for a New Experiment

> Self-contained context for designing the next experiment. Pair this with
> `user/alper/src/Final_pipeline_methods_Summary/Final_Pipeline_Methods_Summary.md`
> when prompting Claude for deep research.

## 1. Task & metric

- **Task.** Multi-label species identification on vegetation-plot quadrat images.
  Each test image (≈3000×3000 px) is tagged with the set of plant species visible.
- **Test set.** 2,105 JPGs (`shared/data/raw/PlanCLEF_test_data/images/`),
  organised into ≈425 physical quadrats with 1–N visits per quadrat
  (~75.8% multi-visit). `PlantCLEF2025_test.csv` carries `quadrat_id, author,
  date, license`. The `image_name` itself encodes location + date.
- **Label space.** 7,806 species (PlantCLEF-2024 backbone). A geo-mask
  `aerten3/species_masking/sw_europe_mask_bool.npy` zeroes 3,014 species
  irrelevant to SW Europe, leaving 4,792 active classes.
- **Metric.** Kaggle macro-averaged F1 over per-image species sets. No public
  ground truth — every iteration costs a Kaggle submission.

## 2. Current best (Alper, F1 = 0.43790 public LB)

Probability-level **ConvNeXt-L DINOv3 + ViT-B DINOv2 ensemble**:

```
P_final = 0.77·P_vit+kNN+temporal(α=0.15) + 0.23·P_cnxt+temporal(α=0.30)
threshold = 0.100, topK = 10, geo mask
```

Pipeline order (per image):

1. JPEG round-trip q=85 4:2:2 after Lanczos resize.
2. Multiscale tiles: 4×4 + 5×5 + 6×6 = **77 non-overlapping 518×518 tiles**.
3. ViT-B DINOv2 forward → CLS (768-d) + logits (7806). Softmax T=1.5.
4. **Per-tile** kNN blend (multi-visit only): `p_tile = β·softmax + (1−β)·p_knn`
   with FAISS `IndexFlatIP` over ~860 K L2-normalised ViT CLS train embeddings,
   k=20, τ=0.07, β_multi=0.80, β_single=1.00 (single-visit images bypass kNN).
5. ConvNeXt-L DINOv3 forward on the same 77 tiles → softmax T=1.5.
6. `max_prob` aggregation across 77 tiles (per class), `image_topk=64` sparse
   storage. (Note: max_prob is the *only* aggregator whose F1 doesn't decay as
   tile count grows — this is what unlocks multiscale.)
7. Temporal `prior_blend` per location ℓ:
   `P_final = α·P_q + (1−α)·max_q' P_q'`, applied only to multi-visit quadrats.
   α_vit = 0.15, α_cnxt = 0.30 (per-model).
8. Probability-level ensemble (Approach A: kNN-on-ViT first, *then* blend with
   ConvNeXt — because the FAISS index lives in ViT-embedding space).
9. Threshold 0.100 + topK=10 + geo mask → submission CSV.

Headline gains, in order added: max_prob+T1.5 (+0.076), multiscale 4+5+6
(+0.044), ViT kNN with diff-β (+0.017), ConvNeXt ensemble + α retune (+0.020).

**Already-tested losers** (don't retry without a fresh angle): adaptive
per-image thresholds (Exp-6), border cropping 3/5/7% (Exp-12), inverse-frequency
kNN reweighting (Exp-9 Ph4), tile-level pre-kNN blending (Exp-11 Approach C),
TTA (hflip, rot, top-M avg) on per-tile preds (Exp-4), noisy-OR / mean / weighted
aggregators (Exp-4), KL/soft distillation across all 7806 classes (Exp-3 → mode
collapse), 1×1/2×2/3×3 scales added to multiscale (coarse-tile false confidence),
`quadrat_agg` post-processing on top of weak per-image lists (Murilo exp 002:
F1 0.171).

## 3. Available processed data (paths absolute under `/storage/project/ps-clef2026_plantclef-0/`)

### 3.1 Per-tile **test-set** softmax — Murilo's exp 001 output
Path: `shared/inference/dinov2-vit-base/{no_grid,grid_2x2…grid_10x10}.parquet`
(10 files, total ≈12 GB).

```
schema: image_name (str), tile_idx (int32), probs (binary = float16[7806])
reconstruction: np.frombuffer(probs, dtype=np.float16)
total tiles: 1·N + 4·N + 9·N + … + 100·N for N=2105 images
```

This is **ViT-B DINOv2** (`vit_base_patch14_reg4_dinov2.pth`, the Alper-shared
checkpoint), no kNN, no temporal fusion, no geo mask, no JPEG round-trip.
The ViT *probability* signal at every grid is already cached — any aggregation,
tile-selection or sparse-fusion experiment can iterate at CPU speed against
these parquets without re-running the 8-GPU-h forward pass.

Note: Alper's ViT branch uses the *same* checkpoint but with JPEG-q85 round-trip
preprocessing. Whether this preprocessing matters at multiscale-only (no kNN)
hasn't been isolated for *our* parquets — Alper reports +0.002 standalone,
+0.016 once temporal fusion is enabled.

### 3.2 Per-image **train-set** CLS embeddings (ViT-B DINOv2)
Path: `shared/embeddings/train/sample_id={0..19}/part-*.snappy.parquet`
(20 partitions × 21 part files ≈ 4.2 GB).

```
schema: image_name (str), species_id (int32), cls_embedding (list<float32>[768])
rows:   1,408,033   (full PlantCLEF-2024 single-plant training set)
```

This is the source for FAISS retrieval / prototype building / contrastive
re-ranking / class-prototype centroids. Alper's FAISS index uses a geo-filtered
~860 K subset (LUCAS-SWE region + train_full intersection). The full 1.4 M
table is here — you can build different sub-indices (per-family, per-organ,
rare-species-only, etc.) without recomputing embeddings.

### 3.3 Raw test images & training images
- `shared/data/raw/PlanCLEF_test_data/images/*.jpg` — 2,105 quadrat photos.
- `shared/data/raw/PlanCLEF_test_data/PlantCLEF2025_test.csv` —
  `quadrat_id; author; date; license` (the only test metadata available).
- `shared/data/raw/train_full/{species_id}/*.jpg` — full 1.4 M training set,
  one directory per species (7806 dirs).

### 3.4 Training-set metadata
`shared/data/raw/metadata_PlantCLEF_training_set/PlantCLEF2024singleplanttrainingdata.csv`
columns: `image_name, organ, species_id, obs_id, license, partner, author,
altitude, latitude, longitude, gbif_species_id, species, genus, family, dataset,
publisher, references, url, learn_tag, image_backup_url`.

Useful pivots: organ filter ({leaf, flower, fruit, bark, …} for prototype
purity), latitude/longitude (geo subsetting), family/genus (taxonomy-aware
losses or post-hoc re-ranking), obs_id (multi-image observations of the same
plant — natural pairs for contrastive / consistency objectives).

### 3.5 LUCAS auxiliary data
`shared/data/raw/metadata_lucas/`:
- `training_metadata_212k.csv` — LUCAS images with `point_id, year, lat, lon,
  altitude, land_cover` (vegetation-plot photos closer to the test domain than
  PlantCLEF crops).
- `LUCAS_metadata_SWE_Wide_lat35_50_lon-10_10_77k.csv` — 77 K SW-Europe LUCAS
  records with rich land-cover/land-use attributes.
- `lucas_cover_attr.csv` — joinable LUCAS attributes.
- Alper's geo mask was built off a SW-Europe filter analogous to this
  bounding box.

### 3.6 Class mappings
`shared/data/raw/class_id_to_species_mapping/`:
- `class_mapping.txt` — line `i` ↔ `species_id` (head row order, 7806 lines).
- `class_to_species_id.json` — JSON of the same, easier for python.
- `species_id_to_name.txt` — `species_id` ↔ Latin binomial.

### 3.7 Pretrained / finetuned weights
- `shared/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth`
  — the ViT-B DINOv2 finetune used by both Alper and exp 001.
- `shared/pretrained_models/reference_models/{…onlyclassifier, …onlyclassifier_then_all}/`
  — `args.yaml` + `summary.csv` (timm training history, no weights here).
- ConvNeXt-L DINOv3 phase-2 checkpoint (Alper):
  `aerten3/checkpoints/exp10_convnext/convnext_large_phase2_4gpu/model_best.pth.tar`.
- FAISS index (Alper):
  `aerten3/exp9_knn/faiss_index/{faiss_index_geo.bin, faiss_labels_geo.npy, class_counts_geo.npy}`.

### 3.8 Per-tile **test-set** outputs from Alper (multiscale)
- ViT 4+5+6 tile parquets:
  `aerten3/parquet/plantclef_2026_test/exp7_multiscale/scale_{4x4,5x5,6x6}.parquet`
  (these include CLS embedding *per tile*, not just probs — schema:
  `image_name, tile_index = scale*1000+local_idx, embedding_orig_f16[D],
  teacher_logits_f16[7806], max_prob, passed_aug_consistency`,
  D=768 for ViT-B).
- ConvNeXt 4+5+6 tile parquets:
  `aerten3/parquet/exp11_ensemble/convnext_large/scale_{4x4,5x5,6x6}.parquet`
  (D=1536).

These are critical: any experiment that wants to compose with Alper's pipeline
(re-aggregation, alternative kNN, prototype re-ranking, ensembling with a third
model) can read these directly — no re-extraction.

## 4. What Murilo has built locally

- `user/murilo/src/plantclef/{model.py, tiling.py, inference/dataset.py, gpu_monitor.py}`
  — minimal ViT-B inference library (load_model + load_class_mapping +
  TileDataset that loads each image once and returns all tiles as a stacked
  tensor; pin_memory + persistent_workers + prefetch_factor=4).
- `user/murilo/experiments/001-dinov2-grid-ablation/` — `inference.py` (Stage-1
  GPU → parquet, all 10 grid sizes complete) + `convert_parquet.py`. Validated
  no_grid baseline F1 = 0.14506.
- `user/murilo/experiments/002-classification-methods/` — `classify.py` self-
  contained CPU stage-2. Confirmed 2025 reference CSV was *sorted-prob top-k=9*
  on grid_4x4 → F1 = 0.30810 (99.9% exact match, 2103/2105 images identical).
  Quadrat-agg post-processing on these per-image lists hurts.

The exp-002 `classify.py` already implements `sorted-prob`, `union`,
`expanded-list`, `max-prob`, and an optional `--quadrat-agg` post-pass — useful
as a starting point for new aggregator/decoder experiments.

## 5. Constraints to design around

- **Submission budget** is the binding cost. Every method change should produce
  a delta that's distinguishable from Kaggle noise (~0.001–0.003 F1 between
  near-tied configs in Alper's sweeps). Prefer experiments with multiple
  defensible configs evaluable in one submission slot via diverse-seed
  ensembling, or with a strong offline proxy.
- **No labelled validation set.** Only proxies are: (a) overlap with Alper's
  best CSV per-image, (b) per-image species-count distribution shape vs.
  known good submissions, (c) training-set-style held-out probes (but the
  domain gap to vegetation-plot quadrats is large — be careful).
- **Compute.** PACE Phoenix; ViT-B inference at 77 tiles × 2105 images is
  ~2 h on 1 RTX 6000; ConvNeXt-L is ~8 h on H200. Embedding all 1.4 M training
  images already done. CPU stages (aggregation, kNN, ensembling, threshold
  tuning) iterate in 1–10 min. So new ideas that *re-use* cached parquets are
  drastically cheaper than ones that need a fresh forward pass.
- **Already-cached forward passes:**
  - ViT-B grid 0/2/3/.../10 *probs only* (Murilo, exp 001).
  - ViT-B multiscale 4+5+6 *embeddings + logits* (Alper, Exp-7).
  - ConvNeXt-L multiscale 4+5+6 *embeddings + logits* (Alper, Exp-11).
  - ViT-B *training* CLS embeddings, full 1.4 M (Murilo, `shared/embeddings/train`).

## 6. White-space / candidate angles for a new experiment

These are gaps consistent with what's been tried and rejected; treat as seeds
for a Claude deep-research prompt rather than recommendations.

1. **Quadrat-aware temporal fusion variants.** Alper's `prior_blend` uses
   element-wise max across visits. Alternatives untested at the ensemble F1
   level: rank-averaging, calibrated logistic pooling, time-decayed weighting
   (visits closer in date weighted higher), per-class learned α.
2. **Open-set / unknown-species head.** SW-Europe geo mask kills 3,014 classes
   to zero — but the model still has prediction mass there. A learned reject
   class on the test domain (semi-supervised from quadrat embeddings) might
   re-allocate mass to in-mask classes.
3. **Prototype / NCM classifier on the cached 1.4 M embeddings.** Compute a
   per-species centroid (or median, or LDA prototype) once, then blend with
   the softmax classifier — different inductive bias from kNN on individual
   training images (which Alper already does), and trivially cheap.
4. **Tile-level *consistency filtering* across grids.** Use Murilo's grid_NxN
   parquets to mark a tile prediction as confident only if it appears
   consistently across multiple grid scales centred on the same crop region.
   Pre-filter tiles before max_prob.
5. **Cross-model logit calibration via temperature *per class*.** Alper uses a
   single T=1.5 globally. Per-class T fitted on the cached training embeddings
   (e.g. via Platt scaling against species-id) might reduce the bias toward
   common species kNN was meant to fix.
6. **Use LUCAS land-cover priors.** `LUCAS_metadata_SWE_Wide` carries land-cover
   labels for nearby points; conditioning species priors on land-cover (dry
   grassland → Stipa…, shrubland → Cistus…) gives location-conditional priors
   not yet exploited.
7. **Organ-aware retrieval.** Train metadata has `organ ∈ {leaf, flower, fruit,
   bark, …}`. Build organ-specific FAISS sub-indices and route tiles by an
   organ classifier; current single-index retrieval mixes all organs.
8. **Better aggregator for ensemble-stage logits.** max_prob is best for
   single-model 77 tiles. For ensembles, the optimum may differ — e.g.
   geometric mean / log-domain pooling on calibrated probabilities.

## 7. Pointers to authoritative source files

- Pipeline summary (Alper, mandatory companion read):
  `user/alper/src/Final_pipeline_methods_Summary/Final_Pipeline_Methods_Summary.md`.
- Aggregator reference: `user/alper/src/Exp-4_Enhanced_Tile_Aggregation/utils/aggregator.py`.
- Temporal fusion: `user/alper/src/Exp-4_Enhanced_Tile_Aggregation/temporal_fusion.py`.
- Multiscale extractor: `user/alper/src/Exp-7_Multiscale_tiling/extract_multiscale_embeddings.py`.
- FAISS kNN: `user/alper/src/Exp-9_kNN_Retrieval_Ensemble/{build_faiss_index.py, knn_retriever.py, multiscale_aggregate_knn.py}`.
- ConvNeXt training/extraction: `user/alper/src/Exp-10_ConvNext_finetuning/{train_convnext.py, extract_convnext_embeddings.py, configs/}`.
- Ensemble fusion (winner): `user/alper/src/Exp-11_Conv_ViT_Ensemble/ensemble_with_knn.py`.
- Murilo's stage-1 inference: `user/murilo/experiments/001-dinov2-grid-ablation/inference.py`.
- Murilo's stage-2 classifier (sorted-prob, union, expanded-list, max-prob, quadrat_agg):
  `user/murilo/experiments/002-classification-methods/classify.py`.
- Final submission CSV (F1=0.4379):
  `aerten3/submissions/exp11_final_tuning3/w0.77_a0.30/ens_knn_w0.77_thr0.100_top10.csv`.
