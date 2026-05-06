# PlantCLEF 2026 — Techniques That Worked (short summary)

Cumulative progression: **F1 0.28093 (baseline) → 0.50623 (current best, Exp-16C lock 2026-05-05).**

Below are the techniques that improved leaderboard score, in roughly the order they enter the pipeline. Negative experiments are omitted.

---

## Original Proposal (Exp-1, abandoned)

- **Cross-Region Transformer with noisy-student distillation on the 212K LUCAS unlabeled set (Exp-1, original proposal)** — The original research plan was an end-to-end *cross-region transformer*: each quadrat would be sliced into region windows, each region encoded independently by a frozen pretrained ViT to produce region embeddings, and a lightweight self-attention transformer would learn inter-tile spatial / co-occurrence structure across regions and pool to a single quadrat-level multi-label prediction. A teacher-student / noisy-student loop would supervise the student on the ~212K unlabeled "pseudo-quadrat" LUCAS images that PlantCLEF provides for optional adaptation, with the teacher being a frozen ViT + simple tile-aggregation head. The expected gain was +0.01 to +0.03 macro-F1 over standard tiling-and-aggregation baselines.
  - **Why it was abandoned.** The mechanism was sound, but the supervision signal was not: the 212K LUCAS unlabeled adaptation set comes from a substantially different sampling distribution and image-statistics regime than the actual 2105-image PlantCLEF test set (different sites, different camera setups, different ecological mix). Pseudo-labels distilled on LUCAS taught the student to model a distribution the leaderboard does not score against, so any in-domain transfer was washed out by the train↔test domain gap. Two related symptoms made the failure visible: KL/soft-distillation training over all 7806 classes caused mode collapse (the ~7742 near-zero targets per image overwhelmed the ~64 positive signals, F1 → 0); switching to BCE with negative sampling restored a working signal but only to F1≈0.143, well below what direct tiling on the test set was already achieving. The cross-region transformer idea was shelved and the project pivoted to the tiling-and-aggregation lineage that became Exp-2 onward.

---

## Preprocessing

- **JPEG round-trip preprocessing (Exp-2)** — After Lanczos-resizing each test image, re-encode it through PIL JPEG at quality=85, 4:2:2 chroma subsampling, then decode back to pixels before tiling. The DINO backbones were pretrained on JPEG-compressed natural images, so passing the test images through one round of compression matches the pixel-statistic distribution they expect. Cheap, no parameters to tune, downside is zero. Δ ≈ +0.002 standalone, compounds with downstream stages.

- **Multiscale non-overlapping tiling, scales 4+5+6 = 77 tiles (Exp-7)** — A single 4×4 grid on a 3000×3000 quadrat fragments individual plants and mixes species. Instead, tile each image at three grid sizes (4×4, 5×5, 6×6), giving 77 non-overlapping 518×518 tiles at three effective source resolutions (~747, 600, 500 px). The downstream `max_prob` aggregator can then pick whichever scale best isolates each species. Coarser scales (1×1, 2×2, 3×3) hurt because whole-image crops produce confident wrong labels; 7×7 is diminishing return. Δ ≈ +0.028 over 4×4 alone.

## Aggregation & Fusion

- **`max_prob` per-tile aggregation, T = 1.5 (Exp-4)** — For each species class, take the maximum softmax probability across the 77 tiles as the image-level prediction (with temperature T=1.5 sharpening per-tile peaks). Noisy-OR multiplies tile noise; mean/quantile pooling dilutes strong signals. Max preserves the single-best tile evidence per class independently, and is the only aggregator whose F1 is invariant to tile count — the precondition that lets multiscale work. Δ ≈ +0.039.

- **Temporal `prior_blend` across location visits (Exp-4)** — Each physical plot in the test set is photographed multiple times. Build a location prior `P_loc(c) = max over visits of P_visit(c)`, then blend with the current image's prediction: `P_final = (1−α)·P_loc + α·P_image`. Applied only to multi-visit quadrats (~76% of the test set). Lets a weak current photo borrow species evidence from earlier visits at the same plot. Δ ≈ +0.022.

- **Per-model α (Exp-11)** — Each branch of the ensemble gets its own α: ViT+kNN α=0.15 (already has external evidence from kNN, so leans on the location prior), ConvNeXt α=0.30 (no kNN arm, trusts its own per-image prediction more, which preserves architecture diversity for the ensemble). Δ ≈ +0.001.

- **Per-source `sim_w` aggregator on RNNB + α=0.30 (Exp-14, Stage 4)** — RNNB sites (saltmarsh / coastal dunes in SW France) show year-to-year species turnover: a lush May visit looks nothing like the bare-sand January visit at the same plot, but `max`-pooled prior was making them produce identical predictions. Replace with a per-image prior aggregator weighted by cosine similarity between the current image embedding and each historical sister visit: similar visits contribute fully, dissimilar ones collapse out. Restricted to RNNB only — applying it globally cratered alpine species counts (10.0 → 4.85). Alpine sources keep `max`. Δ +0.00982.

## Classifier Model Ensembling (ViT-L-DINOv2 + ConvNext-L-DINOv3)

- **ConvNeXt-Large DINOv3 finetune, 2-phase (Exp-10)** — `convnext_large.dinov3_lvd1689m` (196M params). Phase 1: freeze the backbone, train only the 7806-way classifier head with Adam, lr=0.01 for 100 epochs (cheap, stable head warmup). Phase 2: unfreeze everything and finetune end-to-end from the Phase 1 EMA weights with AdamW, lr=3e-5, layer-wise lr decay 0.8, drop_path=0.3 (higher reg for the larger model), RandAugment + mixup + cutmix, EMA decay 0.9998, 100 epochs on 4×H200. Provides the second ensemble member needed for architecture diversity.

- **ViT-L Phase 2v3 cooldown (Exp-13)** — `vit_large_patch14_reg4_dinov2.lvd142m` finetuned in two stages following the same head-only → full-finetune protocol, then a 10-epoch cooldown at lr=2e-5 on combined train+val+test data on top of Phase 2. Replaces the earlier ViT-B as the production ViT backbone. Standalone +0.011 F1.

- **ConvNeXt + ViT probability-level ensemble, w=0.77 (Exp-11, "Approach A")** — Run the ViT branch with kNN blending first to get the ViT-side per-image probabilities, then linearly mix in ConvNeXt: `P = 0.77·P_vit+knn + 0.23·P_cnxt`. Order matters: blending logits *before* kNN breaks the embedding↔logit alignment the FAISS index assumes (it was built on ViT embeddings only) and loses ~0.02. Both branches use the same 4+5+6 scale scheme — letting ConvNeXt use its own preferred 3+4+5+6+7 introduced disagreements ViT couldn't compensate for. Δ ≈ +0.020.

## Retrieval Model Ensembling (FAISS)

- **ViT-only FAISS kNN blend (Exp-9)** — Build an exact-search FAISS `IndexFlatIP` over ~860K geo-filtered ViT training-set CLS embeddings (768-d, L2-normalised so inner product = cosine). For each test tile embedding, retrieve the k=20 nearest training neighbours, softmax the similarities with τ=0.07 to get neighbour weights, and aggregate by class label to form a retrieval distribution `p_knn`. Blend per-tile *before* `max_prob` aggregation: `p_tile = β·softmax(logits/T) + (1−β)·p_knn`. The classifier softmax is biased toward common species (long tail under-trained); the retrieval arm is frequency-agnostic and supplies complementary "look-up" evidence for rare species clusters. Δ ≈ +0.017.

- **Differentiated β: β_multi=0.80, β_single=1.00 (Exp-9)** — Inject kNN only on multi-visit quadrats. Single-visit images don't have the temporal prior to absorb retrieval noise; using kNN there cost up to −0.08 F1. Multi-visit quadrats can tolerate β=0.80 (20% kNN weight) because the location prior smooths out noisy retrievals.

## Ecological Masking

- **Aquatic-species habitat-impossibility mask (Exp-14, Stage 12)** — Hand-curate 35 fully-aquatic genera (Posidonia, Zostera, Cymodocea, Lemna, Potamogeton, etc.) and 6 aquatic specific epithets (oceanica, submersa, fluitans, natans, …). Mark any of the 7806 species whose Latin binomial matches as habitat-impossible (91 species), and zero those probabilities post-fusion across all sources. Triggered by inspection: Posidonia oceanica (a marine seagrass) was the most-predicted species on dry-soil RNNB sand-dune quadrats. Pure precision improvement, zero parameters, ecologically defensible. Δ +0.00162.

- **Per-source biogeographic mask on alpine sources (Exp-14, Stage 13)** — On CBN-PdlC and CBN-Pla (alpine summit plots only), zero 5 hand-vetted species that survived Stage 12 (no aquatic match) but cannot occur at alpine elevation: tropical C4 carpetgrass (Axonopus compressus), coastal sea-thrift (Armeria maritima), Mediterranean coastal annuals, etc. Other sources are unchanged. Δ +0.00039 LB but proxy-positive on the full cohort; adopted as new best.

## Source-Aware Admission & Demotion

- **N-floor admission with confidence-tiered ratio gate, "N3_r0.5" (Exp-16)** — On count-bound sources where the model under-predicts (GUARDEN, LISAH, OPTMix, RNNB), if fewer than 3 species pass the standard threshold=0.100, admit additional candidates up to 3 — but only those whose probability ratio to the already-admitted top-1 is ≥ 0.5 (so we admit confident near-ties, not random low-confidence noise). Doesn't touch sources that already predict enough species (alpine). Largest single-mechanism gain since Stage 4. Δ +0.00557.

- **Habitat-fit demotion using external training-data geography (Exp-16)** — For each (test image, candidate species) pair, compute a habitat-fit factor in [β, 1]. Two components multiplied together: (1) **geo score** — count training observations of that species within 200 km of the source's refined regional centroid; if ≥ n_near_min (40) the species is unpenalised, if 0 it's demoted to β, linear in between; (2) **altitude score** — penalise species whose training-data median altitude lies outside the source's expected band. Multiply into the species probability *before* the N-floor admission. Uses external information (PlantCLEF2024 training-data lat/lon/alt) the model never saw at inference. Cosmopolitans (high n_near everywhere) untouched; range-restricted species observed only in distant regions get demoted to β. Δ +0.02361 standalone, final +0.03674 combined with admission retune.

- **Per-source β override on alpine sources (Exp-16C, NEW MECHANISM)** — The single global β was demoting legitimate alpine species too hard, because alpine quadrats were pinned at the topk-10 cap and any extra demotion just dropped real positives below threshold. Override β to 0.45 on CBN-PdlC and CBN-Pla (vs global β=0.15). Tested OPTMix per-source β too — no-op because OPTMix's "out-of-place" species (Plantago subulata-style) have high local n_near and aren't hit by the multiplicative knob. Δ +0.00065 LB, locks final F1=0.50623.

- **Habitat-fit retune at new ViT-L+ConvNeXt-P backbone (Exp-16C)** — When the production backbone changed, the habitat-fit hyperparameters needed re-tuning: n_near_min 30→40, β_global 0.10→0.15. R=200 km, k=10, thr=0.085 unchanged. Final lock 2026-05-05.

## Submission

- **Fixed global threshold = 0.085 + topK = 10 + SW Europe geo mask** — Hard-threshold per-class probabilities, then cap at 10 species per image. Adaptive per-image thresholds (Exp-6: max-gap, relative-ratio, elbow) all lost to a well-tuned fixed threshold. The geo mask zeroes ~3000 of the 7806 classes that don't occur in southwest Europe. Threshold value drifted across experiments (0.04 → 0.10 → 0.085) only because the probability scale shifts when the pipeline re-blends — the *rule* (fixed global threshold) is unchanged.
