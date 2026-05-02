"""Exp-14C: build pre-fusion parquets from ViT-L K test parquets + 1024-d FAISS.

Per beta_multi value:
  - For each image's tiles: classifier softmax (T=1.5) + LoRA-kNN votes
  - Blend: beta_multi * softmax + (1 - beta_multi) * kNN
    (single-visit images use beta_single, default 1.00 → no kNN)
  - Aggregate per image: image_topk=64 + max_prob across tiles
  - Apply geo mask if provided
  - Save as pre-fusion parquet (columns: image_name, image_probs_f16)

The output is consumed by Exp-14 ensemble_v2.py as `--vit-prefusion-parquet`.
Temporal fusion is intentionally NOT applied here — ensemble_v2.py's
ControllerConfig handles that as part of the per-branch fusion step.

Pre-computes classifier softmax + kNN per image once, then sweeps betas in
memory (each beta is just a blend + max_prob recomputation).

Usage:
  python make_vitL_prefusion.py \
    --vit-scales scale_4x4.parquet scale_5x5.parquet scale_6x6.parquet scale_7x7.parquet \
    --faiss-index .../faiss_index_arcface_geo.bin \
    --faiss-labels .../faiss_labels_arcface_geo.npy \
    --species-map .../class_to_species_id.json \
    --class-mask-npy .../sw_europe_mask_bool.npy \
    --out-dir .../exp14c_prefusion \
    --beta-multis 0.75,0.85,0.95 \
    --beta-single 1.00 \
    --temperature 1.5 \
    --knn-k 20 --knn-tau 0.07
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP9_DIR = os.path.join(SCRIPT_DIR, "..", "Exp-9_kNN_Retrieval_Ensemble")
EXP8_DIR = os.path.join(SCRIPT_DIR, "..", "Exp-8_Quantile_Aggregation_SS")
sys.path.insert(0, os.path.abspath(EXP9_DIR))
sys.path.insert(0, os.path.abspath(EXP8_DIR))

from aggregate_tiles import save_aggregated  # noqa: E402
from temporal_fusion import group_by_location  # noqa: E402
from utils.io_bytes import f16_bytes_list_to_f32_matrix  # noqa: E402
from knn_retriever import KNNRetriever  # noqa: E402

NUM_CLASSES = 7806
VIT_EMB_DIM = 1024  # ViT-L LoRA-ArcFace head output dim


def load_vit_scales(paths):
    """Load all per-tile ViT parquets and concat across scales."""
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        print(f"  {os.path.basename(p)}: {len(df)} tiles, "
              f"{df['image_name'].nunique()} images")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined: {len(combined)} tiles, "
          f"{combined['image_name'].nunique()} images")
    return combined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit-scales", nargs="+", required=True,
                    help="Per-tile ViT-L parquets (one per scale; merged with "
                         "LoRA emb_arcface_f16 column)")
    ap.add_argument("--faiss-index", required=True)
    ap.add_argument("--faiss-labels", required=True)
    ap.add_argument("--species-map", required=True,
                    help="Unused here, kept for symmetry with peer scripts.")
    ap.add_argument("--class-mask-npy", default=None,
                    help="Geo mask .npy (sw_europe_mask_bool.npy)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--knn-tau", type=float, default=0.07)
    ap.add_argument("--beta-multis", default="0.75,0.85,0.95",
                    help="Comma-separated beta_multi values to sweep")
    ap.add_argument("--beta-single", type=float, default=1.00)
    ap.add_argument("--emb-col", default="emb_arcface_f16",
                    help="Column with the LoRA-projected embedding bytes")
    ap.add_argument("--image-topk", type=int, default=64)
    ap.add_argument("--tag", default="vitL")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    geo_mask = (np.load(args.class_mask_npy).astype(np.bool_)
                if args.class_mask_npy else None)
    if geo_mask is not None:
        print(f"[init] geo mask: {geo_mask.sum()}/{len(geo_mask)} classes pass")

    # ------------------------------------------------------------------
    # Load tiles
    # ------------------------------------------------------------------
    print(f"\n=== Loading ViT-L tiles ===")
    tile_data = load_vit_scales(args.vit_scales)

    if args.emb_col not in tile_data.columns:
        raise ValueError(
            f"Embedding column '{args.emb_col}' missing from input parquets. "
            f"Available columns: {list(tile_data.columns)}"
        )

    # ------------------------------------------------------------------
    # Load FAISS
    # ------------------------------------------------------------------
    print(f"\n=== Loading FAISS index ===")
    t0 = time.time()
    knn = KNNRetriever(
        index_path=args.faiss_index,
        labels_path=args.faiss_labels,
        k=args.knn_k,
        tau=args.knn_tau,
        num_classes=NUM_CLASSES,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s ({knn.index.ntotal} vectors, "
          f"d={knn.index.d})")
    if knn.index.d != VIT_EMB_DIM:
        raise ValueError(
            f"FAISS index dim {knn.index.d} != expected {VIT_EMB_DIM}. "
            f"Wrong index for ViT-L?"
        )

    # ------------------------------------------------------------------
    # Identify single-visit images (same logic as run_knn_on_merged.py)
    # ------------------------------------------------------------------
    image_names = tile_data["image_name"].unique()
    qids = [os.path.splitext(n)[0] for n in image_names]
    lg = group_by_location(qids)
    qid_to_name = {os.path.splitext(n)[0]: n for n in image_names}
    single_visit = set()
    for loc, loc_qids in lg.items():
        if len(loc_qids) < 2:
            for qid in loc_qids:
                if qid in qid_to_name:
                    single_visit.add(qid_to_name[qid])
    print(f"\n  Single-visit images: {len(single_visit)} / {len(image_names)}")

    # ------------------------------------------------------------------
    # Per-image: classifier softmax + kNN votes (precompute once)
    # ------------------------------------------------------------------
    print(f"\n=== Pre-computing per-image classifier + kNN ===")
    t0 = time.time()
    image_groups: Dict[str, dict] = {}
    n_images = 0
    for name, group in tile_data.groupby("image_name"):
        # Classifier softmax (with temperature)
        logits_bufs = group["teacher_logits_f16"].tolist()
        tile_logits = f16_bytes_list_to_f32_matrix(logits_bufs, NUM_CLASSES)
        x = tile_logits.astype(np.float32) / args.temperature
        x -= np.max(x, axis=1, keepdims=True)
        ex = np.exp(x, dtype=np.float32)
        p_softmax = ex / np.sum(ex, axis=1, keepdims=True)

        # kNN votes via FAISS (using LoRA-projected embedding)
        emb_bufs = group[args.emb_col].tolist()
        tile_embs = f16_bytes_list_to_f32_matrix(emb_bufs, VIT_EMB_DIM)
        p_knn = knn.query_batch(tile_embs)

        is_single = str(name) in single_visit
        image_groups[str(name)] = {
            "p_softmax": p_softmax,
            "p_knn": p_knn,
            "is_single": is_single,
        }
        n_images += 1
        if n_images % 1000 == 0:
            elapsed = time.time() - t0
            rate = n_images / elapsed
            eta = (len(image_names) - n_images) / rate
            print(f"  {n_images}/{len(image_names)}  {rate:.1f} img/s  "
                  f"ETA {eta:.0f}s")
    print(f"  Pre-computed {len(image_groups)} images in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Sweep betas (in-memory blend + max_prob aggregate)
    # ------------------------------------------------------------------
    beta_multis = [float(b) for b in args.beta_multis.split(",")]
    print(f"\n=== Sweeping beta_multis: {beta_multis} (beta_single={args.beta_single}) ===")

    for beta_m in beta_multis:
        print(f"\n--- beta_multi={beta_m:.2f} ---")
        t0 = time.time()
        results: Dict[str, np.ndarray] = {}
        for name, data in image_groups.items():
            b = args.beta_single if data["is_single"] else beta_m

            if b < 1.0:
                p_blended = b * data["p_softmax"] + (1.0 - b) * data["p_knn"]
            else:
                p_blended = data["p_softmax"]

            # Per-image: max_prob aggregation across tiles
            p_image = np.max(p_blended, axis=0).astype(np.float32)
            np.clip(p_image, 0.0, 1.0, out=p_image)
            p_image[p_image < 1e-6] = 0.0

            # image_topk: keep top-k probs only, zero the rest
            tk = min(args.image_topk, len(p_image))
            idx = np.argpartition(p_image, -tk)[-tk:]
            mask_arr = np.zeros(len(p_image), dtype=np.bool_)
            mask_arr[idx] = True
            p_image[~mask_arr] = 0.0

            # Apply geo mask (zeros outside SW Europe)
            if geo_mask is not None:
                p_image[~geo_mask] = 0.0

            results[name] = p_image.astype(np.float16)

        # Save pre-fusion parquet (no temporal fusion — that's ensemble_v2.py's job)
        out_path = os.path.join(
            args.out_dir,
            f"{args.tag}_prefusion_b{beta_m:.2f}.parquet",
        )
        save_aggregated(results, out_path)
        print(f"  wrote {out_path}  ({len(results)} images, {time.time() - t0:.1f}s)")

    print(f"\n=== Done. {len(beta_multis)} pre-fusion parquets in {args.out_dir} ===")


if __name__ == "__main__":
    main()
