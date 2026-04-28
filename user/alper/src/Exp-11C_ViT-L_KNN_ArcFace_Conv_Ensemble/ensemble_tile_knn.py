"""Exp-11: Tile-level ensemble + kNN.

1. Load ViT and ConvNeXt tile parquets (same scales, same tiles)
2. Blend logits per tile: blended = w * vit_logits + (1-w) * cnxt_logits
3. Create merged parquet with blended logits + ViT embeddings (for kNN)
4. Run Exp-9 kNN pipeline on the merged tiles
5. Temporal fusion → submission

Usage:
  python ensemble_tile_knn.py \
    --vit-scales vit_4x4.parquet vit_5x5.parquet vit_6x6.parquet \
    --cnxt-scales cnxt_4x4.parquet cnxt_5x5.parquet cnxt_6x6.parquet \
    --faiss-index /path/to/faiss_index_geo.bin \
    --faiss-labels /path/to/faiss_labels_geo.npy \
    --species-map /path/to/map.json \
    --out-dir /path/to/output
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP9_DIR = os.path.join(SCRIPT_DIR, "..", "Exp-9_kNN_Retrieval_Ensemble")
EXP8_DIR = os.path.join(SCRIPT_DIR, "..", "Exp-8_Quantile_Aggregation_SS")
sys.path.insert(0, os.path.abspath(EXP9_DIR))
sys.path.insert(0, os.path.abspath(EXP8_DIR))

from aggregate_tiles import AggregationConfig, save_aggregated
from generate_submission import load_species_map, load_class_mask_npy
from temporal_fusion import (
    load_quadrat_probs, group_by_location,
    build_location_profiles, fuse_all,
)
from utils.io_bytes import f16_bytes_list_to_f32_matrix, f16_vector_to_bytes
from knn_retriever import KNNRetriever

NUM_CLASSES = 7806
VIT_EMB_DIM = 1024  # ViT-L embedding dim (was 768 for ViT-B in Exp-11)


def load_and_merge_tiles(vit_paths, cnxt_paths, blend_w):
    """Load ViT and ConvNeXt tiles, blend logits, keep ViT embeddings."""
    # Load all ViT tiles
    vit_frames = []
    for p in vit_paths:
        df = pd.read_parquet(p)
        print(f"  ViT: {os.path.basename(p)}: {len(df)} tiles")
        vit_frames.append(df)
    vit_df = pd.concat(vit_frames, ignore_index=True)

    # Load all ConvNeXt tiles
    cnxt_frames = []
    for p in cnxt_paths:
        df = pd.read_parquet(p)
        print(f"  CNxt: {os.path.basename(p)}: {len(df)} tiles")
        cnxt_frames.append(df)
    cnxt_df = pd.concat(cnxt_frames, ignore_index=True)

    print(f"  ViT total: {len(vit_df)} tiles, {vit_df['image_name'].nunique()} images")
    print(f"  CNxt total: {len(cnxt_df)} tiles, {cnxt_df['image_name'].nunique()} images")

    # Merge on (image_name, tile_index)
    merged = vit_df.merge(
        cnxt_df[["image_name", "tile_index", "teacher_logits_f16"]],
        on=["image_name", "tile_index"],
        suffixes=("_vit", "_cnxt"),
        how="inner",
    )
    print(f"  Merged: {len(merged)} tiles, {merged['image_name'].nunique()} images")

    # Blend logits (vectorized)
    print(f"  Blending logits (w_vit={blend_w:.2f})...")
    n = len(merged)
    vit_logits_all = np.stack([
        np.frombuffer(b, dtype=np.float16).astype(np.float32)
        for b in merged["teacher_logits_f16_vit"].values
    ])  # [N, 7806]
    cnxt_logits_all = np.stack([
        np.frombuffer(b, dtype=np.float16).astype(np.float32)
        for b in merged["teacher_logits_f16_cnxt"].values
    ])  # [N, 7806]
    blended_all = (blend_w * vit_logits_all + (1 - blend_w) * cnxt_logits_all).astype(np.float16)
    blended_logits = [blended_all[i].tobytes() for i in range(n)]
    del vit_logits_all, cnxt_logits_all, blended_all

    # Build result with blended logits + ViT embeddings
    result = pd.DataFrame({
        "image_name": merged["image_name"].values,
        "tile_index": merged["tile_index"].values,
        "embedding_orig_f16": merged["embedding_orig_f16"].values,
        "teacher_logits_f16": blended_logits,
        "max_prob": merged["max_prob"].values,
        "passed_aug_consistency": [True] * len(merged),
    })

    return result


def aggregate_with_knn(
    tile_data: pd.DataFrame,
    knn_retriever: KNNRetriever,
    config: AggregationConfig,
    beta: float,
    geo_mask: Optional[np.ndarray],
    knn_k: int,
    knn_tau: float,
    single_visit_images: set,
    beta_single: float,
) -> Dict[str, np.ndarray]:
    """Aggregate tiles with kNN blend per image."""
    results = {}
    for name, group in tile_data.groupby("image_name"):
        is_single = str(name) in single_visit_images
        b = beta_single if is_single else beta

        pdf = group
        if config.aug_filter and "passed_aug_consistency" in pdf.columns:
            pdf = pdf[pdf["passed_aug_consistency"] == True]
        if pdf.empty:
            continue

        # Softmax from blended logits
        logits_bufs = pdf["teacher_logits_f16"].tolist()
        tile_logits = f16_bytes_list_to_f32_matrix(logits_bufs, NUM_CLASSES)
        x = tile_logits.astype(np.float32) / config.temperature
        x -= np.max(x, axis=1, keepdims=True)
        ex = np.exp(x, dtype=np.float32)
        p_softmax = ex / np.sum(ex, axis=1, keepdims=True)

        # kNN from ViT embeddings
        if b < 1.0:
            emb_bufs = pdf["embedding_orig_f16"].tolist()
            tile_embs = f16_bytes_list_to_f32_matrix(emb_bufs, VIT_EMB_DIM)
            p_knn = knn_retriever.query_batch(tile_embs, k=knn_k, tau=knn_tau)
            p_blended = b * p_softmax + (1.0 - b) * p_knn
        else:
            p_blended = p_softmax

        # max_prob aggregation
        p_image = np.max(p_blended, axis=0).astype(np.float32)
        np.clip(p_image, 0.0, 1.0, out=p_image)
        if config.denoise_eps > 0:
            p_image[p_image < config.denoise_eps] = 0.0
        if config.image_topk and config.image_topk > 0:
            k = min(config.image_topk, len(p_image))
            if k < len(p_image):
                idx = np.argpartition(p_image, -k)[-k:]
                mask = np.zeros(len(p_image), dtype=np.bool_)
                mask[idx] = True
                p_image[~mask] = 0.0
        if geo_mask is not None:
            p_image[~geo_mask] = 0.0

        results[str(name)] = p_image.astype(np.float16)

    return results


def temporal_fusion_step(probs, alpha, out_dir, tag):
    tmp = os.path.join(out_dir, f"_tmp_{tag}.parquet")
    save_aggregated(probs, tmp)
    qp = load_quadrat_probs(tmp)
    lg = group_by_location(list(qp.keys()))
    lp = build_location_profiles(qp, lg, NUM_CLASSES)
    fused = fuse_all(qp, lg, lp, strategy="prior_blend", params={"alpha": alpha})
    os.remove(tmp)
    return {k: v.astype(np.float16) for k, v in fused.items()}


def generate_csv(probs, species_map, class_mask, topk, threshold, out_csv):
    rows = []
    for name, p in sorted(probs.items()):
        p = p.astype(np.float32)
        if class_mask is not None:
            p[~class_mask] = 0.0
        top_idx = np.argsort(p)[::-1][:topk]
        selected = [i for i in top_idx if p[i] >= threshold]
        if not selected:
            selected = [top_idx[0]]
        rows.append({
            "quadrat_id": os.path.splitext(name)[0],
            "species_ids": str([species_map[i] for i in selected])
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"    {os.path.basename(out_csv)}")


def main():
    ap = argparse.ArgumentParser(
        description="Exp-11: Tile-level ensemble + kNN")
    ap.add_argument("--vit-scales", nargs="+", required=True)
    ap.add_argument("--cnxt-scales", nargs="+", required=True)
    ap.add_argument("--faiss-index", required=True)
    ap.add_argument("--faiss-labels", required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=0.15,
                    help="Temporal fusion alpha (Exp-9 best: 0.15)")
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--knn-tau", type=float, default=0.07)
    ap.add_argument("--knn-beta-multis", default="0.80",
                    help="Comma-separated kNN beta values for multi-visit")
    ap.add_argument("--knn-beta-single", type=float, default=1.00)
    ap.add_argument("--blend-weights", default="0.78,0.80,0.82,0.85",
                    help="ViT logit blend weights to sweep")
    ap.add_argument("--thresholds", default="0.100")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    blend_weights = [float(w) for w in args.blend_weights.split(",")]
    beta_multis = [float(b) for b in args.knn_beta_multis.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    class_mask = load_class_mask_npy(args.class_mask_npy, NUM_CLASSES) if args.class_mask_npy else None
    geo_mask = np.load(args.class_mask_npy).astype(np.bool_) if args.class_mask_npy else None

    # Load kNN retriever
    print("Loading FAISS index...")
    knn_retriever = KNNRetriever(
        index_path=args.faiss_index,
        labels_path=args.faiss_labels,
        num_classes=NUM_CLASSES,
        k=args.knn_k,
        tau=args.knn_tau,
    )
    print(f"  Index loaded: {knn_retriever.index.ntotal} vectors")

    config = AggregationConfig(
        temperature=args.temperature,
        aug_filter=False,
        tile_filter_k=0,
        image_topk=64,
        num_classes=NUM_CLASSES,
    )

    for blend_w in blend_weights:
        # Merge and blend tiles (once per blend weight)
        print(f"\n{'=' * 60}")
        print(f"BLEND WEIGHT: {blend_w:.2f} (ViT)")
        print(f"{'=' * 60}")

        merged_tiles = load_and_merge_tiles(
            args.vit_scales, args.cnxt_scales, blend_w)

        # Identify single-visit images
        single_visit = set()
        qids = [os.path.splitext(n)[0] for n in merged_tiles["image_name"].unique()]
        lg = group_by_location(qids)
        qid_to_name = {os.path.splitext(n)[0]: n for n in merged_tiles["image_name"].unique()}
        for loc, loc_qids in lg.items():
            if len(loc_qids) < 2:
                for qid in loc_qids:
                    if qid in qid_to_name:
                        single_visit.add(qid_to_name[qid])
        print(f"  Single-visit images: {len(single_visit)}")

        for beta_m in beta_multis:
            print(f"\n  --- beta_multi={beta_m:.2f} ---")
            t0 = time.time()
            image_probs = aggregate_with_knn(
                merged_tiles, knn_retriever, config,
                beta=beta_m,
                geo_mask=geo_mask,
                knn_k=args.knn_k,
                knn_tau=args.knn_tau,
                single_visit_images=single_visit,
                beta_single=args.knn_beta_single,
            )
            print(f"  Aggregated {len(image_probs)} images in {time.time() - t0:.1f}s")

            # Temporal fusion
            print(f"  Temporal fusion (alpha={args.alpha})...")
            fused = temporal_fusion_step(
                image_probs, args.alpha, args.out_dir,
                f"w{blend_w:.2f}_b{beta_m:.2f}")
            print(f"  Fused {len(fused)} images")

            # Generate CSVs
            for thr in thresholds:
                csv = os.path.join(
                    args.out_dir,
                    f"ens_tile_knn_w{blend_w:.2f}_b{beta_m:.2f}_thr{thr:.3f}_top{args.topk}.csv"
                )
                generate_csv(fused, species_map, class_mask, args.topk, thr, csv)

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
