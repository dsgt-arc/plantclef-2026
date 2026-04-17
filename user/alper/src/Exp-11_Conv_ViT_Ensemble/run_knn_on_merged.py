"""Run kNN + aggregation + temporal fusion on pre-merged tile parquet."""
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
from utils.io_bytes import f16_bytes_list_to_f32_matrix
from knn_retriever import KNNRetriever

NUM_CLASSES = 7806
VIT_EMB_DIM = 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-parquet", required=True,
                    help="Pre-merged tile parquet from merge_tiles.py")
    ap.add_argument("--faiss-index", required=True)
    ap.add_argument("--faiss-labels", required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--knn-tau", type=float, default=0.07)
    ap.add_argument("--knn-beta-multis", default="0.80",
                    help="Comma-separated beta values to sweep")
    ap.add_argument("--knn-beta-single", type=float, default=1.00)
    ap.add_argument("--threshold", type=float, default=0.100)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    class_mask = load_class_mask_npy(args.class_mask_npy, NUM_CLASSES) if args.class_mask_npy else None
    geo_mask = np.load(args.class_mask_npy).astype(np.bool_) if args.class_mask_npy else None

    # Load merged tiles
    print(f"Loading merged tiles from {args.merged_parquet}")
    tile_data = pd.read_parquet(args.merged_parquet)
    print(f"  {len(tile_data)} tiles, {tile_data['image_name'].nunique()} images")

    # Load FAISS
    print("Loading FAISS index...")
    t0 = time.time()
    knn = KNNRetriever(
        index_path=args.faiss_index,
        labels_path=args.faiss_labels,
        k=args.knn_k,
        tau=args.knn_tau,
        num_classes=NUM_CLASSES,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s ({knn.index.ntotal} vectors)")

    # Identify single-visit images
    qids = [os.path.splitext(n)[0] for n in tile_data["image_name"].unique()]
    lg = group_by_location(qids)
    qid_to_name = {os.path.splitext(n)[0]: n for n in tile_data["image_name"].unique()}
    single_visit = set()
    for loc, loc_qids in lg.items():
        if len(loc_qids) < 2:
            for qid in loc_qids:
                if qid in qid_to_name:
                    single_visit.add(qid_to_name[qid])
    print(f"  Single-visit: {len(single_visit)} images")

    beta_multis = [float(b) for b in args.knn_beta_multis.split(",")]
    beta_s = args.knn_beta_single

    config = AggregationConfig(
        temperature=args.temperature,
        aug_filter=False,
        tile_filter_k=0,
        image_topk=64,
        num_classes=NUM_CLASSES,
    )

    # Pre-compute per-image softmax and embeddings (reused across betas)
    print("Pre-computing softmax and loading embeddings...")
    t0 = time.time()
    image_groups = {}
    for name, group in tile_data.groupby("image_name"):
        logits_bufs = group["teacher_logits_f16"].tolist()
        tile_logits = f16_bytes_list_to_f32_matrix(logits_bufs, NUM_CLASSES)
        x = tile_logits.astype(np.float32) / config.temperature
        x -= np.max(x, axis=1, keepdims=True)
        ex = np.exp(x, dtype=np.float32)
        p_softmax = ex / np.sum(ex, axis=1, keepdims=True)

        emb_bufs = group["embedding_orig_f16"].tolist()
        tile_embs = f16_bytes_list_to_f32_matrix(emb_bufs, VIT_EMB_DIM)
        p_knn = knn.query_batch(tile_embs)

        is_single = str(name) in single_visit
        image_groups[str(name)] = {
            "p_softmax": p_softmax,
            "p_knn": p_knn,
            "is_single": is_single,
        }
    print(f"  Pre-computed {len(image_groups)} images in {time.time() - t0:.1f}s")

    # Sweep betas (fast — just blending pre-computed arrays)
    for beta_m in beta_multis:
        print(f"\n--- beta_multi={beta_m:.2f} ---")
        t0 = time.time()
        results = {}
        for name, data in image_groups.items():
            b = beta_s if data["is_single"] else beta_m

            if b < 1.0:
                p_blended = b * data["p_softmax"] + (1.0 - b) * data["p_knn"]
            else:
                p_blended = data["p_softmax"]

            p_image = np.max(p_blended, axis=0).astype(np.float32)
            np.clip(p_image, 0.0, 1.0, out=p_image)
            p_image[p_image < 1e-6] = 0.0
            topk_k = min(64, len(p_image))
            idx = np.argpartition(p_image, -topk_k)[-topk_k:]
            mask_arr = np.zeros(len(p_image), dtype=np.bool_)
            mask_arr[idx] = True
            p_image[~mask_arr] = 0.0
            if geo_mask is not None:
                p_image[~geo_mask] = 0.0
            results[name] = p_image.astype(np.float16)

        print(f"  Aggregated {len(results)} images in {time.time() - t0:.1f}s")

        # Temporal fusion
        tmp = os.path.join(args.out_dir, f"_tmp_b{beta_m:.2f}.parquet")
        save_aggregated(results, tmp)
        qp = load_quadrat_probs(tmp)
        lg2 = group_by_location(list(qp.keys()))
        lp = build_location_profiles(qp, lg2, NUM_CLASSES)
        fused = fuse_all(qp, lg2, lp, strategy="prior_blend", params={"alpha": args.alpha})
        os.remove(tmp)
        fused_f16 = {fk: fv.astype(np.float16) for fk, fv in fused.items()}

        # Generate CSV
        tag = args.tag or "w0.78"
        csv_path = os.path.join(
            args.out_dir,
            f"ens_knn_{tag}_k{args.knn_k}_tau{args.knn_tau}_b{beta_m:.2f}_thr{args.threshold:.3f}_top{args.topk}.csv"
        )
        rows = []
        for name, p in sorted(fused_f16.items()):
            p = p.astype(np.float32)
            if class_mask is not None:
                p[~class_mask] = 0.0
            top_idx = np.argsort(p)[::-1][:args.topk]
            selected = [i for i in top_idx if p[i] >= args.threshold]
            if not selected:
                selected = [top_idx[0]]
            rows.append({
                "quadrat_id": os.path.splitext(name)[0],
                "species_ids": str([species_map[i] for i in selected])
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  Wrote {os.path.basename(csv_path)}")


if __name__ == "__main__":
    main()
