"""Exp-9: kNN Retrieval Ensemble on multi-scale tiles.

Extends Exp-8 pipeline by blending softmax predictions with kNN retrieval
votes at the tile level before max_prob aggregation.

Pipeline:
  1. Load + combine tiles from scale directories (same as Exp-8)
  2. For each tile: blend softmax probs with kNN vote probs
  3. max_prob aggregation across blended tile probs
  4. Temporal fusion (prior_blend)
  5. Generate CSVs at multiple thresholds

Usage:
  python multiscale_aggregate_knn.py \
    --scale4-dir /path/to/phase1_tiles_test_jpeg85 \
    --scale5-dir /path/to/scale_5x5 \
    --scale6-dir /path/to/scale_6x6 \
    --faiss-index /path/to/faiss_index_geo.bin \
    --faiss-labels /path/to/faiss_labels_geo.npy \
    --species-map /path/to/species_map.json \
    --class-mask-npy /path/to/sw_europe_mask_bool.npy \
    --out-dir /path/to/output
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add Exp-8 to path for shared modules
EXP8_DIR = os.path.join(os.path.dirname(__file__), "..", "Exp-8_Quantile_Aggregation")
sys.path.insert(0, os.path.abspath(EXP8_DIR))

from aggregate_tiles import (
    AggregationConfig,
    load_per_tile_parquet,
    save_aggregated,
)
from generate_submission import (
    generate_submission_csv,
    load_class_mask_npy,
    load_species_map,
)
from temporal_fusion import (
    build_location_profiles,
    fuse_all,
    group_by_location,
    load_quadrat_probs,
)
from utils.io_bytes import f16_bytes_list_to_f32_matrix, f16_vector_to_bytes

from knn_retriever import KNNRetriever

NUM_CLASSES = 7806

# --------------------------------------------------------------------------
# Scale loading (reused from Exp-8)
# --------------------------------------------------------------------------

def load_scale_tiles(parquet_path: str, scale: int) -> pd.DataFrame:
    """Load per-tile data for a specific scale."""
    import glob as _glob

    if os.path.isfile(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.isdir(parquet_path):
        scale_file = os.path.join(parquet_path, f"scale_{scale}x{scale}.parquet")
        if os.path.exists(scale_file):
            df = pd.read_parquet(scale_file)
        else:
            shard_files = sorted(_glob.glob(
                os.path.join(parquet_path, "shard_*/*.parquet")
            ))
            if shard_files:
                df = load_per_tile_parquet(parquet_path, pattern="shard_*/*.parquet")
            else:
                raise FileNotFoundError(
                    f"No scale_{scale}x{scale}.parquet or shard_*/*.parquet "
                    f"found in {parquet_path}"
                )
    else:
        raise FileNotFoundError(f"Path does not exist: {parquet_path}")

    df["scale"] = scale
    if "tile_index" in df.columns:
        df["tile_index"] = scale * 1000 + df["tile_index"]
    return df


# --------------------------------------------------------------------------
# kNN-blended aggregation
# --------------------------------------------------------------------------

def aggregate_one_image_knn(
    tiles_df: pd.DataFrame,
    config: AggregationConfig,
    knn_retriever: KNNRetriever,
    beta: float,
    geo_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Aggregate tiles with kNN blend for one image.

    For each tile:
      p_tile = beta * softmax(logits/T) + (1-beta) * knn_vote(embedding)
    Then: p_image = max(p_tile across all tiles)

    Args:
        beta: weight for softmax (1.0 = no kNN, 0.0 = pure kNN)
    """
    pdf = tiles_df

    # Aug consistency filter
    if config.aug_filter and "passed_aug_consistency" in pdf.columns:
        pdf = pdf[pdf["passed_aug_consistency"] == True]  # noqa: E712
    if pdf.empty:
        return None

    # Tile confidence filter
    if "max_prob" in pdf.columns:
        pdf = pdf.sort_values("max_prob", ascending=False)
    if config.tile_filter_k > 0 and len(pdf) > config.tile_filter_k:
        pdf = pdf.head(config.tile_filter_k)

    # 1. Softmax probabilities from logits
    logits_bufs = pdf["teacher_logits_f16"].tolist()
    tile_logits = f16_bytes_list_to_f32_matrix(logits_bufs, config.num_classes)
    x = tile_logits.astype(np.float32, copy=False) / config.temperature
    x -= np.max(x, axis=1, keepdims=True)
    ex = np.exp(x, dtype=np.float32)
    p_softmax = ex / np.sum(ex, axis=1, keepdims=True, dtype=np.float32)  # [T, C]

    # 2. kNN probabilities from embeddings
    if beta < 1.0:
        emb_bufs = pdf["embedding_orig_f16"].tolist()
        tile_embs = f16_bytes_list_to_f32_matrix(emb_bufs, 768)  # [T, 768]
        p_knn = knn_retriever.query_batch(tile_embs)  # [T, C]

        # 3. Blend at tile level
        p_blended = beta * p_softmax + (1.0 - beta) * p_knn  # [T, C]
    else:
        # beta=1.0: pure softmax, skip kNN for speed
        p_blended = p_softmax

    # 4. max_prob aggregation
    p_image = np.max(p_blended, axis=0).astype(np.float32)

    # 5. Postprocess
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

    return p_image.astype(np.float16)


def aggregate_multiscale_knn(
    tile_data: pd.DataFrame,
    config: AggregationConfig,
    knn_retriever: KNNRetriever,
    beta: float,
    image_col: str = "image_name",
) -> Dict[str, np.ndarray]:
    """Aggregate all images with kNN-blended tile probabilities."""
    geo_mask = None
    if config.geo_mask_path:
        geo_mask = np.load(config.geo_mask_path)
        if geo_mask.dtype != np.bool_:
            geo_mask = geo_mask.astype(np.bool_)

    results = {}
    for name, group in tile_data.groupby(image_col):
        probs = aggregate_one_image_knn(
            group, config, knn_retriever, beta, geo_mask
        )
        if probs is not None:
            results[str(name)] = probs

    return results


# --------------------------------------------------------------------------
# Main sweep
# --------------------------------------------------------------------------

def run_knn_sweep(
    all_scale_dirs: Dict[int, str],
    species_map_path: str,
    class_mask_path: Optional[str],
    out_dir: str,
    knn_retriever: KNNRetriever,
    temperature: float = 1.5,
    alphas: Optional[List[float]] = None,
    betas: Optional[List[float]] = None,
    thresholds: Optional[List[float]] = None,
    topk: int = 10,
    image_topk: int = 64,
    tile_filter_k: int = 0,
    scales: Optional[List[int]] = None,
) -> None:
    """Run kNN ensemble sweep over beta, alpha, threshold."""
    if alphas is None:
        alphas = [0.05]
    if betas is None:
        betas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    if thresholds is None:
        thresholds = [0.100]
    if scales is None:
        scales = [4, 5, 6]

    os.makedirs(out_dir, exist_ok=True)

    species_map = load_species_map(species_map_path, NUM_CLASSES)
    class_mask = (
        load_class_mask_npy(class_mask_path, NUM_CLASSES)
        if class_mask_path
        else None
    )

    # Load tile data
    print("=" * 70)
    print("Loading tile data...")
    print("=" * 70)
    frames = []
    for scale in scales:
        if scale not in all_scale_dirs:
            raise ValueError(f"No directory provided for scale {scale}x{scale}")
        path = all_scale_dirs[scale]
        print(f"  Scale {scale}x{scale}: {path}")
        t0 = time.time()
        df = load_scale_tiles(path, scale)
        n_images = df["image_name"].nunique()
        print(f"    -> {len(df)} tiles, {n_images} images, {time.time() - t0:.1f}s")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    n_images = combined["image_name"].nunique()
    total_tiles = len(combined)
    print(f"\nCombined: {total_tiles} tiles, {n_images} images")

    config = AggregationConfig(
        temperature=temperature,
        aug_filter=True,
        tile_filter_k=tile_filter_k,
        geo_mask_path=class_mask_path,
        image_topk=image_topk,
        num_classes=NUM_CLASSES,
    )

    results_summary = []

    for beta in betas:
        print(f"\n{'=' * 70}")
        print(f"Beta={beta:.2f} ({'pure softmax' if beta == 1.0 else f'{beta:.0%} softmax + {1-beta:.0%} kNN'})")
        print(f"{'=' * 70}")

        # Aggregate with kNN blend
        t0 = time.time()
        image_probs = aggregate_multiscale_knn(
            combined, config, knn_retriever, beta
        )
        agg_time = time.time() - t0
        print(f"  Aggregated {len(image_probs)} images in {agg_time:.1f}s")

        # Save pre-fusion parquet
        tag = f"knn_beta{beta:.2f}_T{temperature:.2f}"
        prefusion_path = os.path.join(out_dir, f"{tag}_prefusion.parquet")
        save_aggregated(image_probs, prefusion_path)

        for alpha in alphas:
            # Temporal fusion
            t0 = time.time()
            quadrat_probs = load_quadrat_probs(prefusion_path)
            location_groups = group_by_location(list(quadrat_probs.keys()))
            location_profiles = build_location_profiles(
                quadrat_probs, location_groups, NUM_CLASSES
            )
            fused_probs = fuse_all(
                quadrat_probs,
                location_groups,
                location_profiles,
                strategy="prior_blend",
                params={"alpha": alpha},
            )
            fusion_time = time.time() - t0

            # Save fused parquet
            fused_f16 = {
                qid: p.astype(np.float16) for qid, p in fused_probs.items()
            }
            fused_path = os.path.join(out_dir, f"{tag}_fused_a{alpha}.parquet")
            save_aggregated(fused_f16, fused_path)

            # Generate CSVs at each threshold
            for thr in thresholds:
                csv_name = f"{tag}_a{alpha}_thr{thr:.3f}_top{topk}.csv"
                csv_path = os.path.join(out_dir, csv_name)

                n_rows = generate_submission_csv(
                    in_parquet=fused_path,
                    out_csv=csv_path,
                    species_map=species_map,
                    num_classes=NUM_CLASSES,
                    class_mask=class_mask,
                    topk=topk,
                    threshold=thr,
                    min_k=1,
                    id_col="image_name",
                )

                results_summary.append({
                    "beta": beta,
                    "alpha": alpha,
                    "threshold": thr,
                    "images": n_rows,
                    "csv": csv_name,
                })

    # Print summary
    print(f"\n{'=' * 100}")
    print("EXP-9 kNN RETRIEVAL ENSEMBLE SWEEP SUMMARY")
    print(f"{'=' * 100}")
    print(f"kNN: k={knn_retriever.k}, tau={knn_retriever.tau}")
    print(f"Temperature: {temperature}  TopK: {topk}  "
          f"tile_filter_k: {'ALL' if tile_filter_k <= 0 else tile_filter_k}")
    print()

    fmt = "{:>6} | {:>6} | {:>9} | {:>6} | {}"
    print(fmt.format("Beta", "Alpha", "Threshold", "Images", "CSV"))
    print(fmt.format("-" * 6, "-" * 6, "-" * 9, "-" * 6, "-" * 50))
    for r in results_summary:
        print(fmt.format(
            f"{r['beta']:.2f}", f"{r['alpha']:.2f}",
            f"{r['threshold']:.3f}", r["images"], r["csv"],
        ))
    print(f"{'=' * 100}")
    print(f"\nTotal CSVs: {len(results_summary)}")
    print(f"Output: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exp-9: kNN Retrieval Ensemble on multi-scale tiles"
    )

    # Scale directories
    ap.add_argument("--scale4-dir", required=True)
    ap.add_argument("--scale5-dir", default=None)
    ap.add_argument("--scale6-dir", default=None)

    # FAISS index
    ap.add_argument("--faiss-index", required=True,
                    help="Path to faiss_index_geo.bin")
    ap.add_argument("--faiss-labels", required=True,
                    help="Path to faiss_labels_geo.npy")

    # kNN params
    ap.add_argument("--knn-k", type=int, default=20)
    ap.add_argument("--knn-tau", type=float, default=0.07)

    # Data
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None)
    ap.add_argument("--out-dir", required=True)

    # Aggregation params
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--image-topk", type=int, default=64)
    ap.add_argument("--tile-filter-k", type=int, default=0)

    # Sweep params
    ap.add_argument("--betas", default="0.5,0.6,0.7,0.8,0.9,0.95,1.0",
                    help="Comma-separated beta values (softmax weight)")
    ap.add_argument("--alphas", default="0.05",
                    help="Comma-separated alpha values (temporal fusion)")
    ap.add_argument("--thresholds", default="0.100",
                    help="Comma-separated thresholds")
    ap.add_argument("--scales", default="4,5,6",
                    help="Comma-separated scale values to use")

    args = ap.parse_args()

    # Build scale directory mapping
    all_scale_dirs: Dict[int, str] = {4: args.scale4_dir}
    if args.scale5_dir:
        all_scale_dirs[5] = args.scale5_dir
    if args.scale6_dir:
        all_scale_dirs[6] = args.scale6_dir

    scales = [int(s) for s in args.scales.split(",")]
    betas = [float(b) for b in args.betas.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]

    print("Exp-9: kNN Retrieval Ensemble")
    print(f"Scales: {scales}")
    print(f"kNN k={args.knn_k}, tau={args.knn_tau}")
    print(f"Betas: {betas}")
    print(f"Alphas: {alphas}")
    print(f"Thresholds: {thresholds}")
    print()

    # Load kNN retriever
    print("Loading FAISS index...")
    t0 = time.time()
    knn_retriever = KNNRetriever(
        index_path=args.faiss_index,
        labels_path=args.faiss_labels,
        k=args.knn_k,
        tau=args.knn_tau,
        num_classes=NUM_CLASSES,
    )
    print(f"  Index loaded in {time.time() - t0:.1f}s, "
          f"ntotal={knn_retriever.index.ntotal}")

    run_knn_sweep(
        all_scale_dirs=all_scale_dirs,
        species_map_path=args.species_map,
        class_mask_path=args.class_mask_npy,
        out_dir=args.out_dir,
        knn_retriever=knn_retriever,
        temperature=args.temperature,
        alphas=alphas,
        betas=betas,
        thresholds=thresholds,
        topk=args.topk,
        image_topk=args.image_topk,
        tile_filter_k=args.tile_filter_k,
        scales=scales,
    )


if __name__ == "__main__":
    main()
