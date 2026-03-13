"""Phase 1 teacher aggregation: filter tiles, Noisy-OR aggregate, geo mask,
produce per-image predictions + packed tile embeddings for student training.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Iterator, List

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BinaryType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from utils.aggregator import TemperatureNoisyORAggregator
from utils.io_bytes import f16_bytes_list_to_f32_matrix, f16_vector_to_bytes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1 teacher aggregation")

    p.add_argument("--in-parquet-root", required=True, help="Phase 1 per-tile parquet root")
    p.add_argument("--metadata-csv", default=None, help="77k metadata CSV for lat/lon join")
    p.add_argument("--out-parquet", required=True, help="Output parquet path")

    p.add_argument("--image-col", default="image_name")
    p.add_argument("--logits-col", default="teacher_logits_f16")

    # Filtering
    p.add_argument("--aug-filter", action="store_true", default=True,
                    help="Filter tiles by aug consistency (default: True)")
    p.add_argument("--no-aug-filter", action="store_false", dest="aug_filter")
    p.add_argument("--keep-top-tiles", type=int, default=8,
                    help="Keep top-K tiles by max_prob after aug filter (default: 8)")

    # Aggregation
    p.add_argument("--num-classes", type=int, default=7806)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--denoise-eps", type=float, default=1e-6)
    p.add_argument("--image-topk", type=int, default=64,
                    help="Sparse top-K classes per image (default: 64)")
    p.add_argument("--final-topk", type=int, default=10,
                    help="Final top-K species for submission (informational)")

    # Geo mask
    p.add_argument("--geo-mask-npy", default=None, help="Path to boolean geo species mask .npy")

    # Metadata join
    p.add_argument("--metadata-image-col", default="image_name")
    p.add_argument("--keep-metadata-cols", default="point_id,year,latitude,longitude,altitude")
    p.add_argument("--join-type", default="left", choices=["left", "inner"])

    return p


def main() -> None:
    args = build_parser().parse_args()
    spark = SparkSession.builder.getOrCreate()

    # Load per-tile parquet
    df = spark.read.option("recursiveFileLookup", "true").parquet(args.in_parquet_root).where(
        F.col(args.logits_col).isNotNull()
    )

    # Load geo mask if provided
    geo_mask = None
    if args.geo_mask_npy:
        geo_mask = np.load(args.geo_mask_npy)
        if geo_mask.dtype != np.bool_:
            geo_mask = geo_mask.astype(np.bool_)
        print(f"Loaded geo mask: {geo_mask.sum()} / {geo_mask.shape[0]} species allowed")

    image_topk = int(args.image_topk) if int(args.image_topk) > 0 else None

    aggregator = TemperatureNoisyORAggregator(
        num_classes=int(args.num_classes),
        temperature=float(args.temperature),
        denoise_eps=float(args.denoise_eps),
        output_dtype=np.float16,
        image_topk=image_topk,
        return_stats=False,
    )

    # Output schema
    out_fields = [
        StructField(args.image_col, StringType(), nullable=False),
        StructField("teacher_image_probs_f16", BinaryType(), nullable=False),
        StructField("num_tiles_kept", IntegerType(), nullable=False),
        StructField("tile_embeddings_orig_f16", BinaryType(), nullable=True),
        StructField("tile_embeddings_flip_f16", BinaryType(), nullable=True),
        StructField("tile_embeddings_rot90_f16", BinaryType(), nullable=True),
        StructField("kept_tile_indices", BinaryType(), nullable=True),
    ]
    out_schema = StructType(out_fields)

    # Broadcast geo_mask
    geo_mask_bc = spark.sparkContext.broadcast(geo_mask) if geo_mask is not None else None
    aug_filter = bool(args.aug_filter)
    keep_top_tiles = int(args.keep_top_tiles)
    num_classes = int(args.num_classes)
    embed_dim = 768

    def agg_per_image(pdf_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for pdf in pdf_iter:
            if pdf.empty:
                continue

            image_name = str(pdf[args.image_col].iloc[0])

            # Step 1: Aug consistency filter
            if aug_filter and "passed_aug_consistency" in pdf.columns:
                pdf = pdf[pdf["passed_aug_consistency"] == True]  # noqa: E712

            if pdf.empty:
                continue

            # Step 2: Sort by max_prob, keep top-K
            if "max_prob" in pdf.columns:
                pdf = pdf.sort_values("max_prob", ascending=False)
            if keep_top_tiles > 0 and len(pdf) > keep_top_tiles:
                pdf = pdf.head(keep_top_tiles)

            # Step 3: Aggregate logits
            logits_bufs: List[bytes] = pdf[args.logits_col].tolist()
            tile_logits = f16_bytes_list_to_f32_matrix(logits_bufs, num_classes=num_classes)
            p_image_f16 = aggregator.aggregate(tile_logits)

            # Step 4: Apply geo mask
            gm = geo_mask_bc.value if geo_mask_bc is not None else None
            if gm is not None:
                p_image_f32 = p_image_f16.astype(np.float32)
                p_image_f32[~gm] = 0.0
                p_image_f16 = p_image_f32.astype(np.float16)

            # Step 5: Pack tile embeddings
            k = len(pdf)
            tile_indices = pdf["tile_index"].values.astype(np.int32) if "tile_index" in pdf.columns else np.arange(k, dtype=np.int32)

            def _pack_embeddings(col_name: str) -> bytes | None:
                if col_name not in pdf.columns:
                    return None
                bufs = pdf[col_name].tolist()
                packed = np.empty((k, embed_dim), dtype=np.float16)
                for j, b in enumerate(bufs):
                    packed[j] = np.frombuffer(b, dtype=np.float16)
                return packed.tobytes()

            yield pd.DataFrame(
                {
                    args.image_col: [image_name],
                    "teacher_image_probs_f16": [f16_vector_to_bytes(p_image_f16)],
                    "num_tiles_kept": [k],
                    "tile_embeddings_orig_f16": [_pack_embeddings("embedding_orig_f16")],
                    "tile_embeddings_flip_f16": [_pack_embeddings("embedding_flip_f16")],
                    "tile_embeddings_rot90_f16": [_pack_embeddings("embedding_rot90_f16")],
                    "kept_tile_indices": [tile_indices.tobytes()],
                }
            )

    agg_df = df.groupBy(args.image_col).applyInPandas(agg_per_image, schema=out_schema)

    # Metadata join
    if args.metadata_csv:
        meta = spark.read.csv(args.metadata_csv, header=True, inferSchema=True)
        keep_cols = [c.strip() for c in args.keep_metadata_cols.split(",") if c.strip()]
        select_cols = [F.col(args.metadata_image_col).alias(args.image_col)]
        for c in keep_cols:
            select_cols.append(F.col(c))
        meta_small = meta.select(*select_cols)
        agg_df = agg_df.join(meta_small, on=args.image_col, how=args.join_type)

    # Write output
    (
        agg_df.coalesce(1)
        .write.mode("overwrite")
        .option("compression", "snappy")
        .parquet(args.out_parquet)
    )

    print(f"Wrote aggregated teacher output to {args.out_parquet}")


if __name__ == "__main__":
    main()
