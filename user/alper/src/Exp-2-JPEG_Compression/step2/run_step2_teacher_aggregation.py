# src/step2/run_step2.py
from __future__ import annotations

import argparse
from typing import Iterator, List

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType, IntegerType, StringType, StructField, StructType

from step2.aggregator import TemperatureNoisyORAggregator
from step2.io_bytes import f16_bytes_list_to_f32_matrix, f16_vector_to_bytes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--in-parquet-root", required=True, help="Step 1 parent folder that contains shard_0000 etc")
    p.add_argument("--metadata-csv", required=True, help="Metadata CSV path")
    p.add_argument("--out-parquet", required=True, help="Step 2 output parquet path")

    p.add_argument("--image-col", default="image_name")
    p.add_argument("--logits-col", default="teacher_logits_f16")

    p.add_argument("--metadata-image-col", default="image_name")
    p.add_argument(
        "--keep-metadata-cols",
        default="point_id,year,latitude,longitude,altitude",
        help="Comma separated list of metadata columns to keep",
    )
    p.add_argument("--join-type", default="left", choices=["left", "inner"])

    p.add_argument("--num-classes", type=int, default=7806)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--denoise-eps", type=float, default=1e-6)
    p.add_argument("--min-tiles", type=int, default=1)

    p.add_argument(
        "--image-topk",
        type=int,
        default=64,
        help="If >0, keep only TopK classes after Noisy OR. Set 0 to keep dense.",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()
    spark = SparkSession.builder.getOrCreate()

    df = (
        spark.read.parquet(args.in_parquet_root)
        .select(args.image_col, args.logits_col)
        .where(F.col(args.logits_col).isNotNull())
    )

    image_topk = int(args.image_topk) if int(args.image_topk) > 0 else None

    aggregator = TemperatureNoisyORAggregator(
        num_classes=int(args.num_classes),
        temperature=float(args.temperature),
        denoise_eps=float(args.denoise_eps),
        output_dtype=np.float16,
        image_topk=image_topk,
        return_stats=False,
    )

    out_schema = StructType(
        [
            StructField(args.image_col, StringType(), nullable=False),
            StructField("teacher_image_probs_f16", BinaryType(), nullable=False),
            StructField("num_tiles", IntegerType(), nullable=False),
        ]
    )

    def agg_per_image(pdf_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for pdf in pdf_iter:
            if pdf.empty:
                continue

            image_name = str(pdf[args.image_col].iloc[0])
            buffers: List[bytes] = pdf[args.logits_col].tolist()

            if len(buffers) < int(args.min_tiles):
                continue

            tile_logits = f16_bytes_list_to_f32_matrix(buffers, num_classes=int(args.num_classes))
            p_image_f16 = aggregator.aggregate(tile_logits)

            yield pd.DataFrame(
                {
                    args.image_col: [image_name],
                    "teacher_image_probs_f16": [f16_vector_to_bytes(p_image_f16)],
                    "num_tiles": [int(len(buffers))],
                }
            )

    agg_df = df.groupBy(args.image_col).applyInPandas(agg_per_image, schema=out_schema)

    meta = spark.read.csv(args.metadata_csv, header=True, inferSchema=True)

    keep_cols = [c.strip() for c in args.keep_metadata_cols.split(",") if c.strip()]
    select_cols = [F.col(args.metadata_image_col).alias(args.image_col)]
    for c in keep_cols:
        select_cols.append(F.col(c))
    meta_small = meta.select(*select_cols)

    joined = agg_df.join(meta_small, on=args.image_col, how=args.join_type)

    (joined
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .parquet(args.out_parquet))


if __name__ == "__main__":
    main()
