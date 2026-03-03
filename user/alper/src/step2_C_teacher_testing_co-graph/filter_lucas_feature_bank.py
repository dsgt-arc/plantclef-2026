#!/usr/bin/env python3
import argparse
import math
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in-root", required=True,
                    help="Input parquet root, eg .../lucas_212k/feature_bank")
    ap.add_argument("--metadata-csv", required=True,
                    help="Wide SWE metadata csv containing image_name, latitude, longitude, etc")
    ap.add_argument("--out-root", required=True,
                    help="Output parquet root for filtered and re sharded feature bank")

    ap.add_argument("--images-per-shard", type=int, default=4000,
                    help="Target unique images per shard folder")
    ap.add_argument("--num-shards", type=int, default=0,
                    help="If set > 0, overrides images-per-shard")
    ap.add_argument("--files-per-shard", type=int, default=1,
                    help="Approx number of parquet part files per shard folder")

    args = ap.parse_args()

    spark = (
        SparkSession.builder
        .appName("filter_and_reshard_lucas_feature_bank")

        # You already tried this, keep it
        .config("spark.sql.parquet.enableVectorizedReader", "false")

        # Reduce how much parquet tries to read per batch
        .config("spark.sql.parquet.columnarReaderBatchSize", "256")

        # Force parquet not to do big vectored range reads
        .config("spark.sql.parquet.enableVectorizedReader", "false")  # ok duplicated
        .config("spark.hadoop.fs.file.impl.disable.cache", "true")
        .getOrCreate()
    )

    # Load allow list from metadata
    meta = (
        spark.read
        .option("header", True)
        .csv(args.metadata_csv)
    )

    if "image_name" not in meta.columns:
        raise RuntimeError(f"metadata-csv must contain image_name column. Found columns: {meta.columns}")

    allow = (
        meta.select(F.col("image_name").cast("string").alias("image_name"))
        .dropna()
        .dropDuplicates()
    )

    allow_count = allow.count()

    # Decide shard count
    if args.num_shards and args.num_shards > 0:
        num_shards = int(args.num_shards)
    else:
        num_shards = int(math.ceil(allow_count / float(args.images_per_shard)))

    print(f"Allowed unique images: {allow_count}")
    print(f"Target images per shard: {args.images_per_shard}")
    print(f"Num shards to write: {num_shards}")

    # Read feature bank
    fb = spark.read.parquet(args.in_root)

    required_cols = {"image_name", "tile_index", "embedding_f16", "teacher_logits_f16"}
    missing = required_cols - set(fb.columns)
    if missing:
        raise RuntimeError(f"Input feature bank missing columns: {sorted(missing)}. Found: {fb.columns}")

    fb = fb.select("image_name", "tile_index", "embedding_f16", "teacher_logits_f16")

    # Filter to SWE images
    # Broadcast because allow list is small
    fb_filt = fb.join(F.broadcast(allow), on="image_name", how="semi")

    # Assign new shard id deterministically from image_name
    # pmod(xxhash64(...), num_shards) gives 0..num_shards-1
    fb_filt = fb_filt.withColumn(
        "shard_idx",
        F.pmod(F.xxhash64(F.col("image_name")), F.lit(num_shards)).cast("int")
    )

    fb_filt = fb_filt.withColumn(
        "shard_id",
        F.format_string("shard_%04d", F.col("shard_idx"))
    ).drop("shard_idx")

    # Control number of output files.
    # repartition by shard_id and optionally bump partitions to get multiple files per shard.
    # Total partitions roughly equals num_shards * files_per_shard.
    target_partitions = max(num_shards * max(1, args.files_per_shard), num_shards)
    fb_filt = fb_filt.repartition(target_partitions, F.col("shard_id"))

    # Write partitioned output. This creates folders shard_id=shard_0000, etc.
    (
        fb_filt.write
        .mode("overwrite")
        .partitionBy("shard_id")
        .parquet(args.out_root)
    )

    # Quick sanity stats
    out_images = fb_filt.select("image_name").distinct().count()
    out_rows = fb_filt.count()
    print(f"Output distinct images: {out_images}")
    print(f"Output rows (tiles): {out_rows}")
    print(f"Wrote output to: {args.out_root}")

    spark.stop()


if __name__ == "__main__":
    main()


'''
spark-submit \
  --driver-memory 32g \
  --conf spark.sql.parquet.enableVectorizedReader=false \
  --conf spark.sql.parquet.columnarReaderBatchSize=256 \
  /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/step2_C_teacher_testing_co-graph/filter_lucas_feature_bank.py \
  --in-root "/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/lucas_212k/feature_bank/shard_*" \
  --metadata-csv /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/data/raw/metadata_lucas/LUCAS_metadata_SWE_Wide_lat35_50_lon-10_10_77k.csv \
  --out-root /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/lucas_sw77k/feature_bank_shard_4k \
  --images-per-shard 4000 \
  --files-per-shard 1

'''