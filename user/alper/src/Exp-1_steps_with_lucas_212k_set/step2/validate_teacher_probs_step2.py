#!/usr/bin/env python3
import argparse
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def decode_f16(binary_blob: bytes) -> np.ndarray:
    """
    Decode a Spark BinaryType field that stores float16 array bytes.
    Returns a 1D numpy float32 array.
    """
    if binary_blob is None:
        return None
    arr = np.frombuffer(binary_blob, dtype=np.float16)
    return arr.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to teacher_image_probs_step2 parquet folder",
    )
    parser.add_argument(
        "--expected_len",
        type=int,
        default=7806,
        help="Expected number of classes in the probability vector",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=5,
        help="Number of random images to print top-K for",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-K classes to display per sampled image",
    )
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("validate_teacher_probs_step2")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )

    df = spark.read.parquet(args.parquet).cache()

    print("\n=== BASIC CHECKS ===")
    print("Schema:")
    df.printSchema()

    n_rows = df.count()
    print(f"\nRow count: {n_rows}")

    print("\nnum_tiles distribution:")
    df.groupBy("num_tiles").count().orderBy("num_tiles").show(50, truncate=False)

    print("\nBinary size (bytes) distribution for teacher_image_probs_f16:")
    df.select(F.length("teacher_image_probs_f16").alias("nbytes")).groupBy("nbytes").count().orderBy("nbytes").show(
        50, truncate=False
    )

    expected_bytes = args.expected_len * 2  # float16 = 2 bytes
    print(f"\nExpected bytes per row (expected_len * 2): {expected_bytes}")

    bad_bytes = df.filter(F.length("teacher_image_probs_f16") != expected_bytes).count()
    print(f"Rows with unexpected byte length: {bad_bytes}")

    if bad_bytes > 0:
        print("\nExample rows with unexpected byte length:")
        df.filter(F.length("teacher_image_probs_f16") != expected_bytes) \
          .select("image_name", F.length("teacher_image_probs_f16").alias("nbytes")) \
          .show(20, truncate=False)

    print("\n=== DECODE + PROBABILITY SANITY ===")

    # Pull a small sample to driver for numeric checks
    sample = (
        df.select("image_name", "teacher_image_probs_f16")
        .orderBy(F.rand())
        .limit(max(args.sample_n, 20))  # take a bit more for stats
        .collect()
    )

    mins, maxs, nan_counts, gt1_counts, lt0_counts, nnz_1e6, nnz_1e3, nnz_1e2 = [], [], [], [], [], [], [], []

    for row in sample:
        name = row["image_name"]
        blob = row["teacher_image_probs_f16"]
        probs = decode_f16(blob)

        if probs is None:
            print(f"[WARN] {name}: probs is None")
            continue

        mins.append(float(np.min(probs)))
        maxs.append(float(np.max(probs)))
        nan_counts.append(int(np.isnan(probs).sum()))
        gt1_counts.append(int((probs > 1.0).sum()))
        lt0_counts.append(int((probs < 0.0).sum()))

        # sparsity style metrics
        nnz_1e6.append(int((probs > 1e-6).sum()))
        nnz_1e3.append(int((probs > 1e-3).sum()))
        nnz_1e2.append(int((probs > 1e-2).sum()))

    def summarize(name, arr):
        if not arr:
            return
        print(
            f"{name}: min={np.min(arr):.6g}  p50={np.median(arr):.6g}  max={np.max(arr):.6g}"
        )

    summarize("min(probs)", mins)
    summarize("max(probs)", maxs)
    print(f"NaNs per vector: min={min(nan_counts)} median={int(np.median(nan_counts))} max={max(nan_counts)}")
    print(f">1.0 counts:     min={min(gt1_counts)} median={int(np.median(gt1_counts))} max={max(gt1_counts)}")
    print(f"<0.0 counts:     min={min(lt0_counts)} median={int(np.median(lt0_counts))} max={max(lt0_counts)}")

    print("\nNonzero-ish counts (per vector):")
    print(f">1e-6: min={min(nnz_1e6)} median={int(np.median(nnz_1e6))} max={max(nnz_1e6)}")
    print(f">1e-3: min={min(nnz_1e3)} median={int(np.median(nnz_1e3))} max={max(nnz_1e3)}")
    print(f">1e-2: min={min(nnz_1e2)} median={int(np.median(nnz_1e2))} max={max(nnz_1e2)}")

    print("\n=== TOP-K INSPECTION (SAMPLED IMAGES) ===")
    sample_top = sample[: args.sample_n]
    for row in sample_top:
        name = row["image_name"]
        probs = decode_f16(row["teacher_image_probs_f16"])
        if probs is None:
            continue

        top_idx = np.argsort(-probs)[: args.topk]
        top_vals = probs[top_idx]

        print(f"\nImage: {name}")
        for rank, (i, v) in enumerate(zip(top_idx, top_vals), start=1):
            print(f"  {rank:02d}. class_id={int(i):4d}  prob={float(v):.6f}")

        # quick shape + range summary for that image
        print(
            f"  vec_len={probs.size}  min={probs.min():.6g}  max={probs.max():.6g}  "
            f"NaNs={(np.isnan(probs)).sum()}  >1={(probs>1).sum()}  <0={(probs<0).sum()}"
        )
        print(
            f"  count>1e-6={(probs>1e-6).sum()}  count>1e-3={(probs>1e-3).sum()}  count>1e-2={(probs>1e-2).sum()}"
        )

    spark.stop()


if __name__ == "__main__":
    main()
