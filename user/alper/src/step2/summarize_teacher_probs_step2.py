#!/usr/bin/env python3
import argparse
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType


def decode_f16(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float16)
    return arr.astype(np.float32)


def row_metrics(pdf_iter, expected_len: int):
    # mapInPandas iterator
    for pdf in pdf_iter:
        out = {
            "image_name": [],
            "vec_len": [],
            "sum": [],
            "min": [],
            "max": [],
            "mean": [],
            "entropy_nats": [],
            "top1": [],
            "top5_sum": [],
            "top10_sum": [],
            "count_gt_1e2": [],
            "count_gt_1e3": [],
            "count_gt_1e4": [],
        }

        for name, blob in zip(pdf["image_name"].tolist(), pdf["teacher_image_probs_f16"].tolist()):
            p = decode_f16(blob)

            if p.size != expected_len:
                # keep going, but mark obvious bad rows
                out["image_name"].append(name)
                out["vec_len"].append(int(p.size))
                out["sum"].append(float("nan"))
                out["min"].append(float("nan"))
                out["max"].append(float("nan"))
                out["mean"].append(float("nan"))
                out["entropy_nats"].append(float("nan"))
                out["top1"].append(float("nan"))
                out["top5_sum"].append(float("nan"))
                out["top10_sum"].append(float("nan"))
                out["count_gt_1e2"].append(-1)
                out["count_gt_1e3"].append(-1)
                out["count_gt_1e4"].append(-1)
                continue

            s = float(p.sum())
            mn = float(p.min())
            mx = float(p.max())
            mean = float(p.mean())

            # entropy assumes p is a distribution, so it is most informative if sum≈1
            eps = 1e-12
            p_clip = np.clip(p, eps, None)
            ent = float(-(p_clip * np.log(p_clip)).sum())

            top_idx = np.argsort(-p)
            top1 = float(p[top_idx[0]])
            top5_sum = float(p[top_idx[:5]].sum())
            top10_sum = float(p[top_idx[:10]].sum())

            out["image_name"].append(name)
            out["vec_len"].append(int(p.size))
            out["sum"].append(s)
            out["min"].append(mn)
            out["max"].append(mx)
            out["mean"].append(mean)
            out["entropy_nats"].append(ent)
            out["top1"].append(top1)
            out["top5_sum"].append(top5_sum)
            out["top10_sum"].append(top10_sum)
            out["count_gt_1e2"].append(int((p > 1e-2).sum()))
            out["count_gt_1e3"].append(int((p > 1e-3).sum()))
            out["count_gt_1e4"].append(int((p > 1e-4).sum()))

        import pandas as pd
        yield pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--expected_len", type=int, default=7806)
    ap.add_argument("--partitions", type=int, default=64)
    args = ap.parse_args()

    spark = SparkSession.builder.appName("summarize_teacher_probs_step2").getOrCreate()
    df = spark.read.parquet(args.parquet).select("image_name", "teacher_image_probs_f16").repartition(args.partitions)

    schema = StructType(
        [
            StructField("image_name", StringType(), True),
            StructField("vec_len", IntegerType(), True),
            StructField("sum", FloatType(), True),
            StructField("min", FloatType(), True),
            StructField("max", FloatType(), True),
            StructField("mean", FloatType(), True),
            StructField("entropy_nats", FloatType(), True),
            StructField("top1", FloatType(), True),
            StructField("top5_sum", FloatType(), True),
            StructField("top10_sum", FloatType(), True),
            StructField("count_gt_1e2", IntegerType(), True),
            StructField("count_gt_1e3", IntegerType(), True),
            StructField("count_gt_1e4", IntegerType(), True),
        ]
    )

    metrics = df.mapInPandas(lambda it: row_metrics(it, args.expected_len), schema=schema).cache()

    n = metrics.count()
    print(f"\nRows summarized: {n}")

    # Percentiles for the most important fields
    fields = ["sum", "min", "max", "entropy_nats", "top1", "top5_sum", "top10_sum"]
    probs = [0.0, 0.5, 0.9, 0.99, 1.0]

    print("\n=== PERCENTILES ===")
    for f in fields:
        qs = metrics.approxQuantile(f, probs, 0.0)
        print(f"{f:12s}  p0={qs[0]:.6g}  p50={qs[1]:.6g}  p90={qs[2]:.6g}  p99={qs[3]:.6g}  p100={qs[4]:.6g}")

    # How many look like softmax vectors?
    # If sum is near 1 for almost all rows, that is the smoking gun.
    near1 = metrics.filter((F.col("sum") > 0.99) & (F.col("sum") < 1.01)).count()
    print(f"\nRows with sum in [0.99, 1.01]: {near1} / {n}")

    # Basic sanity counts
    bad_len = metrics.filter(F.col("vec_len") != args.expected_len).count()
    print(f"Rows with wrong vec_len: {bad_len}")

    print("\n=== QUICK LOOK (10 rows) ===")
    metrics.select("image_name", "sum", "min", "max", "top1", "top10_sum", "count_gt_1e2").orderBy(F.rand()).show(
        10, truncate=False
    )

    spark.stop()


if __name__ == "__main__":
    main()
