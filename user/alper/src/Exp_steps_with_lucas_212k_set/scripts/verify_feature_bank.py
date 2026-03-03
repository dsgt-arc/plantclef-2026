#!/usr/bin/env python3
"""
verify_feature_bank.py

Verifies LUCAS feature bank parquet outputs produced by extract_lucas_embeddings.py.

What it checks
1. Schema contains expected columns and binary types for float16 bytes
2. Row count and distinct image count
3. Tile index range and count matches grid_size
4. Per image tile completeness, missing tiles, duplicate tile indices
5. Byte length sanity and inferred embedding and logits dimensions
6. Decode float16 arrays and check finiteness, all zeros, basic stats
7. Compare embeddings across tiles within an image, detect identical tiles
8. Teacher logits softmax sanity for a few samples

Usage examples
python verify_feature_bank.py \
  --parquet /path/to/feature_bank/shard_0000 \
  --grid-size 4

python verify_feature_bank.py \
  --parquet /path/to/feature_bank \
  --grid-size 4 \
  --expected-images 10000
"""

import argparse
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import BinaryType, IntegerType, StringType


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str


def _spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )


def _expect_columns(df: DataFrame, required: List[str]) -> Tuple[bool, str]:
    cols = set(df.columns)
    missing = [c for c in required if c not in cols]
    if missing:
        return False, f"Missing columns: {missing}. Found: {sorted(cols)}"
    return True, "All required columns are present."


def _expect_types(df: DataFrame) -> Tuple[bool, str]:
    schema = df.schema
    expected = {
        "image_name": StringType,
        "tile_index": IntegerType,
        "embedding_f16": BinaryType,
        "teacher_logits_f16": BinaryType,
    }
    problems = []
    for col, typ in expected.items():
        field = schema[col]
        if not isinstance(field.dataType, typ):
            problems.append(f"{col} has type {field.dataType} expected {typ().__class__.__name__}")
    if problems:
        return False, "; ".join(problems)
    return True, "Column types look correct."


def _basic_counts(df: DataFrame, grid_size: int, expected_images: Optional[int]) -> List[CheckResult]:
    res: List[CheckResult] = []

    rows = df.count()
    n_images = df.select("image_name").distinct().count()
    tiles_per_image = grid_size * grid_size

    details = f"rows={rows}, distinct_images={n_images}, grid_size={grid_size}, tiles_per_image={tiles_per_image}"
    ok = True
    if expected_images is not None and n_images != expected_images:
        ok = False
        details += f", expected_images={expected_images} mismatch"
    res.append(CheckResult("Counts", ok, details))

    approx_expected_rows = n_images * tiles_per_image
    ratio = rows / approx_expected_rows if approx_expected_rows else 0.0
    ok_rows = 0.98 <= ratio <= 1.02
    res.append(
        CheckResult(
            "Row count roughly matches images times tiles",
            ok_rows,
            f"rows={rows}, images*tiles={approx_expected_rows}, ratio={ratio:.4f}",
        )
    )
    return res


def _tile_index_checks(df: DataFrame, grid_size: int) -> List[CheckResult]:
    res: List[CheckResult] = []
    tiles_per_image = grid_size * grid_size

    stats = df.select(
        F.min("tile_index").alias("min_tile"),
        F.max("tile_index").alias("max_tile"),
        F.countDistinct("tile_index").alias("num_tile_ids"),
    ).collect()[0]

    min_tile = int(stats["min_tile"]) if stats["min_tile"] is not None else None
    max_tile = int(stats["max_tile"]) if stats["max_tile"] is not None else None
    num_tile_ids = int(stats["num_tile_ids"]) if stats["num_tile_ids"] is not None else None

    ok = (min_tile == 0) and (max_tile == tiles_per_image - 1) and (num_tile_ids == tiles_per_image)
    res.append(
        CheckResult(
            "Tile index range and unique ids",
            ok,
            f"min_tile={min_tile}, max_tile={max_tile}, num_tile_ids={num_tile_ids}, expected_max={tiles_per_image - 1}",
        )
    )
    return res


def _per_image_completeness(df: DataFrame, grid_size: int, show_bad_n: int) -> List[CheckResult]:
    res: List[CheckResult] = []
    tiles_per_image = grid_size * grid_size

    per_image = df.groupBy("image_name").agg(
        F.count("*").alias("n_rows"),
        F.countDistinct("tile_index").alias("n_tile_ids"),
    )

    summary = per_image.select(
        F.min("n_rows").alias("min_rows"),
        F.max("n_rows").alias("max_rows"),
        F.min("n_tile_ids").alias("min_ids"),
        F.max("n_tile_ids").alias("max_ids"),
    ).collect()[0]

    min_rows = int(summary["min_rows"])
    max_rows = int(summary["max_rows"])
    min_ids = int(summary["min_ids"])
    max_ids = int(summary["max_ids"])

    ok = (min_rows == tiles_per_image) and (max_rows == tiles_per_image) and (min_ids == tiles_per_image) and (max_ids == tiles_per_image)
    res.append(
        CheckResult(
            "Per image tile completeness",
            ok,
            f"min_rows={min_rows}, max_rows={max_rows}, min_ids={min_ids}, max_ids={max_ids}, expected={tiles_per_image}",
        )
    )

    bad = per_image.filter((F.col("n_rows") != tiles_per_image) | (F.col("n_tile_ids") != tiles_per_image))
    bad_count = bad.count()
    res.append(CheckResult("Images with missing or duplicate tiles", bad_count == 0, f"bad_images={bad_count}"))

    if bad_count > 0 and show_bad_n > 0:
        sample_bad = bad.orderBy(F.desc("n_rows")).limit(show_bad_n).collect()
        lines = []
        for r in sample_bad:
            lines.append(f"{r['image_name']} rows={r['n_rows']} distinct_tile_ids={r['n_tile_ids']}")
        res.append(CheckResult("Sample bad images", False, " | ".join(lines)))

    return res


def _byte_length_checks(df: DataFrame, sample_n: int) -> List[CheckResult]:
    res: List[CheckResult] = []

    sample = df.select("embedding_f16", "teacher_logits_f16").limit(max(sample_n, 1)).collect()
    if not sample:
        return [CheckResult("Byte length check", False, "No rows found in parquet.")]

    emb_lens = [len(r["embedding_f16"]) for r in sample]
    logit_lens = [len(r["teacher_logits_f16"]) for r in sample]

    ok_even = all(x % 2 == 0 for x in emb_lens) and all(x % 2 == 0 for x in logit_lens)
    res.append(
        CheckResult(
            "Byte lengths are even so float16 decoding is possible",
            ok_even,
            f"embedding_bytes={sorted(set(emb_lens))}, logits_bytes={sorted(set(logit_lens))}",
        )
    )

    emb_dims = [x // 2 for x in emb_lens]
    logit_dims = [x // 2 for x in logit_lens]
    res.append(
        CheckResult(
            "Inferred dimensions from byte lengths",
            True,
            f"embedding_dim_candidates={sorted(set(emb_dims))}, logits_dim_candidates={sorted(set(logit_dims))}",
        )
    )
    return res


def _decode_stats_checks(df: DataFrame, sample_n: int) -> List[CheckResult]:
    res: List[CheckResult] = []
    rows = df.select("image_name", "tile_index", "embedding_f16", "teacher_logits_f16").limit(max(sample_n, 1)).collect()
    if not rows:
        return [CheckResult("Decode stats check", False, "No rows found in parquet.")]

    bad_finite = 0
    bad_zero = 0
    emb_std_zeros = 0
    details_examples = []

    for r in rows:
        emb = np.frombuffer(r["embedding_f16"], dtype=np.float16).astype(np.float32)
        logits = np.frombuffer(r["teacher_logits_f16"], dtype=np.float16).astype(np.float32)

        finite = np.isfinite(emb).all() and np.isfinite(logits).all()
        if not finite:
            bad_finite += 1

        if np.all(emb == 0) or np.all(logits == 0):
            bad_zero += 1

        if float(np.std(emb)) == 0.0:
            emb_std_zeros += 1

        if len(details_examples) < 3:
            details_examples.append(
                f"{r['image_name']} tile={r['tile_index']} emb_min={emb.min():.4f} emb_max={emb.max():.4f} emb_mean={emb.mean():.4f} emb_std={emb.std():.4f}"
            )

    res.append(CheckResult("Decoded values are finite", bad_finite == 0, f"non_finite_samples={bad_finite} of {len(rows)}"))
    res.append(CheckResult("Not all zeros", bad_zero == 0, f"all_zero_samples={bad_zero} of {len(rows)}"))
    res.append(CheckResult("Embedding standard deviation not always zero", emb_std_zeros == 0, f"zero_std_samples={emb_std_zeros} of {len(rows)}"))

    res.append(CheckResult("Sample embedding stats", True, " | ".join(details_examples)))
    return res


def _tile_similarity_check(df: DataFrame, grid_size: int, max_images: int) -> List[CheckResult]:
    res: List[CheckResult] = []
    tiles_per_image = grid_size * grid_size

    image_names = [r["image_name"] for r in df.select("image_name").distinct().limit(max_images).collect()]
    if not image_names:
        return [CheckResult("Tile similarity check", False, "No images found.")]

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(a.dot(b) / denom)

    identical_pairs = 0
    checked = 0
    worst = None

    for img in image_names:
        tiles = (
            df.filter(F.col("image_name") == img)
            .orderBy("tile_index")
            .select("tile_index", "embedding_f16")
            .collect()
        )
        if len(tiles) != tiles_per_image:
            continue

        embs = [np.frombuffer(t["embedding_f16"], dtype=np.float16).astype(np.float32) for t in tiles]
        pairs = [(0, 1), (0, tiles_per_image // 2), (tiles_per_image - 2, tiles_per_image - 1)]
        for i, j in pairs:
            checked += 1
            c = cos(embs[i], embs[j])
            l2 = float(np.linalg.norm(embs[i] - embs[j]))
            if l2 == 0.0:
                identical_pairs += 1
            if worst is None or l2 < worst[0]:
                worst = (l2, c, img, i, j)

    ok = identical_pairs == 0
    detail = f"checked_pairs={checked}, identical_pairs={identical_pairs}"
    if worst is not None:
        detail += f", smallest_l2={worst[0]:.6f}, cos_at_smallest_l2={worst[1]:.6f}, image={worst[2]}, tiles={worst[3]} and {worst[4]}"
    res.append(CheckResult("Tile embeddings are not identical within an image", ok, detail))
    return res


def _logits_softmax_check(df: DataFrame, sample_n: int) -> List[CheckResult]:
    res: List[CheckResult] = []
    rows = df.select("image_name", "tile_index", "teacher_logits_f16").limit(max(sample_n, 1)).collect()
    if not rows:
        return [CheckResult("Logits softmax check", False, "No rows found.")]

    bad = 0
    examples = []
    for r in rows[:sample_n]:
        logits = np.frombuffer(r["teacher_logits_f16"], dtype=np.float16).astype(np.float32)
        x = logits - float(np.max(logits))
        probs = np.exp(x)
        s = float(np.sum(probs))
        if not np.isfinite(s) or s <= 0:
            bad += 1
            continue
        probs = probs / s
        psum = float(np.sum(probs))
        top = np.sort(probs)[-5:][::-1]
        if len(examples) < 3:
            examples.append(f"{r['image_name']} tile={r['tile_index']} prob_sum={psum:.6f} top5={np.array2string(top, precision=4)}")
        if abs(psum - 1.0) > 1e-3:
            bad += 1

    res.append(CheckResult("Softmax over logits sums to 1", bad == 0, f"bad_samples={bad} of {min(sample_n, len(rows))}"))
    if examples:
        res.append(CheckResult("Sample softmax stats", True, " | ".join(examples)))
    return res


def _print_results(results: List[CheckResult]) -> int:
    failed = 0
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"[{status}] {r.name}: {r.details}")
        if not r.ok:
            failed += 1
    print()
    print(f"Total checks: {len(results)}. Failed: {failed}.")
    return 0 if failed == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Parquet path. Can be a shard folder or parent folder.")
    parser.add_argument("--grid-size", type=int, default=4, help="Grid size used for tiling. Example 4 means 4x4 tiles.")
    parser.add_argument("--expected-images", type=int, default=None, help="Optional expected number of distinct images.")
    parser.add_argument("--sample-bytes", type=int, default=10, help="How many rows to sample for byte length checks.")
    parser.add_argument("--sample-decode", type=int, default=20, help="How many rows to sample for decode stats.")
    parser.add_argument("--sample-softmax", type=int, default=10, help="How many rows to sample for logits softmax check.")
    parser.add_argument("--max-images-sim", type=int, default=5, help="How many images to use for within image tile similarity checks.")
    parser.add_argument("--show-bad-n", type=int, default=10, help="How many bad images to print if completeness fails.")
    args = parser.parse_args()

    spark = _spark("verify_feature_bank")
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(args.parquet)

    results: List[CheckResult] = []

    ok_cols, detail_cols = _expect_columns(df, ["image_name", "tile_index", "embedding_f16", "teacher_logits_f16"])
    results.append(CheckResult("Required columns", ok_cols, detail_cols))
    if not ok_cols:
        return _print_results(results)

    ok_types, detail_types = _expect_types(df)
    results.append(CheckResult("Expected column types", ok_types, detail_types))

    results.extend(_basic_counts(df, args.grid_size, args.expected_images))
    results.extend(_tile_index_checks(df, args.grid_size))
    results.extend(_per_image_completeness(df, args.grid_size, args.show_bad_n))
    results.extend(_byte_length_checks(df, args.sample_bytes))
    results.extend(_decode_stats_checks(df, args.sample_decode))
    results.extend(_tile_similarity_check(df, args.grid_size, args.max_images_sim))
    results.extend(_logits_softmax_check(df, args.sample_softmax))

    return _print_results(results)


if __name__ == "__main__":
    sys.exit(main())
