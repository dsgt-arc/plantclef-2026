#!/usr/bin/env python3
"""
Majority vote aggregation of per-tile logits, matching the reference paper pipeline.

Reference order of operations:
  1. Per tile: softmax(T=1.0) on ALL classes -> top-K species per tile
  2. Across 16 tiles: count species frequency -> rank by frequency
  3. Geographic filtering applied LATER in the submission script (NOT here)
"""

import sys
from pathlib import Path

# Add Exp-2-JPEG_Compression to import path so sibling folder "step2" is visible
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import glob
import json
import argparse
from typing import List
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def f16_bytes_to_f32_matrix(buf: bytes, n_tiles: int, num_classes: int) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.float16)
    expected = n_tiles * num_classes
    if arr.size != expected:
        raise ValueError(f"Expected {expected} f16 values, got {arr.size}")
    return arr.reshape(n_tiles, num_classes).astype(np.float32, copy=False)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Standard softmax (T=1.0) along axis=1."""
    x = logits - np.max(logits, axis=1, keepdims=True)
    ex = np.exp(x, dtype=np.float32)
    return ex / np.sum(ex, axis=1, keepdims=True, dtype=np.float32)


def majority_vote_aggregate(
    tile_logits: np.ndarray,
    per_tile_topk: int,
) -> List[int]:
    """
    Majority vote aggregation matching the reference paper.

    1. Per tile: softmax(T=1.0) on full 7806 classes -> top-K species indices
    2. Across tiles: count frequency of each species
    3. Return species ranked by vote count (desc), ties broken by frequency order.

    NO geo mask is applied here — that happens post-aggregation.
    """
    n_tiles, num_classes = tile_logits.shape
    probs = softmax(tile_logits)  # [T, C]

    # Collect votes: for each tile, get top-K species
    vote_counter = Counter()

    for t in range(n_tiles):
        tile_probs = probs[t]
        k = min(per_tile_topk, num_classes)
        top_indices = np.argpartition(tile_probs, -k)[-k:]

        for idx in top_indices:
            vote_counter[int(idx)] += 1

    # Rank by vote count (desc) — matches reference Counter.most_common()
    ranked = [species for species, _ in vote_counter.most_common()]

    return ranked


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Majority vote aggregation of per-tile logits (replaces Noisy-OR)"
    )

    p.add_argument("--in-parquet-root", required=True, help="Folder containing shard_0000 etc")
    p.add_argument("--pattern", default="shard_*/*.parquet", help="Glob under in-parquet-root for part files")
    p.add_argument("--out-parquet", required=True, help="Output parquet path")

    p.add_argument("--id-col", default="quadrat_id")
    p.add_argument("--logits-col", default="teacher_logits_f16")

    p.add_argument("--num-classes", type=int, default=7806)
    p.add_argument("--n-tiles", type=int, default=16)

    p.add_argument("--per-tile-topk", type=int, default=5,
                    help="Number of top species to extract per tile for voting (reference paper uses 5)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    files = sorted(glob.glob(os.path.join(args.in_parquet_root, args.pattern)))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {args.in_parquet_root} matching {args.pattern}")

    print(f"Found {len(files)} parquet files")

    out_ids: List[str] = []
    out_ranked: List[str] = []
    out_votes: List[str] = []

    for fp in files:
        table = pq.read_table(fp, columns=[args.id_col, args.logits_col])
        df = table.to_pandas()

        for _, r in df.iterrows():
            qid = str(r[args.id_col])
            buf = r[args.logits_col]
            if buf is None:
                continue

            tile_logits = f16_bytes_to_f32_matrix(
                buf, n_tiles=int(args.n_tiles), num_classes=int(args.num_classes)
            )

            ranked_species = majority_vote_aggregate(
                tile_logits,
                per_tile_topk=int(args.per_tile_topk),
            )

            # Recompute vote counts for the output (for analysis)
            probs = softmax(tile_logits)
            vote_counter = Counter()
            for t in range(tile_logits.shape[0]):
                k = min(int(args.per_tile_topk), int(args.num_classes))
                top_idx = np.argpartition(probs[t], -k)[-k:]
                for idx in top_idx:
                    vote_counter[int(idx)] += 1
            vote_counts = [vote_counter.get(s, 0) for s in ranked_species]

            out_ids.append(qid)
            out_ranked.append(json.dumps(ranked_species))
            out_votes.append(json.dumps(vote_counts))

    out_table = pa.Table.from_pydict(
        {
            args.id_col: out_ids,
            "ranked_class_indices": out_ranked,
            "vote_counts": out_votes,
        }
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_parquet)), exist_ok=True)
    pq.write_table(out_table, args.out_parquet, compression="snappy")
    print(f"Wrote {out_table.num_rows} rows to {args.out_parquet}")


if __name__ == "__main__":
    main()


'''
python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-2-JPEG_Compression/step2_B_teacher_testing/run_step2_majority_vote_on_test_set.py \
  --in-parquet-root /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/feature_bank_JPEG_compressed \
  --pattern "shard_*/*.parquet" \
  --out-parquet /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/teacher_majority_vote/majority_vote_ranked.parquet \
  --logits-col teacher_logits_f16 \
  --num-classes 7806 \
  --n-tiles 16 \
  --per-tile-topk 5

'''
