#!/usr/bin/env python3

import sys
from pathlib import Path

# Add Exp-2-JPEG_Compression to import path so sibling folder "step2" is visible
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import glob
import json
import argparse
from typing import List, Tuple
from collections import Counter

import numpy as np
import pandas as pd
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
    class_mask: np.ndarray = None,
) -> List[int]:
    """
    Majority vote aggregation matching the reference paper approach.

    1. Per tile: softmax (T=1.0) -> top-K species indices
    2. Across tiles: count frequency of each species
    3. Return species ranked by vote count (descending), ties broken by
       highest probability among voting tiles.
    """
    n_tiles, num_classes = tile_logits.shape
    probs = softmax(tile_logits)  # [T, C]

    if class_mask is not None:
        probs[:, ~class_mask] = 0.0

    # Collect votes: for each tile, get top-K species
    vote_counter = Counter()
    best_prob = {}  # track best probability per species for tie-breaking

    for t in range(n_tiles):
        tile_probs = probs[t]
        k = min(per_tile_topk, num_classes)
        top_indices = np.argpartition(tile_probs, -k)[-k:]

        for idx in top_indices:
            idx = int(idx)
            if tile_probs[idx] > 0:
                vote_counter[idx] += 1
                if idx not in best_prob or tile_probs[idx] > best_prob[idx]:
                    best_prob[idx] = float(tile_probs[idx])

    # Rank by vote count (desc), then by best probability (desc)
    ranked = sorted(
        vote_counter.keys(),
        key=lambda c: (vote_counter[c], best_prob.get(c, 0.0)),
        reverse=True,
    )

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

    p.add_argument("--per-tile-topk", type=int, default=10,
                    help="Number of top species to extract per tile for voting")

    p.add_argument("--class-mask-npy", default=None,
                    help="Optional boolean mask [num_classes] to restrict allowed species")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    files = sorted(glob.glob(os.path.join(args.in_parquet_root, args.pattern)))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {args.in_parquet_root} matching {args.pattern}")

    print(f"Found {len(files)} parquet files")

    class_mask = None
    if args.class_mask_npy:
        class_mask = np.load(args.class_mask_npy)
        if class_mask.dtype != np.bool_:
            class_mask = class_mask.astype(np.bool_)
        print(f"Loaded class mask: {class_mask.sum()} allowed out of {class_mask.shape[0]}")

    out_ids: List[str] = []
    out_ranked: List[str] = []  # JSON-encoded list of class indices ranked by votes
    out_votes: List[str] = []   # JSON-encoded vote counts for each ranked species

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
                class_mask=class_mask,
            )

            out_ids.append(qid)
            out_ranked.append(json.dumps(ranked_species))
            # Store vote counts for analysis
            vote_counts = []
            probs = softmax(tile_logits)
            if class_mask is not None:
                probs[:, ~class_mask] = 0.0
            vote_counter = Counter()
            for t in range(tile_logits.shape[0]):
                k = min(int(args.per_tile_topk), int(args.num_classes))
                top_idx = np.argpartition(probs[t], -k)[-k:]
                for idx in top_idx:
                    if probs[t, idx] > 0:
                        vote_counter[int(idx)] += 1
            vote_counts = [vote_counter.get(s, 0) for s in ranked_species]
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
  --per-tile-topk 10 \
  --class-mask-npy /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/species_masking/sw_europe_mask_bool.npy

'''
