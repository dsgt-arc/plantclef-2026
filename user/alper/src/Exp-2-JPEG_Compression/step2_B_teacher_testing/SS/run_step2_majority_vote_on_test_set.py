#!/usr/bin/env python3
"""
Majority vote aggregation of per-tile logits, matching the reference paper pipeline.

Reference order of operations (from classification/aggregation.py):
  1. Per tile: softmax(T=1.0) on ALL classes -> top-K species per tile
  2. Group by base_quadrat_id (strips date suffix to combine seasonal images)
  3. Across ALL tiles from ALL seasonal images: count species frequency -> rank
  4. Assign ranked species to every original quadrat_id in the group
  5. Geographic filtering applied LATER in the submission script (NOT here)
"""

import sys
from pathlib import Path

# Add Exp-2-JPEG_Compression to import path so sibling folder "step2" is visible
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import re
import glob
import json
import argparse
from typing import List, Dict
from collections import Counter, defaultdict

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


def extract_base_quadrat_id(quadrat_id: str) -> str:
    """
    Extract base quadrat ID by removing date suffixes.
    Copied from reference: classification/aggregation.py:22-33

    Groups seasonal images of the same location together.
    E.g., CBN-PdlC-A1-20130807 -> CBN-PdlC-A1
    """
    patterns = [
        (r"^(CBN-.*?-.*?)-\d{8}$", 1),       # CBN
        (r"^(GUARDEN-.*?-.*?)-.*$", 1),       # GUARDEN-AMB
        (r"^(LISAH-.*?)-\d{8}$", 1),          # LISAH
        (r"^(OPTMix-\d+)-.*$", 1),            # OPTMix
        (r"^(RNNB-\d+-\d+)-\d{8}$", 1),       # RNNB
    ]
    for pattern, group_idx in patterns:
        match = re.match(pattern, quadrat_id)
        if match:
            return match.group(group_idx)
    return quadrat_id  # fallback


def get_per_tile_species(
    tile_logits: np.ndarray,
    per_tile_topk: int,
) -> List[List[int]]:
    """
    Per tile: softmax(T=1.0) -> top-K species indices.
    Returns list of lists (one per tile).
    """
    n_tiles, num_classes = tile_logits.shape
    probs = softmax(tile_logits)

    per_tile = []
    for t in range(n_tiles):
        k = min(per_tile_topk, num_classes)
        top_indices = np.argpartition(probs[t], -k)[-k:]
        per_tile.append([int(idx) for idx in top_indices])

    return per_tile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Majority vote aggregation of per-tile logits (matching reference paper)"
    )

    p.add_argument("--in-parquet-root", required=True, help="Folder containing shard_0000 etc")
    p.add_argument("--pattern", default="shard_*/*.parquet", help="Glob under in-parquet-root for part files")
    p.add_argument("--out-parquet", required=True, help="Output parquet path")

    p.add_argument("--id-col", default="quadrat_id")
    p.add_argument("--logits-col", default="teacher_logits_f16")

    p.add_argument("--num-classes", type=int, default=7806)
    p.add_argument("--n-tiles", type=int, default=16)

    p.add_argument("--per-tile-topk", type=int, default=9,
                    help="Number of top species to extract per tile for voting (reference uses 9)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    files = sorted(glob.glob(os.path.join(args.in_parquet_root, args.pattern)))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {args.in_parquet_root} matching {args.pattern}")

    print(f"Found {len(files)} parquet files")

    # --- Pass 1: collect per-tile species for every image ---
    # image_tiles[qid] = list of per-tile species lists
    image_tiles: Dict[str, List[List[int]]] = {}

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

            per_tile = get_per_tile_species(tile_logits, int(args.per_tile_topk))
            image_tiles[qid] = per_tile

    print(f"Loaded per-tile species for {len(image_tiles)} images")

    # --- Pass 2: group by base_quadrat_id and aggregate ---
    # Groups seasonal images of the same location (matching reference aggregation.py)
    groups: Dict[str, List[str]] = defaultdict(list)
    for qid in image_tiles:
        base_id = extract_base_quadrat_id(qid)
        groups[base_id].append(qid)

    n_grouped = sum(1 for g in groups.values() if len(g) > 1)
    print(f"Grouped into {len(groups)} base quadrats ({n_grouped} with multiple seasonal images)")

    # For each group: flatten all per-tile species -> Counter -> ranked
    group_ranked: Dict[str, List[int]] = {}
    group_votes: Dict[str, List[int]] = {}

    for base_id, qids in groups.items():
        vote_counter = Counter()
        total_tiles = 0
        for qid in qids:
            for tile_species in image_tiles[qid]:
                for sp in tile_species:
                    vote_counter[sp] += 1
                total_tiles += 1

        ranked = [species for species, _ in vote_counter.most_common()]
        votes = [vote_counter[s] for s in ranked]
        group_ranked[base_id] = ranked
        group_votes[base_id] = votes

    # --- Output: one row per original quadrat_id, each with its group's ranked species ---
    out_ids: List[str] = []
    out_ranked: List[str] = []
    out_votes: List[str] = []

    for qid in sorted(image_tiles.keys()):
        base_id = extract_base_quadrat_id(qid)
        out_ids.append(qid)
        out_ranked.append(json.dumps(group_ranked[base_id]))
        out_votes.append(json.dumps(group_votes[base_id]))

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
  --in-parquet-root /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/feature_bank \
  --pattern "shard_*/*.parquet" \
  --out-parquet /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/teacher_majority_vote/majority_vote_ranked.parquet \
  --logits-col teacher_logits_f16 \
  --num-classes 7806 \
  --n-tiles 16 \
  --per-tile-topk 9

'''
