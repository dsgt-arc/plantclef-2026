#!/usr/bin/env python3
"""Stage 2: CPU classification for PlantCLEF 2026 grid ablation.

Reads parquet files produced by inference.py (Stage 1) and applies classification
methods to produce submission CSVs. No GPU required.

Usage:
    # Expanded-list aggregation (best method, F1=0.23)
    python classify.py --input logits/grid_4x4.parquet \
        --method expanded-list --top-k 5 --max-species 20

    # Union aggregation (all unique species, no cap)
    python classify.py --input logits/grid_4x4.parquet \
        --method union --top-k 5

    # Max-probability aggregation (organizers' approach)
    python classify.py --input logits/grid_4x4.parquet \
        --method max-prob --min-score 0.01

    # With quadrat-level post-processing
    python classify.py --input logits/grid_4x4.parquet \
        --method expanded-list --top-k 5 --max-species 20 \
        --quadrat-agg --quadrat-top-k 10
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Default paths on PACE shared storage
DEFAULT_CLASS_MAPPING = "/storage/project/ps-clef2026_plantclef-0/shared/data/raw/class_id_to_species_mapping/class_mapping.txt"


def load_class_mapping(class_mapping_path: str) -> dict[int, int]:
    """Load class index -> species ID mapping from class_mapping.txt."""
    with open(class_mapping_path) as f:
        return {i: int(line.strip()) for i, line in enumerate(f)}


def load_parquet(path: Path) -> pd.DataFrame:
    """Load parquet and reconstruct probability arrays.

    Returns DataFrame with columns: image_name, grid_size, tiles_per_image, probs
    where probs is a numpy array of shape (tiles_per_image, 7806) in float32.
    """
    table = pq.read_table(path)
    df = table.to_pandas()
    # Reconstruct numpy arrays from binary, promote to float32 for arithmetic
    df["probs"] = [
        np.frombuffer(row, dtype=np.float16).reshape(t, -1).astype(np.float32)
        for row, t in zip(df["probs"], df["tiles_per_image"])
    ]
    return df


# ---- Aggregation methods ----


def agg_union(
    probs: np.ndarray,
    top_k: int,
    cid_to_spid: dict[int, int],
    max_species: int | None = None,
) -> list[int]:
    """Union of top-k species per tile, sorted by frequency.

    Returns all unique species from every tile's top-k predictions, sorted by
    how many tiles predicted each species (descending). Optional max_species
    caps the unique species count via Counter.most_common().
    """
    top_k_ids = np.argpartition(-probs, top_k, axis=1)[:, :top_k]
    species_counts: Counter = Counter()
    for tile_top_k in top_k_ids:
        for class_idx in tile_top_k:
            species_counts[cid_to_spid[int(class_idx)]] += 1
    return [sp for sp, _ in species_counts.most_common(max_species)]


def agg_expanded_list(
    probs: np.ndarray,
    top_k: int,
    cid_to_spid: dict[int, int],
    max_species: int,
) -> list[int]:
    """Expanded-list aggregation — best method (F1=0.23).

    Build frequency-expanded list (species repeated by tile count), cap at
    max_species, then deduplicate. High-frequency species consume more slots,
    naturally filtering rare species that appeared in only 1-2 tiles.

    Example with 9 tiles, top-k=1, max_species=10:
        Expanded: [111]*8 + [222]*5 + [333]*3 + [444] + [555] + [666]
        After cap at 10: [111]*8 + [222]*2
        After dedup: [111, 222]
    """
    top_k_ids = np.argpartition(-probs, top_k, axis=1)[:, :top_k]
    species_counts: Counter = Counter()
    for tile_top_k in top_k_ids:
        for class_idx in tile_top_k:
            species_counts[cid_to_spid[int(class_idx)]] += 1

    # Build expanded list sorted by frequency, cap, then deduplicate
    expanded = []
    for sp, count in species_counts.most_common():
        expanded.extend([sp] * count)
    expanded = expanded[:max_species]

    seen = set()
    result = []
    for sp in expanded:
        if sp not in seen:
            seen.add(sp)
            result.append(sp)
    return result


def agg_max_prob(
    probs: np.ndarray,
    cid_to_spid: dict[int, int],
    min_score: float = 0.01,
    max_species: int | None = None,
) -> list[int]:
    """Max probability per species across tiles (organizers' approach).

    For each species, take the maximum probability across all tiles. Return
    species above min_score threshold, sorted by probability descending.
    """
    # Max probability per class across tiles: shape (num_classes,)
    max_probs = probs.max(axis=0)

    # Get indices above threshold, sorted by probability
    above_thresh = np.where(max_probs >= min_score)[0]
    sorted_indices = above_thresh[np.argsort(-max_probs[above_thresh])]

    if max_species is not None:
        sorted_indices = sorted_indices[:max_species]

    return [cid_to_spid[int(idx)] for idx in sorted_indices]


# ---- Quadrat aggregation ----


def extract_base_quadrat_id(quadrat_id: str) -> str:
    """Strip date suffix to get the base quadrat identifier.

    Groups images from the same vegetation plot taken on different dates.

    Examples:
        CBN-PdlC-A1-20130807  -> CBN-PdlC-A1
        RNNB-12-3-20210615    -> RNNB-12-3
        OPTMix-42-20200801    -> OPTMix-42
    """
    patterns = [
        (r"^(CBN-.*?-.*?)-\d{8}$", 1),
        (r"^(GUARDEN-.*?-.*?)-.*$", 1),
        (r"^(LISAH-.*?)-\d{8}$", 1),
        (r"^(OPTMix-\d+)-.*$", 1),
        (r"^(RNNB-\d+-\d+)-\d{8}$", 1),
    ]
    for pattern, group_idx in patterns:
        match = re.match(pattern, quadrat_id)
        if match:
            return match.group(group_idx)
    return quadrat_id


def quadrat_aggregate(
    df: pd.DataFrame, top_k: int | None = None
) -> pd.DataFrame:
    """Group images by base quadrat ID, union species across all images.

    Each image in a quadrat group receives the same aggregated species list.
    """
    df = df.copy()
    df["base_quadrat_id"] = df["quadrat_id"].apply(extract_base_quadrat_id)

    def _union_sorted(group_species: pd.Series) -> list[int]:
        counts = Counter(sp for sp_list in group_species for sp in sp_list)
        return [sp for sp, _ in counts.most_common(top_k)]

    agg = (
        df.groupby("base_quadrat_id")["species_ids"]
        .apply(_union_sorted)
        .reset_index()
        .rename(columns={"species_ids": "species_ids_agg"})
    )

    df = df.merge(agg, on="base_quadrat_id", how="left")
    df["species_ids"] = df["species_ids_agg"]
    return df[["quadrat_id", "species_ids"]]


# ---- Main ----


def parse_args():
    parser = argparse.ArgumentParser(
        description="PlantCLEF 2026 — Stage 2: Aggregate Probabilities → Submission CSV"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to parquet file from inference.py",
    )
    parser.add_argument(
        "--class-mapping-path",
        type=str,
        default=DEFAULT_CLASS_MAPPING,
        help="Path to class_mapping.txt",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="expanded-list",
        choices=["union", "expanded-list", "max-prob"],
        help="Aggregation method (default: expanded-list)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K predictions per tile (for union/expanded-list methods)",
    )
    parser.add_argument(
        "--max-species",
        type=int,
        default=None,
        help="Max species cap (required for expanded-list, optional for union/max-prob)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.01,
        help="Min probability threshold for max-prob method",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Output directory for submission CSVs",
    )
    parser.add_argument(
        "--quadrat-agg",
        action="store_true",
        help="Apply quadrat-level aggregation as post-processing",
    )
    parser.add_argument(
        "--quadrat-top-k",
        type=int,
        default=None,
        help="Cap species per quadrat group (only with --quadrat-agg)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.method == "expanded-list" and args.max_species is None:
        print("Error: --max-species is required for expanded-list method")
        raise SystemExit(1)

    print("=" * 60)
    print("PlantCLEF 2026 — Stage 2: Aggregation")
    print("=" * 60)
    print(f"Input:        {args.input}")
    print(f"Method:       {args.method}")
    if args.method in ("union", "expanded-list"):
        print(f"Top-K:        {args.top_k}")
    print(f"Max species:  {args.max_species or 'no cap'}")
    if args.method == "max-prob":
        print(f"Min score:    {args.min_score}")
    print(f"Quadrat agg:  {args.quadrat_agg}")
    if args.quadrat_agg:
        print(f"Quadrat top-k: {args.quadrat_top_k or 'no cap'}")
    print(f"Output dir:   {args.output_dir}")
    print()

    # ---- Load data ----
    print("Loading parquet...")
    df = load_parquet(args.input)
    grid_size = df["grid_size"].iloc[0]
    grid_label = f"grid_{grid_size}x{grid_size}" if grid_size > 0 else "no_grid"
    print(f"  Images: {len(df)}, grid: {grid_label}, tiles/image: {df['tiles_per_image'].iloc[0]}")

    print("Loading class mapping...")
    cid_to_spid = load_class_mapping(args.class_mapping_path)
    print(f"  Classes: {len(cid_to_spid)}")
    print()

    # ---- Aggregate per image ----
    print(f"Aggregating ({args.method})...")
    results = []
    for _, row in df.iterrows():
        probs = row["probs"]  # (T, num_classes) float32
        name = row["image_name"]

        if args.method == "union":
            species_ids = agg_union(probs, args.top_k, cid_to_spid, args.max_species)
        elif args.method == "expanded-list":
            species_ids = agg_expanded_list(probs, args.top_k, cid_to_spid, args.max_species)
        elif args.method == "max-prob":
            species_ids = agg_max_prob(probs, cid_to_spid, args.min_score, args.max_species)

        results.append({"quadrat_id": name, "species_ids": species_ids})

    # ---- Quadrat aggregation (optional post-processing) ----
    if args.quadrat_agg:
        print("Applying quadrat-level aggregation...")
        results_df = pd.DataFrame(results)
        results_df = quadrat_aggregate(results_df, top_k=args.quadrat_top_k)
        results = results_df.to_dict("records")

    # ---- Report ----
    avg_species = sum(len(r["species_ids"]) for r in results) / max(len(results), 1)
    print(f"\n  Avg species per image: {avg_species:.1f}")

    # ---- Build output filename ----
    parts = [grid_label, args.method.replace("-", "_")]
    if args.method in ("union", "expanded-list"):
        parts.append(f"topk{args.top_k}")
    if args.max_species:
        parts.append(f"max{args.max_species}")
    if args.method == "max-prob":
        parts.append(f"minscore{args.min_score}")
    if args.quadrat_agg:
        qk = f"_topk{args.quadrat_top_k}" if args.quadrat_top_k else ""
        parts.append(f"qagg{qk}")
    output_file = args.output_dir / f"{'_'.join(parts)}.csv"

    # ---- Save submission CSV ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(
        [
            {
                "quadrat_id": r["quadrat_id"],
                "species_ids": "[" + ", ".join(str(s) for s in r["species_ids"]) + "]",
            }
            for r in results
        ]
    )
    out_df.to_csv(output_file, sep=",", index=False, quoting=csv.QUOTE_ALL)

    print(f"\nSubmission saved: {output_file}")
    print(f"  Rows: {len(out_df)}")
    print(f"  Avg species/image: {avg_species:.1f}")


if __name__ == "__main__":
    main()
