#!/usr/bin/env python3
"""Quadrat-level aggregation for PlantCLEF 2026 submissions.

Groups images by base quadrat ID (stripping date suffixes) and aggregates
species predictions across all images of the same quadrat. Each image in
a quadrat group receives the same aggregated species list.

This replicates the 2025 PlantCLEF approach where multiple photos of the
same vegetation plot (taken on different dates) are combined.

Usage:
    # Aggregate with no species cap
    python quadrat_aggregate.py artifacts/submission_grid_4x4_union_topk5.csv

    # Aggregate with top-k species cap per quadrat
    python quadrat_aggregate.py artifacts/submission_grid_4x4_union_topk5.csv --top-k 10

    # Process all CSVs in artifacts/
    python quadrat_aggregate.py artifacts/submission_grid_4x4_union_topk5.csv \
        artifacts/submission_grid_4x4_union_topk5_max20.csv --top-k 10
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def extract_base_quadrat_id(quadrat_id: str) -> str:
    """Strip date suffix to get the base quadrat identifier.

    Groups images from the same vegetation plot taken on different dates.

    Examples:
        CBN-PdlC-A1-20130807  -> CBN-PdlC-A1
        RNNB-12-3-20210615    -> RNNB-12-3
        OPTMix-42-20200801    -> OPTMix-42
        2024-CEV3-20240602    -> 2024-CEV3-20240602  (no match, returned as-is)
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


def parse_species_ids(species_str: str) -> list[int]:
    """Parse species IDs from CSV string format '[id1, id2, ...]'."""
    species_str = species_str.strip().strip("[]")
    if not species_str:
        return []
    return [int(s.strip()) for s in species_str.split(",")]


def format_species_ids(species_ids: list[int]) -> str:
    """Format species IDs to CSV string format '[id1, id2, ...]'."""
    return "[" + ", ".join(str(s) for s in species_ids) + "]"


def quadrat_aggregate(
    df: pd.DataFrame, top_k: int | None = None
) -> pd.DataFrame:
    """Aggregate species predictions at the quadrat level.

    Groups images by base quadrat ID, unions all species across images,
    sorts by frequency, and assigns the same list back to every image
    in the group.

    Args:
        df: DataFrame with columns 'quadrat_id' and 'species_ids' (as lists).
        top_k: Optional cap on species per quadrat.

    Returns:
        DataFrame with aggregated species_ids, same row count as input.
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


def process_file(input_path: Path, output_dir: Path, top_k: int | None):
    """Read a submission CSV, apply quadrat aggregation, and write output."""
    df = pd.read_csv(input_path)
    df["species_ids"] = df["species_ids"].apply(parse_species_ids)

    num_images = len(df)
    num_quadrats = df["quadrat_id"].apply(extract_base_quadrat_id).nunique()
    avg_species_before = df["species_ids"].apply(len).mean()

    df_agg = quadrat_aggregate(df, top_k=top_k)

    avg_species_after = df_agg["species_ids"].apply(len).mean()

    # Format species_ids back to string
    df_agg["species_ids"] = df_agg["species_ids"].apply(format_species_ids)

    # Build output filename
    stem = input_path.stem
    topk_label = f"_topk{top_k}" if top_k else ""
    output_file = output_dir / f"agg{topk_label}_{stem}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_agg.to_csv(output_file, sep=",", index=False, quoting=csv.QUOTE_ALL)

    print(f"  Input:          {input_path.name}")
    print(f"  Images:         {num_images}")
    print(f"  Base quadrats:  {num_quadrats}")
    print(f"  Avg species:    {avg_species_before:.1f} -> {avg_species_after:.1f}")
    print(f"  Output:         {output_file}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Quadrat-level aggregation for PlantCLEF submissions"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input submission CSV file(s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Cap species per quadrat (default: no cap)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Output directory (default: artifacts/)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Quadrat-Level Aggregation")
    print("=" * 60)
    print(f"Top-K cap: {args.top_k or 'no cap'}")
    print()

    for input_path in args.inputs:
        process_file(input_path, args.output_dir, args.top_k)


if __name__ == "__main__":
    main()
