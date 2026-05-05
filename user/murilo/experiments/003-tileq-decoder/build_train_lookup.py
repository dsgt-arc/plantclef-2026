"""Build per-class-idx CLS lookup + species sampling weights from train embeddings.

Loads the team-shared ViT-B DINOv2 train CLS embeddings from
``shared/embeddings/train/sample_id={0..19}/`` (1,408,033 rows × 768-d float32)
and persists two artefacts consumed by the TileQ-Decoder samplers (spec §3.2,
§3.3):

  artifacts/train_cls_by_class_idx.npz
    embeddings          : float32 (N, 768)         concatenated, sorted by class_idx
    offsets             : int64   (n_classes + 1,) embeddings[offsets[c]:offsets[c+1]]
                                                   are the CLSs for class_idx c
    class_to_species_id : int32   (n_classes,)     class_idx -> Pl@ntNet species_id

  artifacts/species_weights.npy
    float32 (n_classes,) — w_s ∝ 1/sqrt(count[s]), normalised so sum == 1.

The output is *not* memory-mapped (npz is a zip); samplers should call
``np.load(...)`` once at training start (~5 sec, ~4.3 GB resident).

Run from the experiment directory:
  python build_train_lookup.py
or via:
  bash sbatch/build_train_lookup.sh
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds


def load_class_mapping(class_mapping_path: Path) -> np.ndarray:
    """class_mapping.txt: line i contains the species_id for class_idx i.

    Returns ``class_to_species_id``: int32 array shape (n_classes,).
    """
    species_ids = [
        int(line.strip())
        for line in class_mapping_path.read_text().splitlines()
        if line.strip()
    ]
    arr = np.asarray(species_ids, dtype=np.int32)
    if len(arr) != len(set(arr.tolist())):
        raise ValueError(
            f"duplicate species_id in {class_mapping_path}: "
            f"{len(arr)} lines but only {len(set(arr.tolist()))} unique"
        )
    return arr


def species_ids_to_class_idxs(
    sids: np.ndarray, class_to_species_id: np.ndarray
) -> np.ndarray:
    """Vectorised species_id -> class_idx lookup via sorted-search.

    Raises ValueError if any species_id is not present in class_to_species_id.
    """
    n_classes = len(class_to_species_id)
    sort_perm = np.argsort(class_to_species_id)
    sids_sorted = class_to_species_id[sort_perm]
    pos = np.searchsorted(sids_sorted, sids)

    if (pos >= n_classes).any():
        bad = sorted(set(sids[pos >= n_classes].tolist()))[:10]
        raise ValueError(
            f"species_ids exceed mapping range (first 10): {bad}"
        )
    matched = sids_sorted[pos]
    if not np.array_equal(matched, sids):
        bad_mask = matched != sids
        bad = sorted(set(sids[bad_mask].tolist()))[:10]
        raise ValueError(f"species_ids not in class_mapping (first 10): {bad}")

    return sort_perm[pos].astype(np.int32)


def build_lookup_from_table(
    table: pa.Table,
    class_to_species_id: np.ndarray,
    embedding_dim: int = 768,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (species_id, cls_embedding) table to class-sorted dense arrays.

    Returns
    -------
    embeddings : float32 (N, 768) sorted by class_idx (stable within-class order)
    offsets    : int64   (n_classes + 1,) such that
                 embeddings[offsets[c]:offsets[c+1]] holds class c's CLSs
    counts     : int64   (n_classes,) rows per class
    """
    n_classes = len(class_to_species_id)
    n_rows = table.num_rows

    sids = table.column("species_id").to_numpy()
    class_idxs = species_ids_to_class_idxs(sids, class_to_species_id)

    list_arr = table.column("cls_embedding")
    if isinstance(list_arr, pa.ChunkedArray):
        list_arr = list_arr.combine_chunks()
    flat = list_arr.values.to_numpy(zero_copy_only=False)
    if flat.size != n_rows * embedding_dim:
        raise ValueError(
            f"embedding buffer size {flat.size} != n_rows ({n_rows}) * "
            f"embedding_dim ({embedding_dim}); some rows are not {embedding_dim}-d."
        )
    if flat.dtype != np.float32:
        flat = flat.astype(np.float32, copy=False)
    embs = flat.reshape(n_rows, embedding_dim)

    order = np.argsort(class_idxs, kind="stable")
    sorted_embs = np.ascontiguousarray(embs[order])
    sorted_class_idxs = class_idxs[order]

    counts = np.bincount(sorted_class_idxs, minlength=n_classes).astype(np.int64)
    offsets = np.zeros(n_classes + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    if offsets[-1] != n_rows:
        raise AssertionError(f"offsets[-1]={offsets[-1]} but n_rows={n_rows}")
    return sorted_embs, offsets, counts


def compute_species_weights(counts: np.ndarray) -> np.ndarray:
    """w_s ∝ 1 / sqrt(count[s]); normalised so sum == 1.

    Raises ValueError if any class has zero rows (would divide by zero).
    """
    if (counts <= 0).any():
        zero = np.where(counts == 0)[0]
        raise ValueError(
            f"{len(zero)} classes have zero training rows (first 10): {zero[:10].tolist()}"
        )
    raw = 1.0 / np.sqrt(counts.astype(np.float64))
    return (raw / raw.sum()).astype(np.float32)


def main(
    train_dir: Path,
    class_mapping_path: Path,
    output_dir: Path,
    expected_n_rows: int | None = 1_408_033,
    embedding_dim: int = 768,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading class mapping: {class_mapping_path}")
    class_to_species_id = load_class_mapping(class_mapping_path)
    n_classes = len(class_to_species_id)
    print(
        f"      n_classes={n_classes}, "
        f"species_id range=[{class_to_species_id.min()}, {class_to_species_id.max()}]"
    )

    print(f"[2/4] Reading train embeddings: {train_dir}")
    t0 = time.time()
    dataset = ds.dataset(train_dir, partitioning="hive", format="parquet")
    table = dataset.to_table(columns=["species_id", "cls_embedding"])
    elapsed = time.time() - t0
    print(f"      loaded {table.num_rows:,} rows in {elapsed:.1f}s")
    if expected_n_rows is not None and table.num_rows != expected_n_rows:
        raise RuntimeError(
            f"row count {table.num_rows} != expected {expected_n_rows}"
        )

    print("[3/4] Building per-class-idx lookup (sort + offsets)...")
    t0 = time.time()
    embeddings, offsets, counts = build_lookup_from_table(
        table, class_to_species_id, embedding_dim
    )
    del table
    elapsed = time.time() - t0
    print(
        f"      embeddings={embeddings.shape} {embeddings.dtype}, "
        f"offsets={offsets.shape}, "
        f"counts: min={counts.min()} median={int(np.median(counts))} "
        f"max={counts.max()} ({elapsed:.1f}s)"
    )
    if (counts == 0).any():
        zero = np.where(counts == 0)[0]
        raise RuntimeError(
            f"{len(zero)} classes have zero rows (first 10): {zero[:10].tolist()}"
        )

    print("[4/4] Computing species weights w_s ∝ 1/sqrt(count[s])...")
    weights = compute_species_weights(counts)
    print(
        f"      weights: min={weights.min():.3e} max={weights.max():.3e} "
        f"sum={weights.sum():.6f}"
    )

    lookup_path = output_dir / "train_cls_by_class_idx.npz"
    print(f"      writing {lookup_path}")
    t0 = time.time()
    np.savez(
        lookup_path,
        embeddings=embeddings,
        offsets=offsets,
        class_to_species_id=class_to_species_id,
    )
    size_gb = lookup_path.stat().st_size / 1e9
    print(f"      wrote {size_gb:.2f} GB in {time.time() - t0:.1f}s")

    weights_path = output_dir / "species_weights.npy"
    print(f"      writing {weights_path}")
    np.save(weights_path, weights)

    print("Done.")


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-dir",
        type=Path,
        default=Path(
            "/storage/project/ps-clef2026_plantclef-0/shared/embeddings/train"
        ),
    )
    p.add_argument(
        "--class-mapping",
        type=Path,
        default=Path(
            "/storage/project/ps-clef2026_plantclef-0/shared/data/raw/"
            "class_id_to_species_mapping/class_mapping.txt"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts",
    )
    p.add_argument(
        "--expected-n-rows",
        type=int,
        default=1_408_033,
        help="Assert exact total row count (set 0 to skip).",
    )
    args = p.parse_args()
    main(
        train_dir=args.train_dir,
        class_mapping_path=args.class_mapping,
        output_dir=args.output_dir,
        expected_n_rows=args.expected_n_rows or None,
    )


if __name__ == "__main__":
    cli()
