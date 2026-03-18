"""Build FAISS index from pre-extracted PlantCLEF training embeddings.

Loads CLS embeddings from existing parquets, filters by geo species mask,
L2-normalizes, and builds a FAISS IndexFlatIP for cosine similarity search.

Input parquets at:
  /storage/.../embeddings/train/sample_id={0..19}/part-*.snappy.parquet
  Schema: image_name (str), species_id (int32), cls_embedding (float32 768d)

Usage:
  python build_faiss_index.py \
    --train-emb-dir /storage/.../embeddings/train \
    --species-map /path/to/class_to_species_id.json \
    --geo-mask-npy /path/to/sw_europe_mask_bool.npy \
    --out-dir /path/to/faiss_index_output
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import faiss
import numpy as np
import pandas as pd


def load_species_map_inverse(species_map_path: str) -> dict[int, int]:
    """Load class_to_species_id.json and invert to species_id -> class_index."""
    with open(species_map_path) as f:
        class_to_species = json.load(f)
    return {int(sid): idx for idx, sid in enumerate(class_to_species)}


def load_training_embeddings(
    train_emb_dir: str,
    species_to_class: dict[int, int],
    allowed_classes: set[int],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load all training embedding parquets, filter by geo mask.

    Returns:
        embeddings: [N_filtered, 768] float32
        labels: [N_filtered] int32 class indices
        n_total: total rows before filtering
        n_skipped_species: rows with species_id not in mapping
    """
    pattern = os.path.join(train_emb_dir, "sample_id=*", "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found matching {pattern}"
        )
    print(f"Found {len(files)} parquet files across partitions")

    all_embeddings = []
    all_labels = []
    n_total = 0
    n_skipped_species = 0

    for i, fp in enumerate(files):
        df = pd.read_parquet(fp)
        n_total += len(df)

        for _, row in df.iterrows():
            sid = int(row["species_id"])
            if sid not in species_to_class:
                n_skipped_species += 1
                continue
            class_idx = species_to_class[sid]
            if class_idx not in allowed_classes:
                continue

            emb = row["cls_embedding"]
            if isinstance(emb, np.ndarray):
                all_embeddings.append(emb.astype(np.float32))
            else:
                all_embeddings.append(np.array(emb, dtype=np.float32))
            all_labels.append(class_idx)

        if (i + 1) % 50 == 0 or i == len(files) - 1:
            print(f"  Processed {i + 1}/{len(files)} files, "
                  f"{len(all_embeddings)} vectors kept so far")

    embeddings = np.stack(all_embeddings, axis=0)  # [N, 768]
    labels = np.array(all_labels, dtype=np.int32)  # [N]
    return embeddings, labels, n_total, n_skipped_species


def subsample_balanced(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_per_class: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample so each class has at most max_per_class images.

    Rare classes keep all images. Common classes are randomly subsampled.
    """
    rng = np.random.RandomState(seed)
    unique_classes, counts = np.unique(labels, return_counts=True)

    keep_indices = []
    n_subsampled = 0
    for cls, cnt in zip(unique_classes, counts):
        cls_indices = np.where(labels == cls)[0]
        if cnt > max_per_class:
            chosen = rng.choice(cls_indices, max_per_class, replace=False)
            keep_indices.append(chosen)
            n_subsampled += 1
        else:
            keep_indices.append(cls_indices)

    keep_indices = np.concatenate(keep_indices)
    keep_indices.sort()

    print(f"  Balanced subsampling: max_per_class={max_per_class}")
    print(f"  {n_subsampled} classes subsampled, "
          f"{len(keep_indices)} vectors kept (was {len(labels)})")

    return embeddings[keep_indices], labels[keep_indices]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FAISS index from training embeddings")
    ap.add_argument("--train-emb-dir", required=True,
                    help="Root dir with sample_id=* partitions")
    ap.add_argument("--species-map", required=True,
                    help="class_to_species_id.json")
    ap.add_argument("--geo-mask-npy", required=True,
                    help="sw_europe_mask_bool.npy")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for index + labels")
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="Cap each species to N images (0=disabled, keep all)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load species mapping (inverted)
    print("Loading species mapping...")
    species_to_class = load_species_map_inverse(args.species_map)
    print(f"  {len(species_to_class)} species in mapping")

    # 2. Load geo mask
    print("Loading geo mask...")
    geo_mask = np.load(args.geo_mask_npy)
    if geo_mask.dtype != np.bool_:
        geo_mask = geo_mask.astype(np.bool_)
    allowed_classes = set(int(i) for i in np.where(geo_mask)[0])
    print(f"  {len(allowed_classes)} of {len(geo_mask)} classes pass geo filter")

    # 3. Load and filter training embeddings
    print("\nLoading training embeddings...")
    t0 = time.time()
    embeddings, labels, n_total, n_skipped = load_training_embeddings(
        args.train_emb_dir, species_to_class, allowed_classes
    )
    elapsed = time.time() - t0
    print(f"\nLoading complete in {elapsed:.1f}s:")
    print(f"  Total rows: {n_total}")
    print(f"  Skipped (unmapped species): {n_skipped}")
    print(f"  Skipped (geo filter): {n_total - n_skipped - len(labels)}")
    print(f"  Kept: {len(labels)} vectors")
    print(f"  Unique species in index: {len(np.unique(labels))}")
    print(f"  Embedding shape: {embeddings.shape}")

    # 4. Save class frequency counts (before any subsampling)
    class_counts = np.zeros(7806, dtype=np.int64)
    for lbl in labels:
        class_counts[lbl] += 1
    counts_path = os.path.join(args.out_dir, "class_counts_geo.npy")
    np.save(counts_path, class_counts)
    print(f"\nClass frequency stats (before subsampling):")
    nonzero_counts = class_counts[class_counts > 0]
    print(f"  Species with images: {len(nonzero_counts)}")
    print(f"  Min/median/max images per species: "
          f"{nonzero_counts.min()}/{int(np.median(nonzero_counts))}/{nonzero_counts.max()}")

    # 5. Optional balanced subsampling
    if args.max_per_class > 0:
        embeddings, labels = subsample_balanced(
            embeddings, labels, args.max_per_class
        )

    # 6. L2-normalize
    print("\nL2-normalizing embeddings...")
    faiss.normalize_L2(embeddings)

    # 7. Build FAISS index
    print("Building FAISS IndexFlatIP (768d)...")
    t0 = time.time()
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"  Index built in {time.time() - t0:.1f}s, "
          f"ntotal={index.ntotal}")

    # 8. Sanity check: self-retrieval
    print("\nSanity check: self-retrieval on first 5 vectors...")
    test_embs = embeddings[:5].copy()
    sims, idxs = index.search(test_embs, 1)
    for i in range(5):
        retrieved_label = labels[idxs[i, 0]]
        actual_label = labels[i]
        match = "OK" if retrieved_label == actual_label else "MISMATCH"
        print(f"  Query {i}: retrieved idx={idxs[i,0]}, "
              f"sim={sims[i,0]:.4f}, "
              f"label={retrieved_label} vs {actual_label} [{match}]")

    # 9. Save
    suffix = f"_bal{args.max_per_class}" if args.max_per_class > 0 else ""
    index_path = os.path.join(args.out_dir, f"faiss_index_geo{suffix}.bin")
    labels_path = os.path.join(args.out_dir, f"faiss_labels_geo{suffix}.npy")
    stats_path = os.path.join(args.out_dir, f"index_stats{suffix}.json")

    print(f"\nSaving index to {index_path}...")
    faiss.write_index(index, index_path)

    print(f"Saving labels to {labels_path}...")
    np.save(labels_path, labels)

    stats = {
        "n_vectors": int(index.ntotal),
        "n_species": int(len(np.unique(labels))),
        "n_total_before_filter": n_total,
        "n_skipped_unmapped": n_skipped,
        "embedding_dim": int(embeddings.shape[1]),
        "index_type": "IndexFlatIP",
        "geo_mask": args.geo_mask_npy,
        "max_per_class": args.max_per_class,
        "class_counts_path": counts_path,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    index_size_mb = os.path.getsize(index_path) / (1024 * 1024)
    print(f"\nDone! Index size: {index_size_mb:.1f} MB")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
