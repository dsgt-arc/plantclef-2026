"""Build a 512-d FAISS index from ArcFace-projected training embeddings.

Adapted from Exp-9/build_faiss_index.py. The only substantive differences:
  - Reads the 'emb512' column produced by project_embeddings.py (--mode train)
  - IndexFlatIP is built at 512 dim (verified from the loaded array)
  - L2-normalization is still applied (the head already normalizes, but this is
    cheap and defensive against any numerical drift)

Input parquets at:
  <train-emb-arcface-dir>/sample_id={0..19}/part-*.snappy.parquet
  Schema: image_name (str), species_id (int32), emb512 (float32 512d)

Usage:
  python build_faiss_index_512.py \
    --train-emb-dir /storage/.../exp9b_knn/embeddings_arcface/train \
    --species-map   /path/to/class_to_species_id.json \
    --geo-mask-npy  /path/to/sw_europe_mask_bool.npy \
    --out-dir       /path/to/faiss_index_output
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


NUM_CLASSES = 7806


def load_species_map_inverse(species_map_path: str) -> dict[int, int]:
    with open(species_map_path) as f:
        class_to_species = json.load(f)
    return {int(sid): idx for idx, sid in enumerate(class_to_species)}


def load_projected_embeddings(
    train_emb_dir: str,
    species_to_class: dict[int, int],
    allowed_classes: set[int],
    emb_col: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    pattern = os.path.join(train_emb_dir, "sample_id=*", "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matching {pattern}")
    print(f"[load] {len(files)} parquet files")

    chunks_emb: list[np.ndarray] = []
    chunks_lbl: list[np.ndarray] = []
    n_total = 0
    n_skipped_species = 0

    for i, fp in enumerate(files):
        df = pd.read_parquet(fp, columns=["species_id", emb_col])
        n_total += len(df)
        sids = df["species_id"].to_numpy(dtype=np.int64)

        keep_rows = np.zeros(len(df), dtype=bool)
        label_buf = np.empty(len(df), dtype=np.int32)
        for j, sid in enumerate(sids):
            sid_i = int(sid)
            if sid_i not in species_to_class:
                n_skipped_species += 1
                continue
            cls = species_to_class[sid_i]
            if cls not in allowed_classes:
                continue
            keep_rows[j] = True
            label_buf[j] = cls

        if not keep_rows.any():
            continue

        emb_series = df[emb_col].to_numpy()[keep_rows]
        emb_arr = np.stack(
            [np.asarray(e, dtype=np.float32) for e in emb_series], axis=0
        )
        chunks_emb.append(emb_arr)
        chunks_lbl.append(label_buf[keep_rows])

        if (i + 1) % 5 == 0 or i == len(files) - 1:
            kept = sum(len(c) for c in chunks_lbl)
            print(f"[load]   {i + 1}/{len(files)} shards, {kept} kept")

    embeddings = np.concatenate(chunks_emb, axis=0)
    labels = np.concatenate(chunks_lbl, axis=0)
    return embeddings, labels, n_total, n_skipped_species


def subsample_balanced(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_per_class: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    unique_classes, counts = np.unique(labels, return_counts=True)
    keep_indices: list[np.ndarray] = []
    n_sub = 0
    for cls, cnt in zip(unique_classes, counts):
        cls_idx = np.where(labels == cls)[0]
        if cnt > max_per_class:
            keep_indices.append(rng.choice(cls_idx, max_per_class, replace=False))
            n_sub += 1
        else:
            keep_indices.append(cls_idx)
    keep = np.concatenate(keep_indices)
    keep.sort()
    print(
        f"[subsample] max_per_class={max_per_class} "
        f"capped_classes={n_sub} kept={len(keep)} (was {len(labels)})"
    )
    return embeddings[keep], labels[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 512-d FAISS index from ArcFace embeddings")
    ap.add_argument("--train-emb-dir", required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--geo-mask-npy", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--emb-col", default="emb512")
    ap.add_argument("--max-per-class", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[init] loading species map")
    species_to_class = load_species_map_inverse(args.species_map)
    print(f"[init]   {len(species_to_class)} species")

    print("[init] loading geo mask")
    geo_mask = np.load(args.geo_mask_npy)
    if geo_mask.dtype != np.bool_:
        geo_mask = geo_mask.astype(np.bool_)
    allowed_classes = {int(i) for i in np.where(geo_mask)[0]}
    print(f"[init]   {len(allowed_classes)} / {len(geo_mask)} classes pass geo")

    print("[load] reading projected train embeddings")
    t0 = time.time()
    embeddings, labels, n_total, n_skipped = load_projected_embeddings(
        args.train_emb_dir, species_to_class, allowed_classes, args.emb_col
    )
    print(
        f"[load] done in {time.time() - t0:.1f}s  total={n_total} "
        f"skipped_unmapped={n_skipped} kept={len(labels)} "
        f"dim={embeddings.shape[1]}"
    )
    print(f"[load] unique species in index: {len(np.unique(labels))}")

    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for lbl in labels:
        class_counts[lbl] += 1
    counts_path = os.path.join(args.out_dir, "class_counts_arcface_geo.npy")
    np.save(counts_path, class_counts)
    nz = class_counts[class_counts > 0]
    print(
        f"[stats] species with images: {len(nz)} "
        f"min/median/max per species: {nz.min()}/{int(np.median(nz))}/{nz.max()}"
    )

    if args.max_per_class > 0:
        embeddings, labels = subsample_balanced(
            embeddings, labels, args.max_per_class
        )

    print("[index] L2-normalizing (defensive)")
    faiss.normalize_L2(embeddings)

    print(f"[index] building IndexFlatIP dim={embeddings.shape[1]}")
    t0 = time.time()
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"[index] built in {time.time() - t0:.1f}s ntotal={index.ntotal}")

    print("[check] self-retrieval sanity on first 5 vectors")
    test_embs = embeddings[:5].copy()
    sims, idxs = index.search(test_embs, 1)
    for i in range(5):
        retrieved = labels[idxs[i, 0]]
        actual = labels[i]
        flag = "OK" if retrieved == actual else "MISMATCH"
        print(
            f"  q={i} idx={idxs[i, 0]} sim={sims[i, 0]:.4f} "
            f"lbl={retrieved} vs {actual} [{flag}]"
        )

    suffix = f"_bal{args.max_per_class}" if args.max_per_class > 0 else ""
    index_path = os.path.join(args.out_dir, f"faiss_index_arcface_geo{suffix}.bin")
    labels_path = os.path.join(args.out_dir, f"faiss_labels_arcface_geo{suffix}.npy")
    stats_path = os.path.join(args.out_dir, f"index_stats_arcface{suffix}.json")

    print(f"[out] writing index to {index_path}")
    faiss.write_index(index, index_path)
    print(f"[out] writing labels to {labels_path}")
    np.save(labels_path, labels)

    stats = {
        "n_vectors": int(index.ntotal),
        "n_species": int(len(np.unique(labels))),
        "n_total_before_filter": int(n_total),
        "n_skipped_unmapped": int(n_skipped),
        "embedding_dim": int(embeddings.shape[1]),
        "index_type": "IndexFlatIP",
        "geo_mask": args.geo_mask_npy,
        "max_per_class": int(args.max_per_class),
        "class_counts_path": counts_path,
        "source_emb_col": args.emb_col,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    mb = os.path.getsize(index_path) / (1024 * 1024)
    print(f"[done] index size: {mb:.1f} MB")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
