"""Compute class frequency counts from existing FAISS labels array.

Quick script — no need to rebuild the index, just reads the labels.

Usage:
  python compute_class_counts.py \
    --faiss-labels /path/to/faiss_labels_geo.npy \
    --out-path /path/to/class_counts_geo.npy
"""
import argparse
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faiss-labels", required=True)
    ap.add_argument("--out-path", required=True)
    ap.add_argument("--num-classes", type=int, default=7806)
    args = ap.parse_args()

    labels = np.load(args.faiss_labels).astype(np.int32)
    counts = np.zeros(args.num_classes, dtype=np.int64)
    for lbl in labels:
        counts[lbl] += 1

    np.save(args.out_path, counts)

    nonzero = counts[counts > 0]
    print(f"Labels: {len(labels)} vectors")
    print(f"Species with images: {len(nonzero)}")
    print(f"Min/median/max per species: {nonzero.min()}/{int(np.median(nonzero))}/{nonzero.max()}")
    print(f"Saved to {args.out_path}")


if __name__ == "__main__":
    main()
