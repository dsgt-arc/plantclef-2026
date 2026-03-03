#!/usr/bin/env python3

import os
import csv
import math
import argparse
import numpy as np
import pyarrow.parquet as pq


def dense_f16_to_f32(blob: bytes, num_classes: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float16, count=num_classes).astype(np.float32)


def iter_teacher_blobs(step2_root: str):
    ds = pq.ParquetDataset(step2_root)
    for frag in ds.fragments:
        tbl = frag.to_table(columns=["teacher_image_probs_f16"])
        col = tbl["teacher_image_probs_f16"]
        for blob in col.to_pylist():
            yield blob


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--step2-parquet", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-csv", default="")

    ap.add_argument("--num-classes", type=int, default=7806)

    ap.add_argument("--cap-k", type=int, default=8)

    ap.add_argument(
        "--relative-alpha",
        type=float,
        default=0.80,
        help="Keep labels in Top K only if p >= alpha * p_max within that image",
    )

    ap.add_argument("--min-count", type=int, default=100)
    ap.add_argument("--topn", type=int, default=50)

    ap.add_argument(
        "--min-class-count",
        type=int,
        default=0,
        help="If > 0, only keep edges where both classes have count >= this",
    )

    ap.add_argument(
        "--min-labels",
        type=int,
        default=2,
        help="Minimum kept labels per image to contribute pairs",
    )

    args = ap.parse_args()

    if args.cap_k <= 1:
        raise ValueError("cap_k must be at least 2")
    if not (0.0 < args.relative_alpha <= 1.0):
        raise ValueError("relative_alpha must be in (0, 1]")
    if args.min_count < 1:
        raise ValueError("min_count must be at least 1")
    if args.topn <= 0:
        raise ValueError("topn must be at least 1")
    if args.min_labels < 2:
        raise ValueError("min_labels must be at least 2")

    counts = np.zeros((args.num_classes,), dtype=np.int32)
    pair_counts = {}

    used_images = 0
    skipped = 0
    images_with_pairs = 0

    print("Reading Step 2 parquet shards from:", args.step2_parquet)

    for blob in iter_teacher_blobs(args.step2_parquet):
        if blob is None:
            skipped += 1
            continue

        p = dense_f16_to_f32(blob, args.num_classes)

        if not np.any(p > 0):
            skipped += 1
            continue

        k = min(args.cap_k, args.num_classes)

        topk_idx = np.argpartition(p, -k)[-k:]
        topk_vals = p[topk_idx]

        vmax = float(np.max(topk_vals))
        if vmax <= 0.0:
            skipped += 1
            continue

        thr = args.relative_alpha * vmax

        keep_mask = topk_vals >= thr
        keep_idx = topk_idx[keep_mask]
        keep_vals = topk_vals[keep_mask]

        if keep_idx.size < args.min_labels:
            used_images += 1
            continue

        order = np.argsort(keep_vals)[::-1]
        nz_keep = keep_idx[order]

        used_images += 1
        images_with_pairs += 1

        for i in nz_keep:
            counts[int(i)] += 1

        for a in range(nz_keep.size):
            ia = int(nz_keep[a])
            for b in range(a + 1, nz_keep.size):
                ib = int(nz_keep[b])
                key = (ia, ib) if ia < ib else (ib, ia)
                pair_counts[key] = pair_counts.get(key, 0) + 1

        if used_images % 50000 == 0:
            print("Processed images:", used_images)

    print("Processed images:", used_images)
    print("Images that contributed pairs:", images_with_pairs)
    print("Skipped:", skipped)

    if used_images == 0:
        raise RuntimeError("No images processed")

    N = float(images_with_pairs)
    if N <= 0:
        raise RuntimeError("No images had enough confident labels to form pairs")

    eps = 1e-12
    cand = [[] for _ in range(args.num_classes)]
    csv_rows = []

    kept_pairs = 0
    dropped_min_count = 0
    dropped_min_class = 0
    dropped_pmi = 0

    for (i, j), c in pair_counts.items():
        if c < args.min_count:
            dropped_min_count += 1
            continue

        ci = int(counts[i])
        cj = int(counts[j])
        if ci == 0 or cj == 0:
            continue

        if args.min_class_count > 0 and (ci < args.min_class_count or cj < args.min_class_count):
            dropped_min_class += 1
            continue

        ratio = (float(c) * N) / (float(ci) * float(cj) + eps)
        pmi = math.log(ratio + eps)

        if pmi <= 0.0:
            dropped_pmi += 1
            continue

        kept_pairs += 1

        cand[i].append((j, pmi))
        cand[j].append((i, pmi))

        if args.out_csv:
            csv_rows.append((i, j, int(c), ci, cj, float(pmi)))

    neigh_i = []
    neigh_j = []
    neigh_w = []

    for i in range(args.num_classes):
        if not cand[i]:
            continue
        cand[i].sort(key=lambda x: x[1], reverse=True)
        for j, w in cand[i][: args.topn]:
            neigh_i.append(i)
            neigh_j.append(int(j))
            neigh_w.append(float(w))

    neigh_i = np.asarray(neigh_i, dtype=np.int32)
    neigh_j = np.asarray(neigh_j, dtype=np.int32)
    neigh_w = np.asarray(neigh_w, dtype=np.float32)

    out_dir = os.path.dirname(args.out_npz)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        args.out_npz,
        num_classes=np.int32(args.num_classes),
        neigh_i=neigh_i,
        neigh_j=neigh_j,
        neigh_w=neigh_w,
        class_counts=counts.astype(np.int32),
        cap_k=np.int32(args.cap_k),
        min_count=np.int32(args.min_count),
        topn=np.int32(args.topn),
        min_class_count=np.int32(args.min_class_count),
        relative_alpha=np.float32(args.relative_alpha),
        min_labels=np.int32(args.min_labels),
        weight_type=np.bytes_(b"pmi"),
        used_images=np.int32(used_images),
        images_with_pairs=np.int32(images_with_pairs),
    )

    print("Pairs counted:", len(pair_counts))
    print("Pairs kept after filters:", kept_pairs)
    print("Dropped by min_count:", dropped_min_count)
    print("Dropped by min_class_count:", dropped_min_class)
    print("Dropped by pmi<=0:", dropped_pmi)
    print("Edges written:", neigh_i.shape[0])
    print("Wrote npz:", args.out_npz)

    if args.out_csv:
        csv_dir = os.path.dirname(args.out_csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        csv_rows.sort(key=lambda x: x[5], reverse=True)

        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class_i", "class_j", "pair_count", "count_i", "count_j", "pmi"])
            w.writerows(csv_rows)

        print("Wrote csv:", args.out_csv)


if __name__ == "__main__":
    main()


"""
Example:

python build_cograph_ecology_from_step2_top8_pmi.py \
  --step2-parquet /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k/step2_teacher_agg \
  --out-npz /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/graphs/lucas_212k_cograph_ecology_top8_alpha080_pmi_min100_top50.npz \
  --out-csv /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/graphs/lucas_212k_pairs_ecology_top8_alpha080_pmi_min100.csv \
  --cap-k 8 \
  --relative-alpha 0.60 \
  --min-count 100 \
  --topn 50
"""
