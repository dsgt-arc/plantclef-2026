#!/usr/bin/env python

import os
import argparse
import numpy as np
import pyarrow.parquet as pq


def topk_from_dense_f16(blob, num_classes: int):
    p = np.frombuffer(blob, dtype=np.float16, count=num_classes).astype(np.float32)
    nz = np.nonzero(p > 0)[0]
    if nz.size == 0:
        return nz, p
    return nz, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step2-parquet", required=True)
    ap.add_argument("--out-npz", required=True)

    ap.add_argument("--num-classes", type=int, default=7806)

    ap.add_argument("--cap-k", type=int, default=8)
    ap.add_argument("--use-relative", action="store_true")
    ap.add_argument("--relative-alpha", type=float, default=0.85)
    ap.add_argument("--abs-threshold", type=float, default=0.0)

    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--topn", type=int, default=50)

    args = ap.parse_args()

    if args.cap_k <= 1:
        raise ValueError("--cap-k must be >= 2 for co occurrence")
    if args.topn <= 0:
        raise ValueError("--topn must be >= 1")

    t = pq.read_table(args.step2_parquet, columns=["image_name", "teacher_image_probs_f16"]).to_pandas()

    counts = np.zeros((args.num_classes,), dtype=np.int32)
    pair_counts = {}

    used_images = 0
    skipped = 0
    avg_kept = []

    for _, r in t.iterrows():
        blob = r["teacher_image_probs_f16"]
        nz, p = topk_from_dense_f16(blob, args.num_classes)
        if nz.size == 0:
            skipped += 1
            continue

        vals = p[nz]
        order = np.argsort(vals)[::-1]
        nz = nz[order]
        vals = vals[order]

        if args.use_relative:
            thr = args.relative_alpha * float(vals[0])
        else:
            thr = float(args.abs_threshold)

        keep_mask = vals >= thr
        nz_keep = nz[keep_mask]
        vals_keep = vals[keep_mask]

        if nz_keep.size == 0:
            skipped += 1
            continue

        if nz_keep.size > args.cap_k:
            nz_keep = nz_keep[: args.cap_k]
            vals_keep = vals_keep[: args.cap_k]

        if nz_keep.size < 2:
            skipped += 1
            continue

        used_images += 1
        avg_kept.append(int(nz_keep.size))

        for i in nz_keep:
            counts[int(i)] += 1

        # undirected pair counts
        for a in range(nz_keep.size):
            ia = int(nz_keep[a])
            for b in range(a + 1, nz_keep.size):
                ib = int(nz_keep[b])
                if ia < ib:
                    key = (ia, ib)
                else:
                    key = (ib, ia)
                pair_counts[key] = pair_counts.get(key, 0) + 1

    print("step2 rows:", len(t))
    print("used images:", used_images)
    print("skipped:", skipped)
    if used_images > 0:
        print("avg kept labels:", float(np.mean(avg_kept)))

    # build adjacency by conditional probability P(j|i) = C(i,j) / count(i)
    neigh_i = []
    neigh_j = []
    neigh_w = []

    # accumulate candidates per i
    cand = [[] for _ in range(args.num_classes)]
    for (i, j), c in pair_counts.items():
        if c < args.min_count:
            continue
        ci = counts[i]
        cj = counts[j]
        if ci > 0:
            cand[i].append((j, c / float(ci)))
        if cj > 0:
            cand[j].append((i, c / float(cj)))

    for i in range(args.num_classes):
        if len(cand[i]) == 0:
            continue
        cand[i].sort(key=lambda x: x[1], reverse=True)
        for j, w in cand[i][: args.topn]:
            neigh_i.append(i)
            neigh_j.append(int(j))
            neigh_w.append(float(w))

    neigh_i = np.asarray(neigh_i, dtype=np.int32)
    neigh_j = np.asarray(neigh_j, dtype=np.int32)
    neigh_w = np.asarray(neigh_w, dtype=np.float32)
    counts = counts.astype(np.int32)

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        num_classes=np.int32(args.num_classes),
        neigh_i=neigh_i,
        neigh_j=neigh_j,
        neigh_w=neigh_w,
        class_counts=counts,
        cap_k=np.int32(args.cap_k),
        min_count=np.int32(args.min_count),
        topn=np.int32(args.topn),
        use_relative=np.int32(1 if args.use_relative else 0),
        relative_alpha=np.float32(args.relative_alpha),
        abs_threshold=np.float32(args.abs_threshold),
    )

    print("edges:", neigh_i.shape[0])
    print("wrote:", args.out_npz)


if __name__ == "__main__":
    main()


'''
python step3/build_cograph_from_step2_top8.py \
  --step2-parquet /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_10k/teacher_image_probs_step2_imageTopK64 \
  --out-npz /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/graphs/lucas_10k_cograph_top8_rel085_min20_top50.npz \
  --use-relative \
  --relative-alpha 0.85 \
  --cap-k 8 \
  --min-count 20 \
  --topn 50

'''