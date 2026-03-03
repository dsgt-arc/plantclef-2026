#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from Cross_Region_Transformer_v3 import CrossRegionTransformer


def iter_batches_from_sharded_root(data_root: str, pattern: str, batch_rows: int):
    files = sorted(glob.glob(os.path.join(data_root, pattern)))
    if not files:
        raise ValueError(f"No shard parquet files found under {data_root} with pattern {pattern}")

    for fp in files:
        d = ds.dataset(fp, format="parquet")
        scanner = d.scanner(
            columns=["image_name", "tile_embeddings_f16"],
            batch_size=batch_rows,
        )
        for rb in scanner.to_reader():
            yield rb


def iter_batches_from_single_parquet(parquet_path: str, batch_rows: int):
    d = ds.dataset(parquet_path, format="parquet")
    scanner = d.scanner(
        columns=["image_name", "tile_embeddings_f16"],
        batch_size=batch_rows,
    )
    for rb in scanner.to_reader():
        yield rb


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-parquet", type=str, default="")
    ap.add_argument("--data-root", type=str, default="")
    ap.add_argument("--pattern", type=str, default="shard_*/data.parquet")

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-parquet", required=True)

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--batch-rows", type=int, default=512)

    ap.add_argument("--num-tiles", type=int, default=16)
    ap.add_argument("--emb-dim", type=int, default=768)
    ap.add_argument("--num-classes", type=int, default=7806)

    # candidate pool size to consider before thresholding and capping
    ap.add_argument("--topk", type=int, default=64)

    # minimum probability threshold, or relative threshold based on top 1
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--use-relative", action="store_true")
    ap.add_argument("--relative-alpha", type=float, default=0.85)

    # final cap after threshold
    ap.add_argument("--cap-k", type=int, default=8)

    args = ap.parse_args()

    if not args.data_root and not args.data_parquet:
        raise ValueError("Provide --data-root for sharded data, or --data-parquet for a single file")

    if args.cap_k <= 0:
        raise ValueError("--cap-k must be >= 1")
    if args.topk < args.cap_k:
        raise ValueError("--topk must be >= --cap-k")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = CrossRegionTransformer(
        input_dim=args.emb_dim,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        num_classes=args.num_classes,
        metadata_dim=4,
        dropout=0.1,
        use_attention_pooling=True,
    ).to(device)

    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()

    out_image = []
    out_idx_blob = []
    out_prob_blob = []
    out_k = []

    total_images = 0
    skipped = 0
    total_kept = 0

    if args.data_root:
        it = iter_batches_from_sharded_root(args.data_root, args.pattern, batch_rows=args.batch_rows)
        print("input mode: sharded", "pattern:", args.pattern)
    else:
        it = iter_batches_from_single_parquet(args.data_parquet, batch_rows=args.batch_rows)
        print("input mode: single parquet")

    for rb in it:
        image_names = rb.column("image_name").to_pylist()
        tile_blobs = rb.column("tile_embeddings_f16").to_pylist()

        B = len(image_names)
        if B == 0:
            continue

        tiles_np = np.empty((B, args.num_tiles, args.emb_dim), dtype=np.float32)

        for i in range(B):
            tiles = np.frombuffer(tile_blobs[i], dtype=np.float16).astype(np.float32, copy=False)
            tiles_np[i] = tiles.reshape(args.num_tiles, args.emb_dim)

        tiles_t = torch.from_numpy(tiles_np).to(device, non_blocking=True)

        logits = model(tiles_t, metadata=None)
        probs = torch.sigmoid(logits)

        topk_probs, topk_idx = torch.topk(probs, k=args.topk, dim=1)

        for i in range(B):
            total_images += 1

            p = topk_probs[i]
            idx = topk_idx[i]

            if args.use_relative:
                thr = args.relative_alpha * float(p[0].item())
            else:
                thr = float(args.threshold)

            keep_mask = p >= thr
            if keep_mask.sum().item() == 0:
                skipped += 1
                continue

            idx_keep = idx[keep_mask]
            p_keep = p[keep_mask]

            if idx_keep.numel() > args.cap_k:
                idx_keep = idx_keep[: args.cap_k]
                p_keep = p_keep[: args.cap_k]

            idx_keep_np = idx_keep.to(torch.int32).cpu().numpy()
            p_keep_np = p_keep.to(torch.float16).cpu().numpy()

            out_image.append(image_names[i])
            out_idx_blob.append(idx_keep_np.tobytes())
            out_prob_blob.append(p_keep_np.tobytes())
            out_k.append(int(idx_keep_np.shape[0]))

            total_kept += int(idx_keep_np.shape[0])

    print("images total:", total_images)
    print("images skipped:", skipped)
    print("images kept:", len(out_image))
    print("avg kept per kept image:", (total_kept / max(1, len(out_image))))

    table = pa.Table.from_arrays(
        [
            pa.array(out_image, type=pa.string()),
            pa.array(out_k, type=pa.int16()),
            pa.array(out_idx_blob, type=pa.binary()),
            pa.array(out_prob_blob, type=pa.binary()),
        ],
        names=["image_name", "k", "pos_idx_i32", "pos_prob_f16"],
    )

    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)
    pq.write_table(table, args.out_parquet, compression="zstd")
    print("wrote:", args.out_parquet)


if __name__ == "__main__":
    main()


'''
Example:

python src/step3_v4/infer_step3b1_pseudolabels_top8y.py \
  --data-root /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k/student_ready_merged_input_step3a \
  --ckpt /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/checkpoints/lucas_212k_SSL_step3a_/student_best_20260223_0319.pt \
  --out-parquet /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k_SSL/step3b_pseudolabels/top8_rel085.parquet \
  --batch-rows 512 \
  --topk 64 \
  --use-relative \
  --relative-alpha 0.85 \
  --cap-k 8
'''