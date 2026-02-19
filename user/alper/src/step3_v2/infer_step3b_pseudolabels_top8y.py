#!/usr/bin/env python

import os
import argparse
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader

from Cross_Region_Transformer_v3 import CrossRegionTransformer


class StudentReadyParquet(Dataset):
    def __init__(
        self,
        parquet_path: str,
        num_tiles: int = 16,
        emb_dim: int = 768,
        num_classes: int = 7806,
    ):
        self.parquet_path = parquet_path
        self.num_tiles = num_tiles
        self.emb_dim = emb_dim
        self.num_classes = num_classes

        table = pq.read_table(
            parquet_path, columns=["image_name", "tile_embeddings_f16", "metadata_tensor_f32"]
        )
        self.df = table.to_pandas()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        image_name = r["image_name"]

        tiles = np.frombuffer(r["tile_embeddings_f16"], dtype=np.float16).astype(np.float32)
        tiles = tiles.reshape(self.num_tiles, self.emb_dim)

        meta = np.asarray(r["metadata_tensor_f32"], dtype=np.float32).copy()

        return image_name, torch.from_numpy(tiles), torch.from_numpy(meta)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-parquet", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-parquet", required=True)

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)

    # candidate pool size to consider before thresholding/capping
    ap.add_argument("--topk", type=int, default=64)

    # minimum probability threshold
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--use-relative", action="store_true")
    ap.add_argument("--relative-alpha", type=float, default=0.85)

    # final cap after threshold
    ap.add_argument("--cap-k", type=int, default=8)

    args = ap.parse_args()

    if args.cap_k <= 0:
        raise ValueError("--cap-k must be >= 1")
    if args.topk < args.cap_k:
        raise ValueError("--topk must be >= --cap-k")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = StudentReadyParquet(args.data_parquet)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = CrossRegionTransformer(
        input_dim=768,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        num_classes=7806,
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

    for batch in loader:
        image_names, tiles, meta = batch
        tiles = tiles.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)

        logits = model(tiles, metadata=meta)
        probs = torch.sigmoid(logits)  # [B, C]

        # get candidate pool
        topk_probs, topk_idx = torch.topk(probs, k=args.topk, dim=1)  # [B, topk]

        for i in range(topk_idx.shape[0]):
            total_images += 1

            p = topk_probs[i]  # [topk], descending
            idx = topk_idx[i]  # [topk]

            # minimum threshold
            if args.use_relative:
                thr = args.relative_alpha * float(p[0].item())
            else:
                thr = float(args.threshold)

            keep_mask = p >= thr
            if keep_mask.sum().item() == 0:
                skipped += 1
                continue

            # keep only those above threshold
            idx_keep = idx[keep_mask]
            p_keep = p[keep_mask]

            # cap to top cap-k among those remaining
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
python step3_v2/infer_step3b_pseudolabels_top8.py \
  --data-parquet /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k/student_ready_merged_input_step3a \
  --ckpt /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/checkpoints/lucas_212k_step3a_distill/student_best_20260218_2210.pt \
  --out-parquet /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k/step3b_pseudolabels/top8_rel085.parquet \
  --batch-size 256 \
  --num-workers 6 \
  --topk 64 \
  --use-relative \
  --relative-alpha 0.85 \
  --cap-k 8

'''