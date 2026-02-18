#!/usr/bin/env python

import os
import time
import glob
import bisect
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pyarrow as pa
import pyarrow.parquet as pq

from Cross_Region_Transformer_v3 import CrossRegionTransformer  # already in your repo


class ShardedStudentReadyParquet(Dataset):
    """
    Reads a sharded dataset created as:
      data_root/shard_XXXX/data.parquet

    Each shard parquet must contain columns:
      tile_embeddings_f16 (binary)
      teacher_image_probs_f16 (binary)
      metadata_tensor_f32 (FixedSizeList[4] float32)
    """

    def __init__(
        self,
        data_root: str,
        num_tiles: int = 16,
        emb_dim: int = 768,
        num_classes: int = 7806,
        pattern: str = "shard_*/data.parquet",
        cache_max_tables: int = 2,
    ):
        self.data_root = data_root
        self.num_tiles = num_tiles
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.cache_max_tables = cache_max_tables

        files = sorted(glob.glob(os.path.join(data_root, pattern)))
        if not files:
            raise ValueError(f"No shard parquet files found under {data_root} with pattern {pattern}")

        self.files = files

        # Build prefix sums of row counts for fast global index mapping
        self.row_counts = []
        self.cum_rows = [0]
        for fp in self.files:
            pf = pq.ParquetFile(fp)
            n = pf.metadata.num_rows
            self.row_counts.append(n)
            self.cum_rows.append(self.cum_rows[-1] + n)

        self.total_rows = self.cum_rows[-1]

        # Small in process table cache
        self._cache = {}        # path -> pa.Table
        self._cache_order = []  # LRU order

        # Validate required columns using the first file
        first = pq.read_table(self.files[0], columns=["tile_embeddings_f16", "teacher_image_probs_f16", "metadata_tensor_f32"])
        required = ["tile_embeddings_f16", "teacher_image_probs_f16", "metadata_tensor_f32"]
        missing = [c for c in required if c not in first.column_names]
        if missing:
            raise ValueError(f"Missing columns {missing}. Found {first.column_names}")

    def __len__(self):
        return self.total_rows

    def _get_table(self, fp: str) -> pa.Table:
        if fp in self._cache:
            # refresh LRU
            if fp in self._cache_order:
                self._cache_order.remove(fp)
            self._cache_order.append(fp)
            return self._cache[fp]

        tbl = pq.read_table(fp, columns=["tile_embeddings_f16", "teacher_image_probs_f16", "metadata_tensor_f32"])
        self._cache[fp] = tbl
        self._cache_order.append(fp)

        # evict LRU
        while len(self._cache_order) > self.cache_max_tables:
            old = self._cache_order.pop(0)
            if old in self._cache:
                del self._cache[old]

        return tbl

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_rows:
            raise IndexError(idx)

        # Find which shard file contains idx
        shard_i = bisect.bisect_right(self.cum_rows, idx) - 1
        local_i = idx - self.cum_rows[shard_i]
        fp = self.files[shard_i]

        tbl = self._get_table(fp)
        row = tbl.slice(local_i, 1)

        tile_blob = row.column("tile_embeddings_f16")[0].as_py()
        teacher_blob = row.column("teacher_image_probs_f16")[0].as_py()
        meta_list = row.column("metadata_tensor_f32")[0].as_py()  # list length 4

        tiles = np.frombuffer(tile_blob, dtype=np.float16).astype(np.float32, copy=False)
        tiles = tiles.reshape(self.num_tiles, self.emb_dim)

        teacher = np.frombuffer(teacher_blob, dtype=np.float16).astype(np.float32, copy=False)
        if teacher.shape[0] != self.num_classes:
            raise ValueError(f"Teacher len {teacher.shape[0]} != {self.num_classes}")

        meta = np.asarray(meta_list, dtype=np.float32).copy()
        if meta.shape[0] != 4:
            raise ValueError("metadata_tensor_f32 must be len 4")

        return (
            torch.from_numpy(tiles),    # [num_tiles, emb_dim]
            torch.from_numpy(meta),     # [4]
            torch.from_numpy(teacher),  # [num_classes]
        )


class SingleFileStudentReadyParquet(Dataset):
    """
    Backward compatible reader for a single parquet file path.
    Suitable for 10k experiments.
    """

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

        table = pq.read_table(parquet_path)
        self.df = table.to_pandas()

        required = ["tile_embeddings_f16", "teacher_image_probs_f16", "metadata_tensor_f32"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing}. Found {list(self.df.columns)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        tile_blob = row["tile_embeddings_f16"]
        teacher_blob = row["teacher_image_probs_f16"]
        meta_list = row["metadata_tensor_f32"]

        tiles = np.frombuffer(tile_blob, dtype=np.float16).astype(np.float32)
        tiles = tiles.reshape(self.num_tiles, self.emb_dim)

        teacher = np.frombuffer(teacher_blob, dtype=np.float16).astype(np.float32)
        if teacher.shape[0] != self.num_classes:
            raise ValueError(f"Teacher len {teacher.shape[0]} != {self.num_classes}")

        meta = np.asarray(meta_list, dtype=np.float32).copy()
        if meta.shape[0] != 4:
            raise ValueError("metadata_tensor_f32 must be len 4")

        return (
            torch.from_numpy(tiles),
            torch.from_numpy(meta),
            torch.from_numpy(teacher),
        )


def masked_distill_loss(
    logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    neg_sample_k: int = 256,
):
    b, c = logits.shape
    device = logits.device

    pos_mask = teacher_probs > 0
    pos_count = pos_mask.sum().item()

    if pos_count > 0:
        pos_logits = logits[pos_mask]
        pos_targets = teacher_probs[pos_mask]
        pos_loss = nn.functional.binary_cross_entropy_with_logits(pos_logits, pos_targets, reduction="mean")
    else:
        pos_loss = logits.sum() * 0.0

    if neg_sample_k > 0:
        neg_mask = ~pos_mask
        neg_losses = []
        for i in range(b):
            zero_idx = torch.nonzero(neg_mask[i], as_tuple=False).squeeze(1)
            if zero_idx.numel() == 0:
                continue
            k = min(neg_sample_k, zero_idx.numel())
            sel = zero_idx[torch.randint(0, zero_idx.numel(), (k,), device=device)]
            neg_logits = logits[i, sel]
            neg_targets = torch.zeros_like(neg_logits)
            neg_losses.append(
                nn.functional.binary_cross_entropy_with_logits(neg_logits, neg_targets, reduction="mean")
            )
        if len(neg_losses) > 0:
            neg_loss = torch.stack(neg_losses).mean()
        else:
            neg_loss = logits.sum() * 0.0
    else:
        neg_loss = logits.sum() * 0.0

    return pos_loss + neg_loss


def co_graph_loss_from_probs(
    probs: torch.Tensor,
    ei,
    ej,
    ew,
    max_edges: int,
):
    if ei is None or ej is None or ew is None:
        return probs.sum() * 0.0
    if ei.numel() == 0:
        return probs.sum() * 0.0

    E = ei.numel()
    if max_edges > 0 and E > max_edges:
        sel = torch.randint(0, E, (max_edges,), device=probs.device)
        ei_s = ei[sel]
        ej_s = ej[sel]
        ew_s = ew[sel]
    else:
        ei_s = ei
        ej_s = ej
        ew_s = ew

    pi = probs.index_select(dim=1, index=ei_s)
    pj = probs.index_select(dim=1, index=ej_s)
    diff2 = (pi - pj) ** 2
    w = ew_s.view(1, -1)
    return (diff2 * w).mean()


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    neg_sample_k,
    co_ei,
    co_ej,
    co_ew,
    lambda_co,
    co_edges_per_batch,
    log_every=50,
):
    model.train()
    running = 0.0
    t0 = time.time()

    for step, (tiles, meta, teacher) in enumerate(loader):
        tiles = tiles.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        teacher = teacher.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(tiles, metadata=meta)
            loss_distill = masked_distill_loss(logits, teacher, neg_sample_k=neg_sample_k)

            probs = torch.sigmoid(logits)
            loss_co = co_graph_loss_from_probs(
                probs,
                co_ei,
                co_ej,
                co_ew,
                max_edges=co_edges_per_batch,
            )

            loss = loss_distill + (lambda_co * loss_co)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        running += loss.item()

        if (step + 1) % log_every == 0:
            dt = time.time() - t0
            with torch.no_grad():
                pos_mask = teacher > 0
                if pos_mask.any():
                    mean_p_pos = torch.sigmoid(logits[pos_mask]).mean().item()
                    mean_t_pos = teacher[pos_mask].mean().item()
                else:
                    mean_p_pos = float("nan")
                    mean_t_pos = float("nan")

            print(
                f"  step {step+1:6d}  "
                f"loss {running/(step+1):.6f}  "
                f"l_dist {float(loss_distill.item()):.6f}  "
                f"l_co {float(loss_co.item()):.6f}  "
                f"mean_t_pos {mean_t_pos:.4f}  "
                f"mean_p_pos {mean_p_pos:.4f}  "
                f"sec {dt:.1f}"
            )

    return running / max(1, len(loader))


@torch.no_grad()
def eval_one_epoch(model, loader, device, neg_sample_k):
    model.eval()
    total = 0.0
    for tiles, meta, teacher in loader:
        tiles = tiles.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        teacher = teacher.to(device, non_blocking=True)
        logits = model(tiles, metadata=meta)
        loss = masked_distill_loss(logits, teacher, neg_sample_k=neg_sample_k)
        total += loss.item()
    return total / max(1, len(loader))


def main():
    ap = argparse.ArgumentParser()

    # Input
    ap.add_argument("--data-parquet", type=str, default="", help="Single parquet file (10k style)")
    ap.add_argument("--data-root", type=str, default="", help="Directory containing shard_*/data.parquet (212k style)")

    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--neg-sample-k", type=int, default=256)
    ap.add_argument("--val-split", type=float, default=0.05)

    ap.add_argument("--num-tiles", type=int, default=16)
    ap.add_argument("--emb-dim", type=int, default=768)
    ap.add_argument("--num-classes", type=int, default=7806)

    # Co-occurrence regularizer
    ap.add_argument("--co-graph-npz", type=str, default="")
    ap.add_argument("--lambda-co", type=float, default=0.01)
    ap.add_argument("--co-edges-per-batch", type=int, default=200000)

    args = ap.parse_args()

    if not args.data_root and not args.data_parquet:
        raise ValueError("Provide --data-root for sharded data, or --data-parquet for a single file")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    os.makedirs(args.out_dir, exist_ok=True)

    # Load dataset
    if args.data_root:
        ds_obj = ShardedStudentReadyParquet(
            data_root=args.data_root,
            num_tiles=args.num_tiles,
            emb_dim=args.emb_dim,
            num_classes=args.num_classes,
        )
        print("dataset mode: sharded")
        print("shards:", len(ds_obj.files), "rows:", len(ds_obj))
    else:
        ds_obj = SingleFileStudentReadyParquet(
            parquet_path=args.data_parquet,
            num_tiles=args.num_tiles,
            emb_dim=args.emb_dim,
            num_classes=args.num_classes,
        )
        print("dataset mode: single file")
        print("rows:", len(ds_obj))

    # Split
    n = len(ds_obj)
    n_val = int(n * args.val_split)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(123)
    train_ds, val_ds = torch.utils.data.random_split(ds_obj, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # Model
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Co-graph
    co_ei = None
    co_ej = None
    co_ew = None
    if args.co_graph_npz:
        g = np.load(args.co_graph_npz)
        for k in ["neigh_i", "neigh_j", "neigh_w"]:
            if k not in g:
                raise ValueError(f"co graph npz missing key {k}. Found keys {list(g.keys())}")

        co_ei = torch.from_numpy(g["neigh_i"].astype(np.int64)).to(device)
        co_ej = torch.from_numpy(g["neigh_j"].astype(np.int64)).to(device)
        co_ew = torch.from_numpy(g["neigh_w"].astype(np.float32)).to(device)

        print("loaded co graph edges:", int(co_ei.numel()))
        if co_ei.numel() != co_ej.numel() or co_ei.numel() != co_ew.numel():
            raise ValueError("co graph arrays must have same length")

    best = float("inf")
    bad = 0
    patience = 3

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            neg_sample_k=args.neg_sample_k,
            co_ei=co_ei,
            co_ej=co_ej,
            co_ew=co_ew,
            lambda_co=args.lambda_co if args.co_graph_npz else 0.0,
            co_edges_per_batch=args.co_edges_per_batch,
            log_every=50,
        )

        vl = eval_one_epoch(model, val_loader, device, args.neg_sample_k)
        print(f"train_loss {tr:.6f}  val_loss {vl:.6f}")

        ckpt = os.path.join(args.out_dir, f"student_epoch_{epoch:02d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": vl,
                "co_graph_npz": args.co_graph_npz,
                "lambda_co": args.lambda_co,
                "co_edges_per_batch": args.co_edges_per_batch,
            },
            ckpt,
        )

        if vl + 1e-6 < best:
            best = vl
            bad = 0
            best_path = os.path.join(args.out_dir, "student_v1_distilled.pt")
            torch.save(model.state_dict(), best_path)
            print("saved best:", best_path)
        else:
            bad += 1
            if bad >= patience:
                print("early stop: plateau")
                break


if __name__ == "__main__":
    main()


"""
Example 212k usage

python src/step3_v2/train_step3a_student_distill_cograph.py \
  --data-root /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k/student_ready_merged_input_step3a \
  --out-dir  /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/checkpoints/lucas_212k_step3a_distill_baseline \
  --epochs 10 \
  --batch-size 64 \
  --num-workers 6 \
  --neg-sample-k 128

Example 212k with co graph

python src/step3_v2/train_step3a_student_distill_cograph.py \
  --data-root /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k/student_ready_merged_input_step3a \
  --out-dir  /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/checkpoints/lucas_212k_step3a_distill_cograph \
  --epochs 10 \
  --batch-size 64 \
  --num-workers 6 \
  --neg-sample-k 128 \
  --co-graph-npz /path/to/lucas212k_cograph_alpha060_min100_cap8_v1.npz \
  --lambda-co 0.01 \
  --co-edges-per-batch 200000
"""
