#!/usr/bin/env python

import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pyarrow.parquet as pq

from Cross_Region_Transformer_v3 import CrossRegionTransformer  # already in your repo


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
        meta_list = row["metadata_tensor_f32"]  # FixedSizeList -> list

        tiles = np.frombuffer(tile_blob, dtype=np.float16).astype(np.float32)
        tiles = tiles.reshape(self.num_tiles, self.emb_dim)

        teacher = np.frombuffer(teacher_blob, dtype=np.float16).astype(np.float32)
        if teacher.shape[0] != self.num_classes:
            raise ValueError(f"Teacher len {teacher.shape[0]} != {self.num_classes}")

        meta = np.asarray(meta_list, dtype=np.float32)
        if meta.shape[0] != 4:
            raise ValueError("metadata_tensor_f32 must be len 4")

        return (
            torch.from_numpy(tiles),          # [16, 768]
            torch.from_numpy(meta),           # [4]
            torch.from_numpy(teacher),        # [7806]
        )


def masked_distill_loss(
    logits: torch.Tensor,         # [B, C]
    teacher_probs: torch.Tensor,  # [B, C] in [0,1], mostly zeros outside TopK
    neg_sample_k: int = 256,
):
    """
    Distillation when teacher is sparse TopK in a dense vector.
    We treat nonzero teacher entries as supervised.
    Zeros are unknown, not true negatives, so we only add a small random negative sample.

    Loss = BCE on positives (teacher>0) + BCE on sampled negatives with target=0
    """
    b, c = logits.shape
    device = logits.device

    pos_mask = teacher_probs > 0
    pos_count = pos_mask.sum().item()

    # BCE on positives only
    if pos_count > 0:
        pos_logits = logits[pos_mask]
        pos_targets = teacher_probs[pos_mask]
        pos_loss = nn.functional.binary_cross_entropy_with_logits(pos_logits, pos_targets, reduction="mean")
    else:
        pos_loss = logits.sum() * 0.0

    # random negative sampling from teacher == 0
    if neg_sample_k > 0:
        neg_mask = ~pos_mask
        # sample indices per batch row
        # build (B, K) indices of classes where teacher==0
        neg_losses = []
        for i in range(b):
            zero_idx = torch.nonzero(neg_mask[i], as_tuple=False).squeeze(1)
            if zero_idx.numel() == 0:
                continue
            k = min(neg_sample_k, zero_idx.numel())
            sel = zero_idx[torch.randint(0, zero_idx.numel(), (k,), device=device)]
            neg_logits = logits[i, sel]
            neg_targets = torch.zeros_like(neg_logits)
            neg_losses.append(nn.functional.binary_cross_entropy_with_logits(neg_logits, neg_targets, reduction="mean"))
        if len(neg_losses) > 0:
            neg_loss = torch.stack(neg_losses).mean()
        else:
            neg_loss = logits.sum() * 0.0
    else:
        neg_loss = logits.sum() * 0.0

    return pos_loss + neg_loss


def train_one_epoch(model, loader, optimizer, scaler, device, neg_sample_k, log_every=50):
    model.train()
    running = 0.0
    t0 = time.time()

    for step, (tiles, meta, teacher) in enumerate(loader):
        tiles = tiles.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        teacher = teacher.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(tiles, metadata=meta)  # [B, 7806]
            loss = masked_distill_loss(logits, teacher, neg_sample_k=neg_sample_k)

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
            print(f"  step {step+1:6d}  loss {running/(step+1):.6f}  sec {dt:.1f}")

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
    ap.add_argument("--data-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--neg-sample-k", type=int, default=256)
    ap.add_argument("--val-split", type=float, default=0.05)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    os.makedirs(args.out_dir, exist_ok=True)

    ds = StudentReadyParquet(args.data_parquet)

    # quick deterministic split
    n = len(ds)
    n_val = int(n * args.val_split)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(123)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best = float("inf")
    bad = 0
    patience = 3

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device, args.neg_sample_k)
        vl = eval_one_epoch(model, val_loader, device, args.neg_sample_k)
        print(f"train_loss {tr:.6f}  val_loss {vl:.6f}")

        ckpt = os.path.join(args.out_dir, f"student_epoch_{epoch:02d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": vl,
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
