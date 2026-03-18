#!/usr/bin/env python

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader

from Cross_Region_Transformer_v3 import CrossRegionTransformer


class Step3BSelfTrainDataset(Dataset):
    """
    Joins merged inputs with step3b pseudo labels by image_name.

    Inputs per item:
      tiles: [16, 768] float32
      meta:  [4] float32
      pos_idx: 1D int64 indices (variable length)
      pos_prob: 1D float32 probabilities aligned with pos_idx (variable length)
    """

    def __init__(
        self,
        merged_parquet: str,
        pseudo_parquet: str,
        num_tiles: int = 16,
        emb_dim: int = 768,
    ):
        self.num_tiles = num_tiles
        self.emb_dim = emb_dim

        m = pq.read_table(
            merged_parquet,
            columns=["image_name", "tile_embeddings_f16", "metadata_tensor_f32"],
        ).to_pandas()

        p = pq.read_table(
            pseudo_parquet,
            columns=["image_name", "k", "pos_idx_i32", "pos_prob_f16"],
        ).to_pandas()

        pseudo = {}
        for _, r in p.iterrows():
            k = int(r["k"])
            idx = np.frombuffer(r["pos_idx_i32"], dtype=np.int32, count=k)
            prob = np.frombuffer(r["pos_prob_f16"], dtype=np.float16, count=k).astype(
                np.float32
            )
            pseudo[r["image_name"]] = (idx, prob)

        self.rows = []
        for _, r in m.iterrows():
            img = r["image_name"]
            if img not in pseudo:
                continue
            idx, prob = pseudo[img]
            self.rows.append(
                (img, r["tile_embeddings_f16"], r["metadata_tensor_f32"], idx, prob)
            )

        print("merged rows:", len(m))
        print("pseudo rows:", len(p))
        print("training rows kept:", len(self.rows))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        img, tile_blob, meta_list, idx, prob = self.rows[i]

        tiles = np.frombuffer(tile_blob, dtype=np.float16).astype(np.float32)
        tiles = tiles.reshape(self.num_tiles, self.emb_dim)

        meta = np.asarray(meta_list, dtype=np.float32).copy()

        return (
            torch.from_numpy(tiles),
            torch.from_numpy(meta),
            torch.from_numpy(idx.astype(np.int64)),
            torch.from_numpy(prob.astype(np.float32)),
        )


def collate_varlen(batch):
    tiles = torch.stack([b[0] for b in batch], dim=0)
    meta = torch.stack([b[1] for b in batch], dim=0)
    idx_list = [b[2] for b in batch]
    prob_list = [b[3] for b in batch]
    return tiles, meta, idx_list, prob_list


def freeze_by_name(model: nn.Module, trainable_substrings):
    trainable_substrings = [s.strip() for s in trainable_substrings if s.strip()]
    for name, p in model.named_parameters():
        p.requires_grad = False
        for s in trainable_substrings:
            if s in name:
                p.requires_grad = True
                break


def masked_selftrain_loss(
    logits: torch.Tensor,
    idx_list,
    prob_list,
    neg_sample_k: int = 256,
):
    """
    logits: [B, C]
    idx_list: list of 1D int64 tensors, each contains positive class indices
    prob_list: list of 1D float tensors, each contains soft targets in [0,1] aligned with idx_list

    Loss = soft-positive BCE on selected indices + sampled-negative BCE toward 0
    This prevents the degenerate "make everything positive" collapse.
    """
    B, C = logits.shape
    losses = []

    for i, (idx, prob) in enumerate(zip(idx_list, prob_list)):
        if idx.numel() == 0:
            continue

        idx = idx.to(logits.device, non_blocking=True)
        prob = prob.to(logits.device, non_blocking=True)

        pos_logits = logits[i, idx]
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, prob, reduction="mean")

        k = min(int(neg_sample_k), C - int(idx.numel()))
        if k > 0:
            # sample candidate negatives
            neg_idx = torch.randint(0, C, (k,), device=logits.device)

            # avoid sampling positives
            if idx.numel() < C:
                pos_mask = torch.zeros(C, dtype=torch.bool, device=logits.device)
                pos_mask[idx] = True
                neg_idx = neg_idx[~pos_mask[neg_idx]]
                if neg_idx.numel() < k:
                    extra = torch.randint(0, C, (k - neg_idx.numel(),), device=logits.device)
                    neg_idx = torch.cat([neg_idx, extra], dim=0)

            neg_idx = neg_idx[:k]
            neg_logits = logits[i, neg_idx]
            neg_targets = torch.zeros_like(neg_logits)
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, neg_targets, reduction="mean")
        else:
            neg_loss = pos_loss * 0.0

        losses.append(pos_loss + neg_loss)

    if len(losses) == 0:
        return logits.sum() * 0.0

    return torch.stack(losses).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-parquet", required=True)
    ap.add_argument("--pseudo-parquet", required=True)
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)

    ap.add_argument("--freeze-epochs", type=int, default=2)
    ap.add_argument("--trainable-names", type=str, default="classifier,fc,head,output,metadata")
    ap.add_argument("--neg-sample-k", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = Step3BSelfTrainDataset(args.merged_parquet, args.pseudo_parquet)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_varlen,
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

    sd = torch.load(args.init_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=True)

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.freeze_epochs:
            freeze_by_name(model, args.trainable_names.split(","))
            print("freeze mode on. trainable substrings:", args.trainable_names)
        else:
            for p in model.parameters():
                p.requires_grad = True
            print("freeze mode off. training all parameters")

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        model.train()
        running = 0.0
        for step, (tiles, meta, idx_list, prob_list) in enumerate(loader):
            tiles = tiles.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            # idx/prob are variable-length; move them in the loss function per-sample

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(tiles, metadata=meta)
                loss = masked_selftrain_loss(
                    logits,
                    idx_list=idx_list,
                    prob_list=prob_list,
                    neg_sample_k=args.neg_sample_k,
                )

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

            running += float(loss.item())

            if (step + 1) % 50 == 0:
                print(f"  step {step+1:6d}  loss {running/(step+1):.6f}")

        epoch_loss = running / max(1, len(loader))
        print(f"epoch {epoch} loss {epoch_loss:.6f}")

        out_path = os.path.join(
            args.out_dir, f"student_v2_selftrained_epoch_{epoch:02d}.pt"
        )
        torch.save(model.state_dict(), out_path)
        print("saved:", out_path)

    final_path = os.path.join(args.out_dir, "student_v2_selftrained.pt")
    torch.save(model.state_dict(), final_path)
    print("saved final:", final_path)


if __name__ == "__main__":
    main()


'''
python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/step3_v2/train_step3b_selftrain.py \
  --merged-parquet /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k/student_ready_merged_input_step3a \
  --pseudo-parquet /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k/step3b_pseudolabels \
  --init-ckpt /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/checkpoints/lucas_212k_step3a_distill/student_best_20260218_2210.pt \
  --out-dir /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/checkpoints/lucas_212k_step3b_distill \
  --epochs 4 \
  --batch-size 64 \
  --num-workers 6 \
  --lr 3e-5

'''