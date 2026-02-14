#!/usr/bin/env python
"""
eval_distill_loss_compare.py

Compares Step 3A vs Step 3B checkpoints by computing the same Step 3A-style
distillation loss against teacher targets on a held-out split.

It prints:
- val_loss_step3a_model
- val_loss_step3b_model
- also optional mean_t_pos / mean_p_pos diagnostics on the teacher-positive mask

This uses your merged parquet:
  image_name, tile_embeddings_f16, teacher_image_probs_f16, metadata_tensor_f32
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader

from Cross_Region_Transformer_v3 import CrossRegionTransformer


class MergedStudentReadyDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        num_tiles: int = 16,
        emb_dim: int = 768,
        num_classes: int = 7806,
        split: str = "val",
        val_frac: float = 0.1,
        seed: int = 1337,
    ):
        self.num_tiles = num_tiles
        self.emb_dim = emb_dim
        self.num_classes = num_classes

        cols = ["image_name", "tile_embeddings_f16", "teacher_image_probs_f16", "metadata_tensor_f32"]
        df = pq.read_table(parquet_path, columns=cols).to_pandas()

        # deterministic split by hashing image_name
        rng = np.random.RandomState(seed)
        perm = np.arange(len(df))
        rng.shuffle(perm)
        n_val = int(round(val_frac * len(df)))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        use_idx = val_idx if split == "val" else train_idx
        self.df = df.iloc[use_idx].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]

        tiles = np.frombuffer(r["tile_embeddings_f16"], dtype=np.float16).astype(np.float32)
        tiles = tiles.reshape(self.num_tiles, self.emb_dim)

        teacher = np.frombuffer(r["teacher_image_probs_f16"], dtype=np.float16).astype(np.float32)
        teacher = teacher.reshape(self.num_classes)

        meta = np.asarray(r["metadata_tensor_f32"], dtype=np.float32).copy()

        return (
            torch.from_numpy(tiles),     # [T, D]
            torch.from_numpy(meta),      # [4]
            torch.from_numpy(teacher),   # [C]
        )


@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    neg_sample_k: int = 128,
    eps: float = 1e-12,
):
    """
    Matches Step 3A logic: BCE on teacher-positive entries plus sampled negatives.

    teacher-positive mask: teacher > 0
    negatives: random sample from remaining classes
    """
    model.eval()

    total_loss = 0.0
    total_batches = 0

    mean_t_pos_acc = 0.0
    mean_p_pos_acc = 0.0
    mean_cnt_pos_acc = 0.0

    C = None

    for tiles, meta, teacher in loader:
        tiles = tiles.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        teacher = teacher.to(device, non_blocking=True)

        logits = model(tiles, metadata=meta)  # [B, C]
        if C is None:
            C = logits.shape[1]

        pos_mask = teacher > 0.0  # [B, C]
        cnt_pos = pos_mask.sum(dim=1).float()  # [B]

        # Positive loss (soft targets)
        if pos_mask.any():
            pos_logits = logits[pos_mask]
            pos_t = teacher[pos_mask].clamp(min=0.0, max=1.0)
            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, pos_t, reduction="mean")
            mean_t_pos = pos_t.mean().item()
            mean_p_pos = torch.sigmoid(pos_logits).mean().item()
        else:
            pos_loss = logits.sum() * 0.0
            mean_t_pos = float("nan")
            mean_p_pos = float("nan")

        # Negative sampling: sample K per example from non-positive classes
        k = int(neg_sample_k)
        if k > 0:
            neg_losses = []
            for b in range(logits.shape[0]):
                # indices of non-positive classes
                nonpos = (~pos_mask[b]).nonzero(as_tuple=False).squeeze(1)
                if nonpos.numel() == 0:
                    continue
                kk = min(k, int(nonpos.numel()))
                sel = nonpos[torch.randint(0, nonpos.numel(), (kk,), device=device)]
                neg_logits = logits[b, sel]
                neg_t = torch.zeros_like(neg_logits)
                neg_losses.append(F.binary_cross_entropy_with_logits(neg_logits, neg_t, reduction="mean"))
            if len(neg_losses) > 0:
                neg_loss = torch.stack(neg_losses).mean()
            else:
                neg_loss = pos_loss * 0.0
        else:
            neg_loss = pos_loss * 0.0

        loss = pos_loss + neg_loss

        total_loss += float(loss.item())
        total_batches += 1

        # diagnostics
        mean_cnt_pos_acc += float(cnt_pos.mean().item())
        if np.isfinite(mean_t_pos):
            mean_t_pos_acc += float(mean_t_pos)
        if np.isfinite(mean_p_pos):
            mean_p_pos_acc += float(mean_p_pos)

    out = {
        "val_loss": total_loss / max(1, total_batches),
        "mean_cnt_pos": mean_cnt_pos_acc / max(1, total_batches),
        "mean_t_pos": mean_t_pos_acc / max(1, total_batches),
        "mean_p_pos": mean_p_pos_acc / max(1, total_batches),
        "batches": total_batches,
    }
    return out


def load_state_dict_flexible(path: str):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        return sd["model_state_dict"]
    return sd


def build_model(device: torch.device):
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
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-parquet", required=True)
    ap.add_argument("--ckpt-a", required=True, help="Step3A distilled checkpoint")
    ap.add_argument("--ckpt-b", required=True, help="Step3B refined checkpoint")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--neg-sample-k", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds_val = MergedStudentReadyDataset(
        args.data_parquet,
        split="val",
        val_frac=args.val_frac,
        seed=args.seed,
    )
    loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    print("val size:", len(ds_val), "batches:", int(np.ceil(len(ds_val) / args.batch_size)))

    # Model A
    model_a = build_model(device)
    model_a.load_state_dict(load_state_dict_flexible(args.ckpt_a), strict=True)

    # Model B
    model_b = build_model(device)
    model_b.load_state_dict(load_state_dict_flexible(args.ckpt_b), strict=True)

    a = eval_model(model_a, loader, device, neg_sample_k=args.neg_sample_k)
    b = eval_model(model_b, loader, device, neg_sample_k=args.neg_sample_k)

    print("\nResults on held-out split")
    print("Step3A:", a)
    print("Step3B:", b)
    print("\nDelta (B - A)")
    print(
        {
            "val_loss": b["val_loss"] - a["val_loss"],
            "mean_cnt_pos": b["mean_cnt_pos"] - a["mean_cnt_pos"],
            "mean_t_pos": b["mean_t_pos"] - a["mean_t_pos"],
            "mean_p_pos": b["mean_p_pos"] - a["mean_p_pos"],
        }
    )


if __name__ == "__main__":
    main()



'''
python eval_distill_loss_compare.py \
  --data-parquet /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_10k/student_ready_merged_input_step3a/data.parquet \
  --ckpt-a /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/checkpoints/lucas_10k_step3a_distill/student_v1_distilled.pt \
  --ckpt-b /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/checkpoints/lucas_10k_step3b_selftrain_rel085_v2/student_v2_selftrained.pt \
  --batch-size 256 \
  --num-workers 6 \
  --val-frac 0.1 \
  --neg-sample-k 128

'''
