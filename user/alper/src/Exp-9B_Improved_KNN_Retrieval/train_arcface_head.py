"""Tier 1: Train an ArcFace projection head on top of cached DINOv2 CLS embeddings.

The DINOv2 backbone is not touched. This script loads the 768-d CLS embeddings
extracted elsewhere (one per training image), trains a projection head
(768 -> 512 -> L2-norm) supervised by a sub-center ArcFace classifier over the
7806 species labels, and writes a checkpoint of just the head.

Downstream:
  project_embeddings.py forwards train+test embeddings through the saved head
  build_faiss_index_512.py builds a 512-d FAISS index over the projected train set

Usage:
  python train_arcface_head.py \
    --train-emb-dir  /storage/.../embeddings/train \
    --species-map    /path/to/class_to_species_id.json \
    --out-dir        /path/to/arcface_head \
    --epochs 5 --batch-size 16384 --lr 1e-3
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


NUM_CLASSES = 7806
IN_DIM = 768


class CachedEmbeddingDataset(Dataset):
    """In-memory dataset of (cls_embedding, class_index) pairs.

    Loads all parquet shards under train_emb_dir/sample_id=*/*.parquet into a
    single float32 tensor. At ~1.4M rows x 768 dims x 4 bytes = ~4.3 GB RAM,
    which fits comfortably on a GPU node. float16 storage drops that to ~2.1 GB
    if needed (cast up on __getitem__).
    """

    def __init__(
        self,
        train_emb_dir: str,
        species_to_class: dict[int, int],
        store_fp16: bool = True,
    ) -> None:
        pattern = os.path.join(train_emb_dir, "sample_id=*", "*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No parquet files under {pattern}")
        print(f"[data] Found {len(files)} parquet shards")

        chunks_emb: list[np.ndarray] = []
        chunks_lbl: list[np.ndarray] = []
        n_total = 0
        n_skipped = 0

        for i, fp in enumerate(files):
            df = pd.read_parquet(fp, columns=["species_id", "cls_embedding"])
            sids = df["species_id"].to_numpy(dtype=np.int64)
            keep = np.fromiter(
                (sid in species_to_class for sid in sids),
                dtype=bool,
                count=len(sids),
            )
            n_total += len(df)
            n_skipped += int((~keep).sum())
            if not keep.any():
                continue

            labels_i = np.fromiter(
                (species_to_class[int(sid)] for sid in sids[keep]),
                dtype=np.int32,
                count=int(keep.sum()),
            )
            emb_series = df["cls_embedding"].to_numpy()[keep]
            emb_arr = np.stack(
                [np.asarray(e, dtype=np.float32) for e in emb_series], axis=0
            )
            if store_fp16:
                emb_arr = emb_arr.astype(np.float16)

            chunks_emb.append(emb_arr)
            chunks_lbl.append(labels_i)
            if (i + 1) % 5 == 0 or i == len(files) - 1:
                print(f"[data]   shard {i + 1}/{len(files)} loaded")

        self.embeddings = np.concatenate(chunks_emb, axis=0)
        self.labels = np.concatenate(chunks_lbl, axis=0)
        print(
            f"[data] total_rows={n_total} skipped_unmapped={n_skipped} "
            f"kept={len(self.labels)} dtype={self.embeddings.dtype}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        emb = self.embeddings[idx]
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        return torch.from_numpy(emb), int(self.labels[idx])


class ArcFaceHead(nn.Module):
    """Projection head: 768 -> out_dim -> BN -> ReLU -> out_dim -> L2-norm."""

    def __init__(self, in_dim: int = IN_DIM, out_dim: int = 512) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)


class SubCenterArcFace(nn.Module):
    """Sub-center ArcFace classifier with K centers per class.

    Adapted from Deng et al. (2020). K=1 reduces to standard ArcFace.
    """

    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        k_sub: int = 3,
        s: float = 30.0,
        m: float = 0.30,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.k_sub = k_sub
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(num_classes * k_sub, emb_dim) * 0.01)

    def forward(
        self, emb: torch.Tensor, labels: torch.Tensor, margin: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.m if margin is None else margin
        w = F.normalize(self.weight, p=2, dim=1)
        cos_all = emb @ w.t()

        cos_all = cos_all.view(-1, self.num_classes, self.k_sub)
        cos, _ = cos_all.max(dim=2)

        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos)
        target_mask = F.one_hot(labels, self.num_classes).to(cos.dtype)
        theta_m = theta + target_mask * m
        cos_m = torch.cos(theta_m)

        logits = self.s * cos_m
        return logits, cos


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    margin: float
    scale: float
    k_sub: int
    out_dim: int
    label_smoothing: float
    num_workers: int
    log_every: int
    seed: int


def load_species_to_class(species_map_path: str) -> dict[int, int]:
    with open(species_map_path) as f:
        class_to_species = json.load(f)
    return {int(sid): idx for idx, sid in enumerate(class_to_species)}


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def in_batch_rank1(emb: torch.Tensor, labels: torch.Tensor) -> float:
    """Split batch in half: query vs gallery. Return rank-1 accuracy."""
    n = emb.shape[0]
    if n < 4:
        return float("nan")
    half = n // 2
    q, q_lbl = emb[:half], labels[:half]
    g, g_lbl = emb[half : 2 * half], labels[half : 2 * half]
    sims = q @ g.t()
    top1 = sims.argmax(dim=1)
    pred = g_lbl[top1]
    return (pred == q_lbl).float().mean().item()


def intra_inter_cosine(emb: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Return (mean same-class cosine, mean different-class cosine) in the batch."""
    if emb.shape[0] < 2:
        return float("nan"), float("nan")
    sims = emb @ emb.t()
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    same = same & ~eye
    diff = ~same & ~eye
    intra = sims[same].mean().item() if same.any() else float("nan")
    inter = sims[diff].mean().item() if diff.any() else float("nan")
    return intra, inter


def train(cfg: TrainConfig, args: argparse.Namespace) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")

    species_to_class = load_species_to_class(args.species_map)
    print(f"[init] {len(species_to_class)} species in mapping")

    dataset = CachedEmbeddingDataset(args.train_emb_dir, species_to_class)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg.epochs
    margin_warmup_steps = steps_per_epoch
    print(
        f"[init] steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
        f"margin_warmup_steps={margin_warmup_steps}"
    )

    head = ArcFaceHead(in_dim=IN_DIM, out_dim=cfg.out_dim).to(device)
    arc = SubCenterArcFace(
        emb_dim=cfg.out_dim,
        num_classes=NUM_CLASSES,
        k_sub=cfg.k_sub,
        s=cfg.scale,
        m=cfg.margin,
    ).to(device)

    params = list(head.parameters()) + list(arc.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log_fh = open(log_path, "w")

    print(f"[init] logging to {log_path}")
    print(
        f"[init] head params: {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M"
    )
    print(
        f"[init] arc params:  {sum(p.numel() for p in arc.parameters()) / 1e6:.2f}M"
    )

    step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        head.train()
        arc.train()
        for batch_emb, batch_lbl in loader:
            batch_emb = batch_emb.to(device, non_blocking=True)
            batch_lbl = batch_lbl.to(device, non_blocking=True).long()

            lr = cosine_lr(step, total_steps, cfg.warmup_steps, cfg.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            if step < margin_warmup_steps:
                cur_margin = cfg.margin * (step + 1) / margin_warmup_steps
            else:
                cur_margin = cfg.margin

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                z = head(batch_emb)
                logits, cos_plain = arc(z, batch_lbl, margin=cur_margin)
                loss = F.cross_entropy(
                    logits, batch_lbl, label_smoothing=cfg.label_smoothing
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            if step % cfg.log_every == 0:
                with torch.no_grad():
                    r1 = in_batch_rank1(z.detach().float(), batch_lbl)
                    intra, inter = intra_inter_cosine(
                        z.detach().float(), batch_lbl
                    )
                entry = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "lr": float(lr),
                    "margin": float(cur_margin),
                    "in_batch_rank1": float(r1),
                    "intra_cos": float(intra),
                    "inter_cos": float(inter),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                log_fh.write(json.dumps(entry) + "\n")
                log_fh.flush()
                print(
                    f"[step {step:>6d}/{total_steps}] ep={epoch} "
                    f"loss={loss.item():.4f} lr={lr:.2e} m={cur_margin:.3f} "
                    f"r1={r1:.3f} intra={intra:.3f} inter={inter:.3f}"
                )
            step += 1

        ckpt_path = os.path.join(args.out_dir, f"arcface_head_ep{epoch + 1}.pt")
        torch.save(
            {
                "head": head.state_dict(),
                "arc": arc.state_dict(),
                "config": cfg.__dict__,
                "epoch": epoch + 1,
                "step": step,
            },
            ckpt_path,
        )
        print(f"[ckpt] wrote {ckpt_path}")

    final_path = os.path.join(args.out_dir, "arcface_head_final.pt")
    torch.save(
        {
            "head": head.state_dict(),
            "arc": arc.state_dict(),
            "config": cfg.__dict__,
            "epoch": cfg.epochs,
            "step": step,
        },
        final_path,
    )
    print(f"[ckpt] wrote {final_path}")
    log_fh.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tier 1: train ArcFace projection head on cached DINOv2 CLS"
    )
    ap.add_argument("--train-emb-dir", required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--margin", type=float, default=0.30)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--k-sub", type=int, default=3)
    ap.add_argument("--out-dim", type=int, default=512)
    ap.add_argument("--label-smoothing", type=float, default=0.10)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        margin=args.margin,
        scale=args.scale,
        k_sub=args.k_sub,
        out_dim=args.out_dim,
        label_smoothing=args.label_smoothing,
        num_workers=args.num_workers,
        log_every=args.log_every,
        seed=args.seed,
    )
    train(cfg, args)


if __name__ == "__main__":
    main()
