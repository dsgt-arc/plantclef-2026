"""Tier 2: train LoRA-adapted DINOv2 + ArcFace head for retrieval.

Backbone: DINOv2 ViT-Base (timm), fully frozen.
LoRA: low-rank adapters injected into attn.qkv of every block (trainable).
Head: 768 -> 768 -> BN -> ReLU -> 768 -> L2-normalize (trainable).
Classifier: Sub-center ArcFace (K=3, s=30, m=0.30) over 7806 species.

Training data: raw PlantCLEF 2024 training images in ImageFolder layout
(species-id subdirs). NOT cached embeddings — we need raw pixels because the
backbone itself is being modified.

Because the backbone runs every step, this is ~1000x more compute per step than
Tier 1. Expected wallclock: 8-16 hours on RTX 6000, ~4-8 hours on H200.

Usage:
  python train_arcface_lora.py \
    --data-dir /storage/.../train/PlantCLEF2024 \
    --backbone-path /storage/.../vit_base_patch14_reg4_dinov2.pth \
    --out-dir /storage/.../exp9b_knn/arcface_lora \
    --epochs 3 --batch-size 64 --lr 3e-4 --lora-r 8 --lora-alpha 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lora import (
    freeze_non_lora,
    inject_lora_into_attention,
    lora_state_dict,
)

NUM_CLASSES = 7806
IN_DIM = 768
DINOV2_SIZE = 518
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_NAME = "vit_base_patch14_reg4_dinov2.lvd142m"


def build_train_transform(image_size: int, jpeg_quality: int) -> transforms.Compose:
    """Training transform: center-resize -> optional JPEG85 -> normalize.

    Keeps augmentation minimal — metric learning doesn't need heavy aug and
    the test pipeline uses center-crop + JPEG85 + resize, so matching that
    distribution at train time is more important than aggressive aug.
    """
    tx_list = [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(image_size),
    ]
    if jpeg_quality > 0:
        tx_list.append(JPEGCompression(quality=jpeg_quality, subsampling=1))
    tx_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(tx_list)


class JPEGCompression:
    """Round-trip JPEG compression to match the test-time JPEG85 tile distribution."""

    def __init__(self, quality: int = 85, subsampling: int = 1) -> None:
        self.quality = quality
        self.subsampling = subsampling

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality, subsampling=self.subsampling)
        buf.seek(0)
        return Image.open(buf).copy()


class ArcFaceHead(nn.Module):
    """Projection head: in_dim -> out_dim -> BN -> ReLU -> out_dim -> L2-norm."""

    def __init__(self, in_dim: int = IN_DIM, out_dim: int = 768) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc2(self.act(self.bn(self.fc1(x))))
        return F.normalize(x, p=2, dim=1)


class SubCenterArcFace(nn.Module):
    """Sub-center ArcFace classifier (Deng et al. 2020)."""

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

    def forward(self, emb: torch.Tensor, labels: torch.Tensor, margin: float | None = None):
        m = self.m if margin is None else margin
        w = F.normalize(self.weight, p=2, dim=1)
        cos_all = emb @ w.t()
        cos_all = cos_all.view(-1, self.num_classes, self.k_sub)
        cos, _ = cos_all.max(dim=2)

        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos)
        target = F.one_hot(labels, self.num_classes).to(cos.dtype)
        cos_m = torch.cos(theta + target * m)
        return self.s * cos_m


class ImageFolderWithSpeciesId(ImageFolder):
    """ImageFolder that maps the string class name (species_id) -> class index
    using a canonical species map.

    ImageFolder normally assigns class indices by folder-name sort order, which
    can drift from our canonical class index if a species is missing. This
    subclass re-maps __getitem__ so labels are guaranteed to match the
    class_to_species_id.json convention.
    """

    def __init__(self, root: str, transform=None, species_to_class: dict[int, int] | None = None):
        super().__init__(root, transform=transform)
        if species_to_class is None:
            self._remap = None
        else:
            new_samples = []
            n_skipped = 0
            for path, local_idx in self.samples:
                dir_name = self.classes[local_idx]
                try:
                    sid = int(dir_name)
                except ValueError:
                    n_skipped += 1
                    continue
                if sid not in species_to_class:
                    n_skipped += 1
                    continue
                new_samples.append((path, species_to_class[sid]))
            self.samples = new_samples
            self.targets = [t for _, t in new_samples]
            self._remap = True
            print(f"[data] remapped ImageFolder to canonical class indices, "
                  f"skipped {n_skipped} samples")


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
    image_size: int
    jpeg_quality: int
    lora_r: int
    lora_alpha: float
    lora_dropout: float
    lora_on_proj: bool
    grad_checkpoint: bool
    num_workers: int
    log_every: int
    seed: int
    max_per_class: int


def load_species_to_class(species_map_path: str) -> dict[int, int]:
    with open(species_map_path) as f:
        class_to_species = json.load(f)
    return {int(sid): idx for idx, sid in enumerate(class_to_species)}


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def maybe_subsample_per_class(
    samples: list[tuple[str, int]],
    max_per_class: int,
    seed: int = 42,
) -> list[tuple[str, int]]:
    if max_per_class <= 0:
        return samples
    rng = np.random.RandomState(seed)
    by_class: dict[int, list[tuple[str, int]]] = {}
    for path, lbl in samples:
        by_class.setdefault(lbl, []).append((path, lbl))

    kept: list[tuple[str, int]] = []
    n_capped = 0
    for lbl, items in by_class.items():
        if len(items) > max_per_class:
            idx = rng.choice(len(items), max_per_class, replace=False)
            kept.extend(items[i] for i in idx)
            n_capped += 1
        else:
            kept.extend(items)
    rng.shuffle(kept)
    print(f"[data] max-per-class subsample: capped {n_capped} classes, "
          f"kept {len(kept)} total")
    return kept


def build_model(
    backbone_path: str,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[nn.Module, ArcFaceHead, SubCenterArcFace, list[str]]:
    print(f"[model] loading {MODEL_NAME} from {backbone_path}")
    backbone = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
        checkpoint_path=backbone_path,
    )
    backbone = backbone.to(device)

    touched = inject_lora_into_attention(
        backbone,
        r=cfg.lora_r,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        include_proj=cfg.lora_on_proj,
    )
    print(f"[model] injected LoRA on {len(touched)} linears (qkv"
          + (" + proj" if cfg.lora_on_proj else "") + ")")
    n_train, n_frozen = freeze_non_lora(backbone)

    # Move freshly-created LoRA parameters to the correct device
    backbone = backbone.to(device)

    if cfg.grad_checkpoint and hasattr(backbone, "set_grad_checkpointing"):
        backbone.set_grad_checkpointing(enable=True)
        print("[model] gradient checkpointing: ON")

    head = ArcFaceHead(in_dim=IN_DIM, out_dim=cfg.out_dim).to(device)
    arc = SubCenterArcFace(
        emb_dim=cfg.out_dim,
        num_classes=NUM_CLASSES,
        k_sub=cfg.k_sub,
        s=cfg.scale,
        m=cfg.margin,
    ).to(device)

    n_head = sum(p.numel() for p in head.parameters())
    n_arc = sum(p.numel() for p in arc.parameters())
    print(
        f"[model] params: backbone frozen={n_frozen/1e6:.1f}M  "
        f"lora_trainable={n_train/1e6:.2f}M  head={n_head/1e6:.2f}M  "
        f"arc={n_arc/1e6:.2f}M"
    )
    return backbone, head, arc, touched


def in_batch_rank1(emb: torch.Tensor, labels: torch.Tensor) -> float:
    n = emb.shape[0]
    if n < 4:
        return float("nan")
    half = n // 2
    q, q_lbl = emb[:half], labels[:half]
    g, g_lbl = emb[half : 2 * half], labels[half : 2 * half]
    sims = q @ g.t()
    return (g_lbl[sims.argmax(dim=1)] == q_lbl).float().mean().item()


def intra_inter_cosine(emb: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
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


def forward_embeddings(
    backbone: nn.Module,
    head: ArcFaceHead,
    x: torch.Tensor,
    grad_checkpoint: bool = False,
) -> torch.Tensor:
    # Gradient-checkpointed blocks (especially with use_reentrant=True in older
    # torch) require the input tensor to require_grad. Our inputs don't, but
    # LoRA adapters do — so we flip the input's grad flag only while
    # checkpointing, to ensure gradients flow through the backbone to LoRA.
    if grad_checkpoint and not x.requires_grad:
        x = x.detach().requires_grad_(True)
    features = backbone.forward_features(x)
    cls = features[:, 0, :]
    return head(cls)


def train(cfg: TrainConfig, args: argparse.Namespace) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")

    species_to_class = load_species_to_class(args.species_map)
    print(f"[init] {len(species_to_class)} species in mapping")

    transform = build_train_transform(cfg.image_size, cfg.jpeg_quality)
    dataset = ImageFolderWithSpeciesId(
        root=args.data_dir,
        transform=transform,
        species_to_class=species_to_class,
    )
    dataset.samples = maybe_subsample_per_class(
        dataset.samples, cfg.max_per_class, seed=cfg.seed
    )
    dataset.targets = [t for _, t in dataset.samples]

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
    margin_warmup_steps = max(1, steps_per_epoch // 2)
    print(
        f"[init] n_samples={len(dataset)} steps_per_epoch={steps_per_epoch} "
        f"total_steps={total_steps} margin_warmup_steps={margin_warmup_steps}"
    )

    backbone, head, arc, touched = build_model(args.backbone_path, cfg, device)

    trainable_params = [p for p in backbone.parameters() if p.requires_grad]
    trainable_params += list(head.parameters())
    trainable_params += list(arc.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log_fh = open(log_path, "w")
    print(f"[init] logging to {log_path}")

    step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        backbone.train()
        head.train()
        arc.train()

        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            lr = cosine_lr(step, total_steps, cfg.warmup_steps, cfg.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            cur_margin = (
                cfg.margin * (step + 1) / margin_warmup_steps
                if step < margin_warmup_steps
                else cfg.margin
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                z = forward_embeddings(
                    backbone, head, imgs, grad_checkpoint=cfg.grad_checkpoint
                )
                logits = arc(z, labels, margin=cur_margin)
                loss = F.cross_entropy(
                    logits, labels, label_smoothing=cfg.label_smoothing
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            if step % cfg.log_every == 0:
                with torch.no_grad():
                    z_f = z.detach().float()
                    r1 = in_batch_rank1(z_f, labels)
                    intra, inter = intra_inter_cosine(z_f, labels)
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
                rate = (step + 1) / max(1e-6, time.time() - t0)
                eta = (total_steps - step - 1) / max(1e-6, rate)
                print(
                    f"[step {step:>7d}/{total_steps}] ep={epoch} "
                    f"loss={loss.item():.4f} lr={lr:.2e} m={cur_margin:.3f} "
                    f"r1={r1:.3f} intra={intra:.3f} inter={inter:.3f} "
                    f"rate={rate:.2f} st/s ETA={eta/60:.1f}m"
                )
            step += 1

        ckpt_path = os.path.join(args.out_dir, f"lora_head_ep{epoch + 1}.pt")
        torch.save(
            {
                "lora": lora_state_dict(backbone),
                "head": head.state_dict(),
                "arc": arc.state_dict(),
                "config": cfg.__dict__,
                "epoch": epoch + 1,
                "step": step,
            },
            ckpt_path,
        )
        print(f"[ckpt] wrote {ckpt_path}")

    final_path = os.path.join(args.out_dir, "lora_head_final.pt")
    torch.save(
        {
            "lora": lora_state_dict(backbone),
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
        description="Tier 2: LoRA+ArcFace fine-tune of DINOv2 for retrieval"
    )
    ap.add_argument("--data-dir", required=True,
                    help="ImageFolder root (species_id subdirs).")
    ap.add_argument("--backbone-path", required=True,
                    help="Frozen DINOv2 ViT-Base checkpoint (.pth).")
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--margin", type=float, default=0.30)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--k-sub", type=int, default=3)
    ap.add_argument("--out-dim", type=int, default=768)
    ap.add_argument("--label-smoothing", type=float, default=0.10)
    ap.add_argument("--image-size", type=int, default=518,
                    help="Must match deployment (518 for DINOv2 tiles).")
    ap.add_argument("--jpeg-quality", type=int, default=85,
                    help="Train-time JPEG round-trip (matches test pipeline). 0=off.")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lora-on-proj", action="store_true",
                    help="Also inject LoRA on attn.proj (default: qkv only).")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="Disable gradient checkpointing (faster but more memory).")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="Cap images per species to this many (0=use all).")
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
        image_size=args.image_size,
        jpeg_quality=args.jpeg_quality,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_on_proj=args.lora_on_proj,
        grad_checkpoint=not args.no_grad_checkpoint,
        num_workers=args.num_workers,
        log_every=args.log_every,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )
    train(cfg, args)


if __name__ == "__main__":
    main()
