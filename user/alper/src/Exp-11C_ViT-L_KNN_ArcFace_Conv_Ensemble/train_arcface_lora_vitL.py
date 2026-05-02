"""Exp-11C Step 2: train LoRA-adapted ViT-L (DINOv2) + ArcFace head with
Mixup/CutMix for retrieval and classifier robustness.

Backbone: DINOv2 ViT-Large fine-tuned in Exp-13 phase2v3 (frozen, LoRA on top).
LoRA: low-rank adapters injected into attn.qkv of every block (24 for ViT-L).
Head: 1024 -> 1024 -> BN -> ReLU -> 1024 -> L2-normalize (trainable).
Classifier: Sub-center ArcFace (K=3, s=30, m=0.30) over 7806 species.
Augmentation: Mixup + CutMix (timm.data.Mixup) — soft labels feed straight
into the soft-margin ArcFace + soft-label cross-entropy.

Ported from Exp-9B_Improved_KNN_Retrieval/train_arcface_lora.py.
Changes vs. Exp-9B (ViT-B):
  - MODEL_NAME: vit_large_patch14_reg4_dinov2.lvd142m
  - IN_DIM: 768 -> 1024
  - default --out-dim: 768 -> 1024
  - Backbone load: timm .pth.tar with state_dict_ema (phase2v3 cooldown EMA)
  - Mixup/CutMix wrapper around the input batch
  - ArcFace forward accepts soft targets (lambda·A + (1-lambda)·B), applies
    margin proportionally to each non-zero target component
  - Loss: soft-label cross-entropy on ArcFace logits

Usage:
  python train_arcface_lora_vitL.py \
    --data-dir /storage/.../data/raw/train_full \
    --backbone-path /storage/.../checkpoints/exp13_vit/vit_dinov2_large_phase2v3/model_best.pth.tar \
    --species-map /storage/.../data/raw/class_id_to_species_mapping/class_to_species_id.json \
    --out-dir /storage/.../aerten3/exp11c_vitL_lora_arcface \
    --epochs 3 --batch-size 64 --lr 3e-4 --lora-r 8 --lora-alpha 16 \
    --mixup 0.8 --cutmix 1.0
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
from timm.data import Mixup
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
IN_DIM = 1024
DEFAULT_OUT_DIM = 1024
DINOV2_SIZE = 518
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_NAME = "vit_large_patch14_reg4_dinov2.lvd142m"


def build_train_transform(image_size: int, jpeg_quality: int) -> transforms.Compose:
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
    def __init__(self, in_dim: int = IN_DIM, out_dim: int = DEFAULT_OUT_DIM) -> None:
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
    """Sub-center ArcFace with soft-target support.

    If `targets` is a 1-D LongTensor (vanilla one-hot semantics), behaves
    exactly like Exp-9B. If `targets` is a 2-D FloatTensor (soft labels from
    Mixup/CutMix), applies the angular margin to each non-zero target
    component, weighted by its target mass.
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
        self,
        emb: torch.Tensor,
        targets: torch.Tensor,
        margin: float | None = None,
    ) -> torch.Tensor:
        m = self.m if margin is None else margin
        w = F.normalize(self.weight, p=2, dim=1)
        cos_all = emb @ w.t()
        cos_all = cos_all.view(-1, self.num_classes, self.k_sub)
        cos, _ = cos_all.max(dim=2)
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos)

        if targets.dim() == 1:
            one_hot = F.one_hot(targets, self.num_classes).to(cos.dtype)
        else:
            one_hot = targets.to(cos.dtype)

        cos_m = torch.cos(theta + one_hot * m)
        return self.s * cos_m


class ImageFolderWithSpeciesId(ImageFolder):
    def __init__(self, root: str, transform=None, species_to_class: dict[int, int] | None = None):
        super().__init__(root, transform=transform)
        print(f"[data] ImageFolder scan: root={root}")
        print(f"[data]   found {len(self.classes)} subdirectories and {len(self.samples)} total images")
        if self.classes:
            print(f"[data]   first 5 subdir names: {self.classes[:5]}")
            print(f"[data]   last 5 subdir names:  {self.classes[-5:]}")
        if species_to_class is None:
            self._remap = None
            return

        new_samples: list[tuple[str, int]] = []
        n_skipped_nonint = 0
        n_skipped_unmapped = 0
        for path, local_idx in self.samples:
            dir_name = self.classes[local_idx]
            try:
                sid = int(dir_name)
            except ValueError:
                n_skipped_nonint += 1
                continue
            if sid not in species_to_class:
                n_skipped_unmapped += 1
                continue
            new_samples.append((path, species_to_class[sid]))

        self.samples = new_samples
        self.targets = [t for _, t in new_samples]
        self._remap = True
        print(f"[data] remap: kept={len(new_samples)}  "
              f"skipped_nonint_dirname={n_skipped_nonint}  "
              f"skipped_unmapped_species={n_skipped_unmapped}")

        if len(new_samples) == 0:
            raise RuntimeError(
                f"After remap, 0 samples remain under {root}. "
                f"Subdirectory names (first 5): {self.classes[:5]}."
            )


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
    mixup_alpha: float
    cutmix_alpha: float
    mixup_prob: float


def load_species_to_class(species_map_path: str) -> dict[int, int]:
    with open(species_map_path) as f:
        class_to_species = json.load(f)
    return {int(sid): idx for idx, sid in enumerate(class_to_species)}


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def maybe_subsample_per_class(samples, max_per_class, seed=42):
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


def load_phase2v3_backbone(
    backbone_path: str,
    cfg: TrainConfig,
    device: torch.device,
) -> nn.Module:
    """Load the timm ViT-L with phase2v3 EMA weights.

    The phase2v3 checkpoint is a timm .pth.tar file with a `state_dict_ema`
    key (preferred) and `state_dict` fallback. We load with `strict=False`
    because the saved checkpoint includes the classifier head dimensioned
    for 7806 classes — that survives the load — but doesn't include LoRA
    adapter keys (which we'll inject after).
    """
    print(f"[model] loading {MODEL_NAME} (no pretrained), then phase2v3 weights")
    backbone = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )

    print(f"[model] reading phase2v3 checkpoint: {backbone_path}")
    # weights_only=False because timm saves an argparse.Namespace alongside
    # the state dict, which PyTorch 2.6's secure default refuses to unpickle.
    # The checkpoint comes from our own training run, so trust is not an issue.
    ckpt = torch.load(backbone_path, map_location="cpu", weights_only=False)

    sd = None
    if isinstance(ckpt, dict):
        if "state_dict_ema" in ckpt:
            sd = ckpt["state_dict_ema"]
            print("[model]   using state_dict_ema (preferred)")
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            print("[model]   using state_dict (no EMA found)")
        else:
            sd = ckpt
            print("[model]   using top-level dict as state_dict")
    else:
        raise RuntimeError(f"Unexpected checkpoint format at {backbone_path}")

    # timm DDP/EMA state dicts may have 'module.' prefix or sufficient parallel-wrap keys
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    print(f"[model]   loaded weights — missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"[model]   first missing keys: {missing[:3]}")
    if unexpected[:3]:
        print(f"[model]   first unexpected keys: {unexpected[:3]}")

    return backbone.to(device)


def build_model(
    backbone_path: str,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[nn.Module, ArcFaceHead, SubCenterArcFace, list[str]]:
    backbone = load_phase2v3_backbone(backbone_path, cfg, device)

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
    if grad_checkpoint and not x.requires_grad:
        x = x.detach().requires_grad_(True)
    features = backbone.forward_features(x)
    cls = features[:, 0, :]
    return head(cls)


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy with soft targets. label_smoothing distributes a small
    mass uniformly across all classes (equivalent to PyTorch's CE smoothing)."""
    if label_smoothing > 0:
        n_class = logits.size(1)
        soft_targets = soft_targets * (1 - label_smoothing) + label_smoothing / n_class
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


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

    mixup_fn = None
    if cfg.mixup_alpha > 0 or cfg.cutmix_alpha > 0:
        mixup_fn = Mixup(
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            cutmix_minmax=None,
            prob=cfg.mixup_prob,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=cfg.label_smoothing,
            num_classes=NUM_CLASSES,
        )
        print(f"[mixup] enabled: mixup_alpha={cfg.mixup_alpha} cutmix_alpha={cfg.cutmix_alpha} "
              f"prob={cfg.mixup_prob}")
    else:
        print("[mixup] disabled")

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

            # Mixup/CutMix produces (mixed_imgs, soft_labels). Without mixup,
            # we keep the int labels for diagnostic metrics and one-hot them
            # below for the soft-CE path.
            if mixup_fn is not None:
                imgs_mix, soft_targets = mixup_fn(imgs, labels)
            else:
                imgs_mix = imgs
                soft_targets = F.one_hot(labels, NUM_CLASSES).to(imgs.dtype)
                if cfg.label_smoothing > 0:
                    soft_targets = (
                        soft_targets * (1 - cfg.label_smoothing)
                        + cfg.label_smoothing / NUM_CLASSES
                    )

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
                    backbone, head, imgs_mix, grad_checkpoint=cfg.grad_checkpoint
                )
                logits = arc(z, soft_targets, margin=cur_margin)
                # Mixup already applied label smoothing; for the no-mixup path
                # we already mixed it in above. Pass 0 here to avoid double-counting.
                loss = soft_cross_entropy(logits, soft_targets, label_smoothing=0.0)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            if step % cfg.log_every == 0:
                with torch.no_grad():
                    z_f = z.detach().float()
                    # Diagnostic metrics use the original int labels (pre-mixup)
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
        description="Exp-11C: LoRA+ArcFace+Mixup/CutMix on ViT-L (phase2v3) "
                    "for retrieval and classifier robustness"
    )
    ap.add_argument("--data-dir", required=True,
                    help="ImageFolder root (species_id subdirs).")
    ap.add_argument("--backbone-path", required=True,
                    help="Phase2v3 ViT-L .pth.tar (timm format with state_dict_ema).")
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
    ap.add_argument("--out-dim", type=int, default=DEFAULT_OUT_DIM)
    ap.add_argument("--label-smoothing", type=float, default=0.10)
    ap.add_argument("--image-size", type=int, default=518,
                    help="Must match deployment (518 for DINOv2 tiles).")
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lora-on-proj", action="store_true")
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-class", type=int, default=0)
    ap.add_argument("--mixup", type=float, default=0.8,
                    help="Mixup alpha (Beta distribution). 0 disables mixup.")
    ap.add_argument("--cutmix", type=float, default=1.0,
                    help="CutMix alpha. 0 disables cutmix.")
    ap.add_argument("--mixup-prob", type=float, default=1.0,
                    help="Probability of applying mixup OR cutmix per batch.")
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
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        mixup_prob=args.mixup_prob,
    )
    train(cfg, args)


if __name__ == "__main__":
    main()
