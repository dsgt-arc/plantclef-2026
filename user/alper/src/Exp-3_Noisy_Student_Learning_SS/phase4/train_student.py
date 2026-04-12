"""Phase 4 — Student training script.

Trains CrossRegionTransformer v4 on Phase 3 pseudo-labels with:
  - Pseudo-label loss (BCE neg sampling OR soft KL-div distillation)
  - Multi-view consistency loss with linear warmup
  - Configurable tile filtering strategy

Usage:
    python phase4/train_student.py \
        --dataset /path/to/lucas_pseudo_labels.parquet \
        --out-dir /path/to/checkpoints \
        --loss-type kl \
        --lambda-cons 0.1 \
        --warmup-epochs 5
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models import CrossRegionTransformer
from phase4.dataset import PseudoLabelDataset, TileFilterConfig, collate_fn
from phase4.losses import BCENegSamplingLoss, ConsistencyLoss, SoftDistillationLoss


def create_tile_drop_mask(
    mask: torch.Tensor, drop_rate: float = 0.3, min_tiles: int = 1,
) -> torch.Tensor:
    """Create a tile-drop augmentation mask by randomly hiding valid tiles.

    For each sample in the batch, randomly drops a fraction of valid tiles.
    Always keeps at least ``min_tiles`` tiles visible.
    """
    mask_drop = torch.zeros_like(mask)
    for b in range(mask.shape[0]):
        valid = mask[b].nonzero(as_tuple=True)[0]
        n_valid = len(valid)
        n_keep = max(min_tiles, int(n_valid * (1.0 - drop_rate)))
        keep = valid[torch.randperm(n_valid, device=mask.device)[:n_keep]]
        mask_drop[b, keep] = True
    return mask_drop


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args) -> CrossRegionTransformer:
    return CrossRegionTransformer(
        input_dim=768,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
        embed_dropout=args.embed_dropout,
    )


def build_pseudo_label_loss(args) -> nn.Module:
    if args.loss_type == "bce":
        return BCENegSamplingLoss(
            num_classes=args.num_classes,
            num_negatives=args.num_negatives,
        )
    return SoftDistillationLoss()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": {
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "num_classes": args.num_classes,
                "dropout": args.dropout,
                "embed_dropout": args.embed_dropout,
                "loss_type": args.loss_type,
                "cons_mode": args.cons_mode,
                "lambda_cons": args.lambda_cons,
                "min_image_conf": args.min_image_conf,
                "loss_topk": args.loss_topk,
                "cons_augment": args.cons_augment,
                "tile_drop_rate": args.tile_drop_rate,
            },
        },
        path,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    pseudo_loss_fn: nn.Module,
    cons_loss_fn: ConsistencyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    scaler: GradScaler | None = None,
) -> dict:
    """Run one training epoch. Returns dict of averaged metrics."""
    model.train()

    # Consistency warmup: linear ramp 0 → lambda_cons over warmup_epochs
    if args.warmup_epochs > 0:
        lambda_cons_t = min(1.0, epoch / args.warmup_epochs) * args.lambda_cons
    else:
        lambda_cons_t = args.lambda_cons

    total_loss = 0.0
    total_pseudo = 0.0
    total_cons = 0.0
    num_batches = 0

    cons_augment = getattr(args, "cons_augment", "dropout")
    drop_rate = getattr(args, "tile_drop_rate", 0.3)

    for batch in dataloader:
        tile_orig = batch["tile_embeddings_orig"].to(device)
        mask = batch["attention_mask"].to(device)
        teacher_probs = batch["teacher_probs"].to(device)
        pseudo_labels = batch["pseudo_labels"]

        use_amp = scaler is not None

        # Forward: orig (with gradients) for both losses
        with autocast(enabled=use_amp):
            logits_orig = model(tile_orig, mask)

        # Forward: consistency views (no gradients, but model stays in
        # train mode so dropout is stochastic)
        with torch.no_grad():
            with autocast(enabled=use_amp):
                if cons_augment == "dropout":
                    # R-Drop: two forward passes with different dropout masks
                    # Model is in train mode → embed_dropout + transformer
                    # dropout produce different activations each pass
                    logits_view1 = model(tile_orig, mask)
                    logits_view2 = model(tile_orig, mask)
                elif cons_augment == "tile_drop":
                    # Tile-drop: same embeddings, different random tile masks
                    mask_drop1 = create_tile_drop_mask(mask, drop_rate)
                    mask_drop2 = create_tile_drop_mask(mask, drop_rate)
                    logits_view1 = model(tile_orig, mask_drop1)
                    logits_view2 = model(tile_orig, mask_drop2)
                else:
                    # Legacy: separate flip/rot90 embedding views
                    tile_flip = batch["tile_embeddings_flip"].to(device)
                    tile_rot90 = batch["tile_embeddings_rot90"].to(device)
                    logits_view1 = model(tile_flip, mask)
                    logits_view2 = model(tile_rot90, mask)

        # Top-K class focusing
        topk_idx = None
        if args.loss_topk > 0:
            _, topk_idx = teacher_probs.topk(args.loss_topk, dim=1)

        # Pseudo-label loss
        with autocast(enabled=use_amp):
            if args.loss_type == "bce":
                l_pseudo = pseudo_loss_fn(logits_orig, teacher_probs)
            else:
                l_pseudo = pseudo_loss_fn(logits_orig, teacher_probs, topk_idx)

            # Consistency loss
            l_cons = cons_loss_fn(
                logits_orig, logits_view1.detach(), logits_view2.detach(), topk_idx
            )

            loss = l_pseudo + lambda_cons_t * l_cons

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_pseudo += l_pseudo.item()
        total_cons += l_cons.item()
        num_batches += 1

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "pseudo_loss": total_pseudo / n,
        "cons_loss": total_cons / n,
        "lambda_cons_t": lambda_cons_t,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4: Train student model")

    # Data
    ap.add_argument("--dataset", required=True, help="Phase 3 output parquet")
    ap.add_argument("--out-dir", required=True, help="Checkpoint output directory")

    # Loss
    ap.add_argument(
        "--loss-type", default="kl", choices=["bce", "kl"],
        help="Pseudo-label loss variant (default: kl)",
    )
    ap.add_argument("--num-negatives", type=int, default=256, help="BCE neg samples")
    ap.add_argument("--lambda-cons", type=float, default=0.01, help="Consistency weight")
    ap.add_argument("--warmup-epochs", type=int, default=5, help="Consistency warmup")
    ap.add_argument(
        "--cons-mode", default="logit_mse",
        choices=["logit_mse", "cosine", "prob_mse"],
        help="Consistency loss mode (default: logit_mse)",
    )
    ap.add_argument(
        "--cons-augment", default="dropout",
        choices=["dropout", "tile_drop", "views"],
        help="Consistency augmentation: 'dropout' (R-Drop, stochastic forward passes), "
             "'tile_drop' (random tile masking), or 'views' (flip/rot90 embeddings) "
             "(default: dropout)",
    )
    ap.add_argument(
        "--tile-drop-rate", type=float, default=0.3,
        help="Fraction of tiles to drop per consistency view (default: 0.3)",
    )

    # Training
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=0, help="Early stop patience (0=disabled)")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="Early stop min improvement")

    # Data quality
    ap.add_argument(
        "--min-image-conf", type=float, default=0.0,
        help="Min max(teacher_probs) to keep an image (0=no filter)",
    )
    ap.add_argument(
        "--loss-topk", type=int, default=0,
        help="Focus losses on top-K teacher classes per image (0=all classes)",
    )

    # Tile filtering
    ap.add_argument(
        "--tile-strategy", default="none",
        choices=["none", "fixed_k", "threshold", "relative"],
    )
    ap.add_argument("--tile-k", type=int, default=8)
    ap.add_argument("--tile-min-prob", type=float, default=0.05)
    ap.add_argument("--tile-alpha", type=float, default=0.3)

    # Model
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-classes", type=int, default=7806)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--embed-dropout", type=float, default=0.1)

    # System
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="Mixed precision training")
    ap.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Save config for reproducibility
    config_path = os.path.join(args.out_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset ---
    tile_filter = TileFilterConfig(
        strategy=args.tile_strategy,
        k=args.tile_k,
        min_prob=args.tile_min_prob,
        alpha=args.tile_alpha,
    )

    print(f"Loading dataset from {args.dataset} ...")
    dataset = PseudoLabelDataset(
        args.dataset, tile_filter=tile_filter, num_classes=args.num_classes,
        min_image_conf=args.min_image_conf,
    )
    print(f"Dataset: {len(dataset)} images")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # --- Model ---
    model = build_model(args).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {num_params:,} trainable parameters")

    # --- Loss ---
    pseudo_loss_fn = build_pseudo_label_loss(args)
    cons_loss_fn = ConsistencyLoss(mode=args.cons_mode)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # --- AMP ---
    scaler = GradScaler() if args.amp and device.type == "cuda" else None

    # --- Training loop ---
    print(f"\nTraining config:")
    print(f"  Loss type:       {args.loss_type}")
    print(f"  Cons mode:       {args.cons_mode}")
    print(f"  Lambda cons:     {args.lambda_cons}")
    print(f"  Warmup epochs:   {args.warmup_epochs}")
    print(f"  Min image conf:  {args.min_image_conf}")
    print(f"  Loss top-K:      {args.loss_topk}")
    print(f"  Cons augment:    {args.cons_augment}")
    print(f"  Tile drop rate:  {args.tile_drop_rate}")
    print(f"  Tile strategy:   {args.tile_strategy}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  LR:              {args.lr}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  AMP:             {args.amp}")
    print()

    best_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            pseudo_loss_fn=pseudo_loss_fn,
            cons_loss_fn=cons_loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            scaler=scaler,
        )

        elapsed = time.time() - t0
        metrics["epoch"] = epoch
        metrics["elapsed_sec"] = elapsed
        history.append(metrics)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={metrics['loss']:.6f} | "
            f"pseudo={metrics['pseudo_loss']:.6f} | "
            f"cons={metrics['cons_loss']:.6f} | "
            f"λ_cons={metrics['lambda_cons_t']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Save periodic checkpoint
        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, metrics["loss"], ckpt_path, args)

        # Save best checkpoint
        if metrics["loss"] < best_loss - args.min_delta:
            best_loss = metrics["loss"]
            patience_counter = 0
            best_path = os.path.join(args.out_dir, "best_model.pt")
            save_checkpoint(model, optimizer, epoch, best_loss, best_path, args)
            print(f"  -> New best loss: {best_loss:.6f}")
        else:
            patience_counter += 1

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    # Save final checkpoint
    final_path = os.path.join(args.out_dir, "final_model.pt")
    save_checkpoint(model, optimizer, epoch, metrics["loss"], final_path, args)

    # Save training history
    history_path = os.path.join(args.out_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete.")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Checkpoints: {args.out_dir}")


if __name__ == "__main__":
    main()
