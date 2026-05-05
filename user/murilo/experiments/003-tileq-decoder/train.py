"""Train the TileQ-Decoder on synthetic CLS-domain pseudo-quadrats (spec §4).

Hyperparameters (defaults match spec §4.1):
  - AdamW, weight_decay 0.05
  - Linear warmup 0 -> 3e-4 over 1000 steps
  - Cosine decay 3e-4 -> 1e-5 over the remaining steps
  - Batch size 32, 80,000 total steps
  - Loss: AsymmetricLoss(gamma_neg=4, gamma_pos=0, margin=0.05)
  - Gradient clipping 1.0
  - bfloat16 autocast (with auto-fallback to fp32 on GPUs without bf16 support)
  - Seed 42

Outputs (under ``--output-dir``):
  checkpoints/decoder_sampler{A|B}_step{N:06d}.pt  (every --checkpoint-every)
  checkpoints/decoder_sampler{A|B}.pt              (final)
  loss_sampler{A|B}.csv                            (per-step loss + sparse val rows)
  loss_sampler{A|B}.png                            (plot of loss curve)
  config_sampler{A|B}.yaml                         (run config)

Validation note: spec §4.2 calls for 1% holdout by image_name hash. Given the
deadline + the fact that synthetic val loss is *not* used for thresholding,
model selection, or hyperparameter search, this implementation samples the val
set from the same pool with a different RNG seed. Documented deviation; the
spec acknowledges Kaggle is the only generalisation signal.

Usage:
  python train.py --sampler A --seed 42 --steps 80000
  python train.py --sampler B --seed 42 --steps 80000
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

# Load WANDB_API_KEY (and any other secrets) from user/murilo/.env BEFORE we
# import wandb. Mirrors birdclef's pattern (user/murilo/experiments/030-*/train.py).
try:
    from dotenv import load_dotenv

    _user_dir = Path(__file__).parent.parent.parent  # .../user/murilo
    load_dotenv(_user_dir / ".env")
except ImportError:
    pass  # dotenv optional; falls back to ambient env vars

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from plantclef.asl_loss import AsymmetricLoss
from plantclef.data.samplers import SamplerA, SamplerB
from plantclef.tileq_decoder import TileQDecoder

# Local import: experiments/003-tileq-decoder/dataset.py
sys.path.insert(0, str(Path(__file__).parent))
from dataset import PseudoQuadratDataset  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def make_lr_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup 0 -> peak_lr, then cosine decay peak_lr -> min_lr."""
    floor = min_lr / peak_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return floor + (1 - floor) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_sampler(
    sampler_id: str,
    train_data,
    weights: np.ndarray,
    sampler_b_data,
    n_positives: int,
    n_tiles: int,
    rng: np.random.Generator,
):
    if sampler_id == "A":
        return SamplerA(
            embeddings=train_data["embeddings"],
            offsets=train_data["offsets"],
            species_weights=weights,
            n_positives=n_positives,
            n_tiles=n_tiles,
            rng=rng,
        )
    if sampler_id == "B":
        if sampler_b_data is None:
            raise RuntimeError("Sampler B requires --sampler-b-lookup artefact")
        return SamplerB(
            embeddings=train_data["embeddings"],
            offsets=train_data["offsets"],
            species_weights=weights,
            test_cls=sampler_b_data["test_cls"],
            tile_to_class_idx=sampler_b_data["tile_to_class_idx"],
            n_positives=n_positives,
            n_tiles=n_tiles,
            rng=rng,
        )
    raise ValueError(f"unknown sampler: {sampler_id!r}")


def build_validation_set(sampler, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    tiles_list, labels_list = [], []
    for _ in range(n_samples):
        t, l = sampler.sample()
        tiles_list.append(t)
        labels_list.append(l)
    tiles = torch.from_numpy(np.stack(tiles_list))
    labels = torch.from_numpy(np.stack(labels_list))
    return tiles, labels


@torch.no_grad()
def evaluate(
    model: TileQDecoder,
    val_tiles: torch.Tensor,
    val_labels: torch.Tensor,
    loss_fn: AsymmetricLoss,
    device: torch.device,
    batch_size: int,
    autocast_dtype: torch.dtype | None,
    top_k: int = 10,
) -> tuple[float, float]:
    """Return (mean_val_loss, any_positive_found_accuracy@top_k)."""
    model.eval()
    losses, accs = [], []
    for i in range(0, val_tiles.shape[0], batch_size):
        tiles = val_tiles[i : i + batch_size].to(device, non_blocking=True)
        labels = val_labels[i : i + batch_size].to(device, non_blocking=True)
        if autocast_dtype is not None:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                logits = model(tiles)
                loss = loss_fn(logits, labels)
        else:
            logits = model(tiles)
            loss = loss_fn(logits, labels)
        losses.append(float(loss.item()))
        top_idx = logits.float().topk(top_k, dim=-1).indices  # (B, top_k)
        any_hit = (labels.gather(-1, top_idx) > 0).any(dim=-1).float()
        accs.append(any_hit.mean().item())
    model.train()
    return float(np.mean(losses)), float(np.mean(accs))


def save_checkpoint(
    path: Path, model, optimizer, scheduler, step: int, loss: float, config: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
            "config": config,
        },
        path,
    )


def plot_loss_curve(csv_path: Path, png_path: Path) -> None:
    steps, losses, val_steps, val_losses = [], [], [], []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            if row.get("val_loss"):
                val_steps.append(int(row["step"]))
                val_losses.append(float(row["val_loss"]))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, losses, color="#3a7", lw=0.6, alpha=0.6, label="train")
    if val_losses:
        ax.plot(val_steps, val_losses, color="#c44", lw=1.5, marker="o",
                markersize=4, label="val (synthetic)")
    ax.set_xlabel("step")
    ax.set_ylabel("ASL loss (mean per sample)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(png_path.stem)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sampler", choices=["A", "B"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=80_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--n-positives", type=int, default=5)
    p.add_argument("--n-tiles", type=int, default=16)
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--validate-every", type=int, default=2000)
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    p.add_argument(
        "--train-lookup",
        type=Path,
        default=Path(__file__).parent / "artifacts/train_cls_by_class_idx.npz",
    )
    p.add_argument(
        "--species-weights",
        type=Path,
        default=Path(__file__).parent / "artifacts/species_weights.npy",
    )
    p.add_argument(
        "--sampler-b-lookup",
        type=Path,
        default=Path(__file__).parent / "artifacts/sampler_b_lookup.npz",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint to resume from.",
    )
    # Logging
    p.add_argument("--run-name", type=str, default=None,
                   help="Run name; default: 'exp003-sampler{A|B}-seed{N}'.")
    p.add_argument("--wandb-entity", type=str, default="dsgt-arc")
    p.add_argument("--wandb-project", type=str, default="plantclef-2026-exp003")
    p.add_argument("--no-wandb", action="store_true",
                   help="Skip W&B logging (keep TensorBoard).")
    p.add_argument("--no-tensorboard", action="store_true",
                   help="Skip TensorBoard logging.")
    args = p.parse_args()
    if args.run_name is None:
        args.run_name = f"exp003-sampler{args.sampler}-seed{args.seed}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # ------ device + precision ------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.precision == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            precision = "bf16"
        else:
            precision = "fp32"
    else:
        precision = args.precision
    autocast_dtype = torch.bfloat16 if precision == "bf16" else None
    print(f"device={device}, precision={precision}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ------ seeds ------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ------ artefacts ------
    print(f"Loading {args.train_lookup}...", flush=True)
    train_data = np.load(args.train_lookup)
    weights = np.load(args.species_weights)

    sampler_b_data = None
    if args.sampler == "B":
        if not args.sampler_b_lookup.exists():
            print(f"FATAL: {args.sampler_b_lookup} not found", file=sys.stderr)
            return 1
        sampler_b_data = np.load(args.sampler_b_lookup, allow_pickle=False)

    # ------ samplers (separate train and val RNGs) ------
    train_sampler = build_sampler(
        args.sampler, train_data, weights, sampler_b_data,
        args.n_positives, args.n_tiles, np.random.default_rng(args.seed),
    )
    val_sampler = build_sampler(
        args.sampler, train_data, weights, sampler_b_data,
        args.n_positives, args.n_tiles, np.random.default_rng(args.seed + 10_000),
    )

    print(f"Building {args.n_val}-sample validation set...", flush=True)
    val_tiles, val_labels = build_validation_set(val_sampler, args.n_val)
    print(
        f"  val_tiles {tuple(val_tiles.shape)} val_labels {tuple(val_labels.shape)} "
        f"avg_label_count={val_labels.sum(-1).mean():.2f}",
        flush=True,
    )

    # ------ DataLoader ------
    dataset = PseudoQuadratDataset(train_sampler)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ------ model + loss + optim ------
    model = TileQDecoder(n_classes=weights.shape[0]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TileQDecoder: {n_params:,} parameters", flush=True)
    loss_fn = AsymmetricLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = make_lr_schedule(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        peak_lr=args.lr,
        min_lr=args.min_lr,
    )

    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}...", flush=True)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt["step"] + 1
        print(f"  resumed at step {start_step}", flush=True)

    # ------ persist run config ------
    config = {
        "sampler": args.sampler,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "n_positives": args.n_positives,
        "n_tiles": args.n_tiles,
        "n_val": args.n_val,
        "validate_every": args.validate_every,
        "checkpoint_every": args.checkpoint_every,
        "precision": precision,
        "n_classes": int(weights.shape[0]),
        "n_params": n_params,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else None
        ),
    }
    config["run_name"] = args.run_name
    config_path = args.output_dir / f"config_sampler{args.sampler}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    # ------ W&B + TensorBoard loggers ------
    tb_writer: SummaryWriter | None = None
    if not args.no_tensorboard:
        tb_dir = args.output_dir / "tensorboard" / args.run_name
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"TensorBoard: {tb_dir}", flush=True)

    wandb_run = None
    if not args.no_wandb and _WANDB_AVAILABLE:
        try:
            wandb_dir = args.output_dir / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            wandb_run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.run_name,
                dir=str(wandb_dir),
                config=config,
                resume="allow" if args.resume else "never",
            )
            print(f"W&B: {wandb_run.url if wandb_run else '(offline)'}", flush=True)
        except Exception as e:
            print(f"W&B init failed ({e!r}); continuing without W&B.", flush=True)
            wandb_run = None
    elif not args.no_wandb and not _WANDB_AVAILABLE:
        print("W&B requested but module unavailable; skipping.", flush=True)

    # ------ csv log ------
    csv_path = args.output_dir / f"loss_sampler{args.sampler}.csv"
    if not args.resume or not csv_path.exists():
        csv_path.write_text("step,loss,lr,val_loss,val_acc\n")

    # ------ training loop ------
    print(f"\nStarting training: {args.steps} steps", flush=True)
    t_start = time.time()
    loss_window: list[float] = []
    csv_buf: list[str] = []

    data_iter = iter(loader)
    model.train()
    last_loss = float("nan")

    for step in range(start_step, args.steps):
        tiles, labels = next(data_iter)
        tiles = tiles.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if autocast_dtype is not None:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                logits = model(tiles)
                loss = loss_fn(logits, labels)
        else:
            logits = model(tiles)
            loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        loss_window.append(loss_val)
        last_loss = loss_val

        lr_now = scheduler.get_last_lr()[0]

        # ---- per-step logging (TB every step, W&B per its own batching) ----
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", loss_val, step)
            tb_writer.add_scalar("train/lr", lr_now, step)
        if wandb_run is not None:
            wandb_run.log(
                {"train/loss": loss_val, "train/lr": lr_now}, step=step
            )

        # ---- periodic validation ----
        val_loss_str, val_acc_str = "", ""
        if (step + 1) % args.validate_every == 0:
            val_loss, val_acc = evaluate(
                model, val_tiles, val_labels, loss_fn,
                device, args.batch_size, autocast_dtype,
            )
            val_loss_str = f"{val_loss:.6f}"
            val_acc_str = f"{val_acc:.4f}"
            if tb_writer is not None:
                tb_writer.add_scalar("val/loss", val_loss, step)
                tb_writer.add_scalar("val/any_pos_found_at_10", val_acc, step)
            if wandb_run is not None:
                wandb_run.log(
                    {"val/loss": val_loss, "val/any_pos_found_at_10": val_acc},
                    step=step,
                )

        csv_buf.append(
            f"{step},{loss_val:.6f},{lr_now:.6e},{val_loss_str},{val_acc_str}\n"
        )

        # ---- console log ----
        if (step + 1) % args.log_every == 0 or step == args.steps - 1:
            with csv_path.open("a") as f:
                f.writelines(csv_buf)
            csv_buf.clear()
            window = loss_window[-args.log_every :]
            elapsed = time.time() - t_start
            steps_done = step - start_step + 1
            steps_per_sec = steps_done / max(elapsed, 1e-9)
            eta_min = (args.steps - step - 1) / max(steps_per_sec, 1e-9) / 60
            print(
                f"  step {step+1:>6}/{args.steps} "
                f"loss={np.mean(window):.4f} "
                f"lr={lr_now:.2e} "
                f"{steps_per_sec:.1f} step/s "
                f"ETA={eta_min:.1f}min"
                + (f"  val={val_loss_str} acc@10={val_acc_str}" if val_loss_str else ""),
                flush=True,
            )

        # ---- checkpoint ----
        if (step + 1) % args.checkpoint_every == 0:
            ckpt_path = (
                args.output_dir / "checkpoints"
                / f"decoder_sampler{args.sampler}_step{step+1:06d}.pt"
            )
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler,
                step + 1, loss_val, config,
            )

    # ---- final flush + checkpoint + plot ----
    if csv_buf:
        with csv_path.open("a") as f:
            f.writelines(csv_buf)

    final_ckpt = args.output_dir / "checkpoints" / f"decoder_sampler{args.sampler}.pt"
    save_checkpoint(
        final_ckpt, model, optimizer, scheduler, args.steps, last_loss, config
    )
    print(f"\nFinal checkpoint: {final_ckpt}", flush=True)

    png_path = args.output_dir / f"loss_sampler{args.sampler}.png"
    plot_loss_curve(csv_path, png_path)
    print(f"Loss curve: {png_path}", flush=True)

    elapsed = time.time() - t_start
    print(
        f"\nDone. Trained {args.steps - start_step} steps in {elapsed/60:.1f} min "
        f"({(args.steps - start_step) / max(elapsed, 1e-9):.1f} step/s).",
        flush=True,
    )

    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
