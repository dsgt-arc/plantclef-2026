"""Phase 4 — OVAT ablation sweep launcher.

One-variable-at-a-time ablation around the baseline config.
Each category sweeps a single axis while keeping all others at baseline.

Usage (called by SLURM array jobs):
    python phase4/ablation_sweep.py \
        --category loss --run-idx 0 \
        --dataset /path/to/pseudo_labels.parquet \
        --base-out-dir /path/to/checkpoints/phase4_ablations
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# ── Baseline config (anchor point for all ablations) ────────────────────
BASELINE = {
    "loss_type": "kl",
    "num_negatives": 64,
    "cons_mode": "logit_mse",
    "lambda_cons": 0.01,
    "warmup_epochs": 5,
    "tile_strategy": "none",
    "tile_k": 8,
    "tile_min_prob": 0.05,
    "tile_alpha": 0.3,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "batch_size": 32,
    "epochs": 15,
    "hidden_dim": 512,
    "num_heads": 8,
    "num_layers": 4,
    "dropout": 0.1,
    "embed_dropout": 0.1,
    "seed": 42,
    "save_every": 5,
    "num_workers": 4,
    "patience": 0,
    "min_delta": 1e-4,
    "num_classes": 7806,
    "min_image_conf": 0.0,
    "loss_topk": 0,
}

# ── Per-category configs: list of (run_name, overrides_dict) ────────────
# Index 0 in each category is the baseline for self-contained comparison.
ABLATIONS: dict[str, list[tuple[str, dict]]] = {
    "loss": [
        ("kl_baseline", {}),
        ("bce_neg64", {"loss_type": "bce"}),
    ],
    "tiles": [
        ("tile_none_baseline", {}),
        ("tile_fixedk6", {"tile_strategy": "fixed_k", "tile_k": 6}),
        ("tile_fixedk8", {"tile_strategy": "fixed_k", "tile_k": 8}),
        ("tile_thr005", {"tile_strategy": "threshold", "tile_min_prob": 0.05}),
        ("tile_rel030", {"tile_strategy": "relative", "tile_alpha": 0.3}),
    ],
    "consistency": [
        ("cons_off", {"lambda_cons": 0.0}),
        ("cons_logit_lam0.001", {"lambda_cons": 0.001}),
        ("cons_logit_lam0.01_baseline", {}),
        ("cons_logit_lam0.1", {"lambda_cons": 0.1}),
        ("cons_cosine_lam0.01", {"cons_mode": "cosine", "lambda_cons": 0.01}),
    ],
    "lr": [
        ("lr_3e-5", {"lr": 3e-5}),
        ("lr_1e-4_baseline", {}),
        ("lr_3e-4", {"lr": 3e-4}),
    ],
    "data_quality": [
        ("conf_0.00_baseline", {}),
        ("conf_0.04", {"min_image_conf": 0.04}),
        ("conf_0.05", {"min_image_conf": 0.05}),
        ("conf_0.07", {"min_image_conf": 0.07}),
    ],
    "loss_focus": [
        ("topk_0_baseline", {}),
        ("topk_32", {"loss_topk": 32}),
        ("topk_64", {"loss_topk": 64}),
        ("topk_128", {"loss_topk": 128}),
    ],
}


def get_config(category: str, run_idx: int) -> tuple[str, dict]:
    """Return (run_name, full_config) for the given category and index."""
    if category not in ABLATIONS:
        raise ValueError(
            f"Unknown category '{category}'. Choose from: {list(ABLATIONS.keys())}"
        )
    runs = ABLATIONS[category]
    if run_idx < 0 or run_idx >= len(runs):
        raise ValueError(
            f"run_idx {run_idx} out of range for category '{category}' "
            f"(0..{len(runs) - 1})"
        )
    run_name, overrides = runs[run_idx]
    config = {**BASELINE, **overrides}
    return run_name, config


def build_command(
    config: dict,
    dataset: str,
    out_dir: str,
    train_script: str,
    amp: bool = True,
) -> list[str]:
    """Build the train_student.py command from a config dict."""
    cmd = [
        sys.executable,
        train_script,
        "--dataset", dataset,
        "--out-dir", out_dir,
        "--loss-type", config["loss_type"],
        "--num-negatives", str(config["num_negatives"]),
        "--cons-mode", config["cons_mode"],
        "--lambda-cons", str(config["lambda_cons"]),
        "--warmup-epochs", str(config["warmup_epochs"]),
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--weight-decay", str(config["weight_decay"]),
        "--patience", str(config["patience"]),
        "--min-delta", str(config["min_delta"]),
        "--min-image-conf", str(config["min_image_conf"]),
        "--loss-topk", str(config["loss_topk"]),
        "--tile-strategy", config["tile_strategy"],
        "--tile-k", str(config["tile_k"]),
        "--tile-min-prob", str(config["tile_min_prob"]),
        "--tile-alpha", str(config["tile_alpha"]),
        "--hidden-dim", str(config["hidden_dim"]),
        "--num-heads", str(config["num_heads"]),
        "--num-layers", str(config["num_layers"]),
        "--num-classes", str(config["num_classes"]),
        "--dropout", str(config["dropout"]),
        "--embed-dropout", str(config["embed_dropout"]),
        "--seed", str(config["seed"]),
        "--num-workers", str(config["num_workers"]),
        "--save-every", str(config["save_every"]),
    ]
    if amp:
        cmd.append("--amp")
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 OVAT ablation launcher")
    ap.add_argument(
        "--category", required=True,
        choices=list(ABLATIONS.keys()),
        help="Ablation category to run",
    )
    ap.add_argument(
        "--run-idx", type=int, required=True,
        help="Run index within category (typically SLURM_ARRAY_TASK_ID)",
    )
    ap.add_argument("--dataset", required=True, help="Phase 3 pseudo-label parquet")
    ap.add_argument("--base-out-dir", required=True, help="Root checkpoint directory")
    ap.add_argument(
        "--train-script", default=None,
        help="Path to train_student.py (auto-detected if not set)",
    )
    ap.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    ap.add_argument("--dry-run", action="store_true", help="Print command without running")
    args = ap.parse_args()

    # Auto-detect train_student.py location
    if args.train_script is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.train_script = os.path.join(script_dir, "train_student.py")

    run_name, config = get_config(args.category, args.run_idx)
    out_dir = os.path.join(args.base_out_dir, f"ablation_{args.category}", run_name)

    print(f"{'=' * 60}")
    print(f"Ablation: {args.category} / {run_name} (idx={args.run_idx})")
    print(f"Output:   {out_dir}")
    print(f"{'=' * 60}")

    # Show what changed from baseline
    _, overrides = ABLATIONS[args.category][args.run_idx]
    if overrides:
        print("Overrides vs baseline:")
        for k, v in overrides.items():
            print(f"  {k}: {BASELINE[k]} -> {v}")
    else:
        print("(baseline config — no overrides)")
    print()

    cmd = build_command(
        config=config,
        dataset=args.dataset,
        out_dir=out_dir,
        train_script=args.train_script,
        amp=not args.no_amp,
    )

    print("Command:")
    print("  " + " \\\n    ".join(cmd))
    print()

    if args.dry_run:
        print("[DRY RUN] Exiting without execution.")
        return

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
