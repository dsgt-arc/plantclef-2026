"""Export a timm training checkpoint to a standalone model weights file.

timm checkpoints (.pth.tar) contain model state_dict, optimizer state,
epoch, etc. This script extracts just the model weights (optionally EMA)
for use in the inference pipeline.

Usage:
  python scripts/export_checkpoint.py \
    --checkpoint /path/to/model_best.pth.tar \
    --output /path/to/convnext_base_plantclef.pth \
    --use-ema
"""
import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser(description="Export timm checkpoint to model weights")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to timm checkpoint (.pth.tar)")
    ap.add_argument("--output", required=True,
                    help="Output path for model weights (.pth)")
    ap.add_argument("--use-ema", action="store_true",
                    help="Use EMA weights if available (recommended)")
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Show checkpoint contents
    if isinstance(ckpt, dict):
        print(f"  Keys: {list(ckpt.keys())}")
        if "epoch" in ckpt:
            print(f"  Epoch: {ckpt['epoch']}")
        if "metric" in ckpt:
            print(f"  Best metric: {ckpt['metric']}")

    # Extract state dict
    if args.use_ema and isinstance(ckpt, dict) and "state_dict_ema" in ckpt:
        state_dict = ckpt["state_dict_ema"]
        print("  Using EMA weights")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print("  Using regular weights")
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        print("  Using 'model' key weights")
    else:
        state_dict = ckpt
        print("  Using raw state dict")

    # Count parameters
    num_params = sum(v.numel() for v in state_dict.values())
    print(f"  Parameters: {num_params:,}")
    print(f"  Keys: {len(state_dict)}")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(state_dict, args.output)

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nSaved to: {args.output} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()
