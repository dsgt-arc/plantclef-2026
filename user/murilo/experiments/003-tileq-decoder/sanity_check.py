"""Sanity-check the TileQ-Decoder pipeline before launching long training runs.

Implements spec §4.3:
  1. Shape audit: trace one forward + backward pass and print every tensor shape.
  2. Overfit-one-batch (Sampler A): freeze 32 pseudo-quadrats, train for N steps,
     assert loss collapses by >= ``--overfit-ratio``.
  3. Overfit-one-batch (Sampler B, optional): same drill on Sampler B output if
     ``artifacts/sampler_b_lookup.npz`` exists. Confirms the test-distribution
     CLS scale (~25 vs train's ~18) doesn't break the training dynamics.

Exits 0 on success, non-zero on any failure. Designed to gate long training
runs: ``python sanity_check.py && sbatch sbatch/train_samplerA.sbatch``.

Usage:
  python sanity_check.py                                 # default: 200 steps
  python sanity_check.py --steps 500 --overfit-ratio 10  # tighter check
  python sanity_check.py --skip-sampler-b
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from plantclef.asl_loss import AsymmetricLoss
from plantclef.data.samplers import SamplerA, SamplerB
from plantclef.tileq_decoder import TileQDecoder


def shape_audit(model: TileQDecoder, batch_size: int = 2) -> bool:
    """Trace one forward + backward and print every intermediate tensor shape."""
    print("=" * 60)
    print("Shape audit")
    print("=" * 60)
    x = torch.randn(batch_size, 16, 768)
    print(f"  input tile_cls:    {tuple(x.shape)}")

    kv = model.proj_kv(x)
    print(f"  after proj_kv:     {tuple(kv.shape)}")

    q = model.queries.unsqueeze(0).expand(batch_size, -1, -1)
    print(f"  queries (expand):  {tuple(q.shape)}")

    attn_out, _ = model.cross_attn(q, kv, kv)
    print(f"  after cross_attn:  {tuple(attn_out.shape)}")

    h = model.norm1(q + attn_out)
    print(f"  after norm1:       {tuple(h.shape)}")

    ffn_out = model.ffn(h)
    print(f"  after ffn:         {tuple(ffn_out.shape)}")

    h = model.norm2(h + ffn_out)
    print(f"  after norm2:       {tuple(h.shape)}")

    group_logits = (
        torch.einsum("bgd,gdc->bgc", h, model.group_heads) + model.group_bias
    )
    print(f"  after group_heads: {tuple(group_logits.shape)}")

    out = group_logits.reshape(batch_size, -1)[:, : model.n_classes]
    print(f"  output logits:     {tuple(out.shape)}")

    out.sum().backward()
    grads = [(name, p.grad is not None) for name, p in model.named_parameters()]
    n_with_grad = sum(1 for _, has in grads if has)
    print(f"  backward: {n_with_grad}/{len(grads)} parameters received gradient")

    expected = (batch_size, model.n_classes)
    ok = (
        out.shape == expected
        and n_with_grad == len(grads)
        and not torch.isnan(out).any()
    )
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    return ok


def overfit_one_batch(
    sampler,
    sampler_name: str,
    *,
    batch_size: int,
    steps: int,
    lr: float,
    overfit_ratio: float,
    seed: int,
) -> bool:
    """Freeze a batch of `batch_size` pseudo-quadrats, train for `steps` steps,
    assert the final loss is at least `overfit_ratio` smaller than the initial.
    """
    print()
    print("=" * 60)
    print(f"Overfit-one-batch ({sampler_name}, batch={batch_size}, steps={steps})")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"  sampling {batch_size} pseudo-quadrats...")
    tiles_list, labels_list = [], []
    for _ in range(batch_size):
        t, lbl = sampler.sample()
        tiles_list.append(t)
        labels_list.append(lbl)
    tiles = torch.from_numpy(np.stack(tiles_list))
    labels = torch.from_numpy(np.stack(labels_list))
    print(
        f"  fixed tiles {tuple(tiles.shape)} labels {tuple(labels.shape)} "
        f"avg label_count={labels.sum(-1).mean():.2f}"
    )

    model = TileQDecoder()
    loss_fn = AsymmetricLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    losses = []
    model.train()
    t_start = time.time()
    log_every = max(1, steps // 10)
    for step in range(steps):
        optimizer.zero_grad()
        logits = model(tiles)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        if step % log_every == 0 or step == steps - 1:
            print(f"    step {step:>4}: loss={loss.item():.4f}")

    elapsed = time.time() - t_start
    initial = losses[0]
    final = losses[-1]
    ratio = initial / max(final, 1e-9)
    print(
        f"\n  initial={initial:.4f}  final={final:.4f}  "
        f"ratio={ratio:.1f}x  ({elapsed:.1f}s, {1000*elapsed/steps:.0f} ms/step)"
    )
    ok = ratio >= overfit_ratio and not np.isnan(final)
    print(f"  required ratio >= {overfit_ratio:.1f}x")
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--overfit-ratio", type=float, default=5.0,
                   help="Required initial/final loss ratio (default 5x).")
    p.add_argument("--seed", type=int, default=42)
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
    p.add_argument("--skip-sampler-b", action="store_true",
                   help="Skip Sampler B overfit check.")
    args = p.parse_args()

    print(f"Loading {args.train_lookup}...")
    train = np.load(args.train_lookup)
    weights = np.load(args.species_weights)
    print(
        f"  embeddings={train['embeddings'].shape} "
        f"weights={weights.shape} sum={weights.sum():.6f}\n"
    )

    model = TileQDecoder()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TileQDecoder: {n_params:,} parameters\n")

    audit_ok = shape_audit(model)
    if not audit_ok:
        print("\nFAIL: shape audit"); return 1

    sampler_a = SamplerA(
        embeddings=train["embeddings"],
        offsets=train["offsets"],
        species_weights=weights,
        rng=np.random.default_rng(args.seed),
    )
    a_ok = overfit_one_batch(
        sampler_a, "Sampler A",
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        overfit_ratio=args.overfit_ratio,
        seed=args.seed,
    )
    if not a_ok:
        print("\nFAIL: Sampler A overfit"); return 1

    if not args.skip_sampler_b:
        if not args.sampler_b_lookup.exists():
            print(f"\n[skip] Sampler B check: {args.sampler_b_lookup} not found")
        else:
            sb_data = np.load(args.sampler_b_lookup, allow_pickle=False)
            sampler_b = SamplerB(
                embeddings=train["embeddings"],
                offsets=train["offsets"],
                species_weights=weights,
                test_cls=sb_data["test_cls"],
                tile_to_class_idx=sb_data["tile_to_class_idx"],
                rng=np.random.default_rng(args.seed + 1),
            )
            b_ok = overfit_one_batch(
                sampler_b, f"Sampler B (tau={float(sb_data['threshold']):.2f})",
                batch_size=args.batch_size,
                steps=args.steps,
                lr=args.lr,
                overfit_ratio=args.overfit_ratio,
                seed=args.seed,
            )
            if not b_ok:
                print("\nFAIL: Sampler B overfit"); return 1

    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS PASSED.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
