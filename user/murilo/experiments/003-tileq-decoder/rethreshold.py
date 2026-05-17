"""Re-apply a new tau to the cached FAISS results — no re-query.

Loads ``artifacts/sampler_b_lookup.npz``, applies ``--threshold`` to the cached
``top1_sim`` + ``top1_class_idx`` arrays, rewrites the npz with the updated
``tile_to_class_idx`` + ``threshold`` field, and refreshes the histogram +
``faiss_threshold.txt`` summary in place.

Usage:
  python rethreshold.py --threshold 0.65
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from build_faiss_background_map import apply_threshold, plot_histogram, summarize


def main(npz_path: Path, output_dir: Path, threshold: float) -> None:
    data = np.load(npz_path, allow_pickle=False)
    image_names = data["image_names"]
    test_cls = data["test_cls"]
    top1_sim = data["top1_sim"]
    top1_class_idx = data["top1_class_idx"]

    tile_to_class_idx = apply_threshold(top1_sim, top1_class_idx, threshold)

    sim_flat = top1_sim.reshape(-1)
    summary = summarize(sim_flat, threshold)
    print(summary)

    plot_histogram(sim_flat, output_dir / "tile_top1_sim_histogram.png")
    (output_dir / "faiss_threshold.txt").write_text(summary + "\n")

    np.savez(
        npz_path,
        image_names=image_names,
        test_cls=test_cls,
        top1_sim=top1_sim,
        top1_class_idx=top1_class_idx,
        tile_to_class_idx=tile_to_class_idx,
        threshold=np.float32(threshold),
    )
    n_labelled = int((tile_to_class_idx >= 0).sum())
    total = int(tile_to_class_idx.size)
    print(
        f"\nUpdated {npz_path}: tau={threshold:.4f} -> "
        f"{n_labelled}/{total} ({100*n_labelled/total:.2f}%) tiles labelled."
    )


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz-path",
        type=Path,
        default=Path(__file__).parent / "artifacts/sampler_b_lookup.npz",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts",
    )
    p.add_argument("--threshold", type=float, required=True)
    args = p.parse_args()
    main(args.npz_path, args.output_dir, args.threshold)


if __name__ == "__main__":
    cli()
