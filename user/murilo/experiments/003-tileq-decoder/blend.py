"""A3 score-blend: alpha-mix decoder probs with cached max_prob baseline.

Per spec §2.5:
    s_c = alpha * decoder_prob_c + (1 - alpha) * max_prob_c

For each ``--alpha`` in the comma-list, emit one Kaggle CSV at the chosen
top-k. ``alpha=0`` is a sanity duplicate of A0a (max_prob top-10);
``alpha=1`` is a sanity duplicate of the chosen decoder source at the
same k.

No retraining — pure NumPy. Reuses ``probs_<tag>_seed{seed}.npy`` produced
by ``infer.py`` and the per-tile cached probs at ``grid_4x4.parquet``.

Usage:
  python blend.py \
      --decoder-probs artifacts/probs_A2_samplerB_seed42.npy \
      --decoder-tag A2_samplerB \
      --alpha 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
      --topk 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Reuse infer.py's CSV writer + class-mapping loader so the format is
# byte-identical to A1/A2 submissions.
sys.path.insert(0, str(Path(__file__).parent))
from infer import load_class_to_species_id, load_test_cls, write_submission_csv  # noqa: E402


def load_max_prob_per_image(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load grid_4x4.parquet and compute per-image max-prob across the 16 tiles.

    Returns
    -------
    image_names : (n_q,) <U64, sorted alphabetically (matches infer.load_test_cls)
    max_prob    : (n_q, 7806) float32 — per-class max across the 16 tiles per image
    """
    table = pq.read_table(parquet_path)
    n_rows = table.num_rows
    names = np.asarray(table.column("image_name").to_pylist(), dtype="<U64")
    tile_idxs = table.column("tile_idx").to_numpy()
    blobs = table.column("probs").to_pylist()
    blob_size = len(blobs[0])
    if blob_size % 2 != 0:
        raise ValueError(f"probs blob size {blob_size} not divisible by 2")
    n_classes = blob_size // 2
    flat = np.frombuffer(b"".join(blobs), dtype=np.float16).reshape(n_rows, n_classes)
    flat = flat.astype(np.float32)

    order = np.lexsort((tile_idxs, names))
    names = names[order]
    flat = flat[order]

    image_names = np.unique(names)
    n_q = len(image_names)
    n_tiles = n_rows // n_q
    if n_q * n_tiles != n_rows:
        raise ValueError(f"{n_q} * {n_tiles} != {n_rows}")

    per_tile = flat.reshape(n_q, n_tiles, n_classes)
    max_prob = per_tile.max(axis=1)  # (n_q, n_classes)
    return image_names, max_prob


def topk_decode(probs: np.ndarray, k: int) -> np.ndarray:
    """For each row, return the indices of the top-k columns sorted descending."""
    n, c = probs.shape
    idx = np.argpartition(-probs, k - 1, axis=1)[:, :k]
    for i in range(n):
        order = np.argsort(-probs[i, idx[i]])
        idx[i] = idx[i][order]
    return idx


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decoder-probs", type=Path, required=True,
                   help="probs_<tag>_seed{seed}.npy from infer.py")
    p.add_argument("--decoder-tag", type=str, required=True,
                   help="Tag prefix for output CSVs, e.g. 'A3_samplerB'")
    p.add_argument(
        "--max-prob-parquet",
        type=Path,
        default=Path(
            "/storage/project/ps-clef2026_plantclef-0/shared/inference/"
            "dinov2-vit-base/grid_4x4.parquet"
        ),
        help="Per-tile probs parquet (image_name, tile_idx, probs[7806] f16).",
    )
    p.add_argument(
        "--test-parquet",
        type=Path,
        default=Path(
            "/storage/project/ps-clef2026_plantclef-0/shared/inference/"
            "dinov2-vit-base/cls_grid_4x4.parquet"
        ),
        help="Used only to recover image_names in the canonical sort order.",
    )
    p.add_argument(
        "--class-mapping",
        type=Path,
        default=Path(
            "/storage/project/ps-clef2026_plantclef-0/shared/data/raw/"
            "class_id_to_species_mapping/class_mapping.txt"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "submissions",
    )
    p.add_argument("--alpha", type=str,
                   default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    alphas = [float(a) for a in args.alpha.split(",") if a.strip()]
    if not alphas:
        print("FATAL: --alpha empty", file=sys.stderr); return 1
    if any(a < 0 or a > 1 for a in alphas):
        print(f"FATAL: alphas out of [0, 1]: {alphas}", file=sys.stderr); return 1

    print(f"Loading decoder probs: {args.decoder_probs}")
    decoder_probs = np.load(args.decoder_probs)
    n_q_d, n_c = decoder_probs.shape
    print(f"  decoder_probs shape={decoder_probs.shape} dtype={decoder_probs.dtype}")

    print(f"Loading max_prob baseline: {args.max_prob_parquet}")
    t0 = time.time()
    image_names_mp, max_prob = load_max_prob_per_image(args.max_prob_parquet)
    print(f"  image_names={len(image_names_mp)} max_prob shape={max_prob.shape} "
          f"({time.time()-t0:.1f}s)")

    # Sanity: image-name ordering must match what infer.py used
    print(f"Loading test_cls image_names for ordering check: {args.test_parquet}")
    image_names_dec, _ = load_test_cls(args.test_parquet)
    if not np.array_equal(image_names_mp, image_names_dec):
        print("FATAL: image_name order disagrees between max_prob parquet and "
              "decoder probs (test parquet). Aborting to avoid mis-aligned blend.",
              file=sys.stderr)
        return 1
    if max_prob.shape[1] != n_c:
        print(f"FATAL: class count mismatch: decoder={n_c} max_prob={max_prob.shape[1]}",
              file=sys.stderr); return 1
    if max_prob.shape[0] != n_q_d:
        print(f"FATAL: image count mismatch: decoder={n_q_d} max_prob={max_prob.shape[0]}",
              file=sys.stderr); return 1

    print(f"Loading class mapping: {args.class_mapping}")
    class_to_species_id = load_class_to_species_id(args.class_mapping)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nBlending and writing CSVs (k={args.topk})...")
    for alpha in alphas:
        blended = alpha * decoder_probs + (1.0 - alpha) * max_prob
        idx = topk_decode(blended, args.topk)
        species_id_lists = [class_to_species_id[idx[i]].tolist() for i in range(n_q_d)]
        alpha_tag = f"{alpha:.2f}".replace(".", "p")
        out_path = (
            args.output_dir / f"{args.decoder_tag}_alpha{alpha_tag}_topk{args.topk}_seed{args.seed}.csv"
        )
        write_submission_csv(out_path, image_names_mp, species_id_lists)
        avg = sum(len(s) for s in species_id_lists) / len(species_id_lists)
        print(f"  alpha={alpha:.2f} -> {out_path.name}  (avg species/img = {avg:.1f})")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
