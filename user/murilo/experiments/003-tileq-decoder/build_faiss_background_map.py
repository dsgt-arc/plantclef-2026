"""Build the FAISS background-cleaning map for Sampler B (spec §3.4).

Pipeline:
  1. Load train CLS lookup (artifacts/train_cls_by_class_idx.npz) and recover
     the class_idx of every train row from the contiguous offsets layout.
  2. L2-normalise all 1.4 M train CLS rows (in place) and build
     ``faiss.IndexFlatIP``.
  3. Load cls_grid_4x4.parquet (33,680 rows = 2,105 imgs * 16 tiles), decode
     binary float16 -> float32, sort by (image_name, tile_idx), reshape to
     ``(2105, 16, 768)``.
  4. L2-normalise test tiles and query top-1 nearest train neighbour per tile.
  5. Map train row index -> class_idx via searchsorted on the train offsets.
  6. Build the per-tile species map: ``tile_to_class_idx[q, t] = class_idx if
     sim[q, t] >= threshold else -1``.
  7. Persist all artefacts in a single ``artifacts/sampler_b_lookup.npz``
     plus a histogram PNG and a threshold/summary text file.

Note (naming): the spec calls this map ``tile_to_species`` keyed by Pl@ntNet
species_id. We standardised on class_idx everywhere internally — see the
build_train_lookup decision. The species_id is recoverable via
``class_to_species_id[class_idx]`` (persisted in the train npz).

Run:
  bash sbatch/build_faiss_background_map.sh
or:
  python build_faiss_background_map.py --threshold 0.85
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import faiss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def l2_normalize_rows(arr: np.ndarray, eps: float = 1e-8) -> None:
    """In-place L2-normalisation along the last axis. Float32 only."""
    if arr.dtype != np.float32:
        raise ValueError(f"l2_normalize_rows requires float32, got {arr.dtype}")
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    np.maximum(norms, eps, out=norms)
    arr /= norms


def train_row_to_class_idx(
    train_rows: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Map FAISS train-row indices to their class_idx via searchsorted on offsets.

    train_emb is sorted by class_idx with ``embeddings[offsets[c]:offsets[c+1]]``
    holding class c's CLSs, so for row r, class_idx = position of r in offsets
    (right-side searchsorted minus 1).
    """
    return (np.searchsorted(offsets, train_rows, side="right") - 1).astype(np.int32)


def decode_test_parquet(
    test_parquet_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Read cls_grid_4x4.parquet -> (image_names, test_cls).

    Returns
    -------
    image_names : (n_quadrats,) <U64
    test_cls    : (n_quadrats, n_tiles=16, 768) float32, raw (not L2-normed)
    """
    table = pq.read_table(test_parquet_path)
    n_rows = table.num_rows
    if n_rows == 0:
        raise RuntimeError("test parquet is empty")

    names = np.asarray(table.column("image_name").to_pylist(), dtype="<U64")
    tile_idxs = table.column("tile_idx").to_numpy()
    cls_blobs = table.column("cls_embedding").to_pylist()

    # Decode all (binary float16[D]) blobs at once. D is derived from the
    # blob byte length to keep this function testable at small dims.
    blob_size = len(cls_blobs[0])
    if any(len(b) != blob_size for b in cls_blobs):
        raise ValueError("cls_embedding blobs have inconsistent byte length")
    if blob_size % 2 != 0:
        raise ValueError(
            f"blob size {blob_size} is not divisible by 2 (float16 expected)"
        )
    dim = blob_size // 2
    flat_f16 = np.frombuffer(b"".join(cls_blobs), dtype=np.float16).reshape(
        n_rows, dim
    )
    flat = flat_f16.astype(np.float32)

    # Sort by (image_name, tile_idx). np.lexsort takes the LAST key as primary,
    # so passing (tile_idxs, names) gives primary=names, secondary=tile_idxs.
    order = np.lexsort((tile_idxs, names))
    names = names[order]
    tile_idxs = tile_idxs[order]
    flat = flat[order]

    unique_names, name_counts = np.unique(names, return_counts=True)
    n_quadrats = len(unique_names)
    n_tiles = int(name_counts[0])
    if not np.all(name_counts == n_tiles):
        raise ValueError(
            f"per-image tile counts not uniform; got distribution "
            f"{np.unique(name_counts, return_counts=True)}"
        )
    if n_quadrats * n_tiles != n_rows:
        raise ValueError(
            f"reshape mismatch: {n_quadrats} * {n_tiles} != {n_rows}"
        )
    # Verify each image has tile_idx == [0, 1, ..., n_tiles-1] in order
    expected = np.tile(np.arange(n_tiles, dtype=tile_idxs.dtype), n_quadrats)
    if not np.array_equal(tile_idxs, expected):
        raise ValueError("tile_idx within each image is not [0..n_tiles-1]")

    test_cls = flat.reshape(n_quadrats, n_tiles, dim)
    image_names = unique_names  # already sorted
    return image_names, test_cls


def apply_threshold(
    top1_sim: np.ndarray, top1_class_idx: np.ndarray, threshold: float
) -> np.ndarray:
    """Return ``tile_to_class_idx`` (-1 below threshold)."""
    if top1_sim.shape != top1_class_idx.shape:
        raise ValueError(
            f"shape mismatch: sim {top1_sim.shape} vs class_idx {top1_class_idx.shape}"
        )
    return np.where(top1_sim >= threshold, top1_class_idx, -1).astype(np.int32)


def plot_histogram(top1_sim_flat: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(top1_sim_flat, bins=80, color="#3a7", edgecolor="white")
    for tau in (0.75, 0.80, 0.85, 0.90, 0.95):
        ax.axvline(tau, color="#888", lw=1, ls="--", alpha=0.6)
        cov = (top1_sim_flat >= tau).mean()
        ax.text(
            tau, ax.get_ylim()[1] * 0.95,
            f"tau={tau}\n{cov*100:.1f}%",
            ha="center", va="top", fontsize=8, color="#333",
        )
    ax.set_xlabel("top-1 cosine similarity (test tile -> nearest train CLS)")
    ax.set_ylabel("# tiles")
    ax.set_title(f"FAISS top-1 sim histogram (n={len(top1_sim_flat)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def summarize(top1_sim_flat: np.ndarray, threshold: float) -> str:
    pcts = [10, 25, 50, 75, 90, 95, 99]
    p = np.percentile(top1_sim_flat, pcts)
    lines = ["Top-1 cosine similarity distribution:"]
    lines.append(f"  n_tiles={len(top1_sim_flat)}")
    lines.append(f"  min={top1_sim_flat.min():.4f} max={top1_sim_flat.max():.4f} "
                 f"mean={top1_sim_flat.mean():.4f}")
    lines.append("  percentiles:")
    for pct, v in zip(pcts, p):
        lines.append(f"    p{pct:>2} = {v:.4f}")
    lines.append("")
    lines.append("Label coverage at candidate thresholds:")
    for tau in (0.75, 0.80, 0.85, 0.90, 0.95):
        cov = (top1_sim_flat >= tau).mean()
        lines.append(f"  tau={tau:.2f} -> {cov*100:5.1f}% of tiles labelled")
    lines.append("")
    cov_used = (top1_sim_flat >= threshold).mean()
    lines.append(f"Chosen threshold: {threshold:.4f} -> "
                 f"{cov_used*100:.1f}% coverage")
    return "\n".join(lines)


def main(
    train_lookup_path: Path,
    test_parquet_path: Path,
    output_dir: Path,
    threshold: float = 0.85,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load train artefacts ----
    print(f"[1/5] Loading train lookup: {train_lookup_path}")
    t0 = time.time()
    data = np.load(train_lookup_path)
    train_emb = data["embeddings"].astype(np.float32, copy=True)
    train_offsets = data["offsets"]
    n_train, dim = train_emb.shape
    print(f"      train_emb={train_emb.shape} {train_emb.dtype} ({time.time()-t0:.1f}s)")

    # ---- 2. L2-normalise + build IndexFlatIP ----
    print("[2/5] L2-normalising train embeddings (in place)...")
    t0 = time.time()
    l2_normalize_rows(train_emb)
    print(f"      done ({time.time()-t0:.1f}s)")

    print("      building faiss.IndexFlatIP...")
    t0 = time.time()
    index = faiss.IndexFlatIP(dim)
    index.add(train_emb)
    print(f"      index ntotal={index.ntotal} ({time.time()-t0:.1f}s)")

    # We don't need the raw train_emb after the index is built.
    del train_emb

    # ---- 3. Load test parquet ----
    print(f"[3/5] Loading test parquet: {test_parquet_path}")
    t0 = time.time()
    image_names, test_cls = decode_test_parquet(test_parquet_path)
    n_quadrats, n_tiles, _ = test_cls.shape
    print(
        f"      image_names={len(image_names)} test_cls={test_cls.shape} "
        f"({time.time()-t0:.1f}s)"
    )

    # ---- 4. L2-normalise test + FAISS query ----
    test_query = test_cls.reshape(n_quadrats * n_tiles, dim).copy()
    print(f"[4/5] L2-normalising test queries ({test_query.shape})...")
    l2_normalize_rows(test_query)

    print("      faiss top-1 search...")
    t0 = time.time()
    sims, train_rows = index.search(test_query, 1)
    print(f"      done ({time.time()-t0:.1f}s)")
    top1_sim_flat = sims[:, 0].astype(np.float32)
    top1_train_row = train_rows[:, 0]
    top1_class_idx_flat = train_row_to_class_idx(top1_train_row, train_offsets)

    # Reshape to per-quadrat
    top1_sim = top1_sim_flat.reshape(n_quadrats, n_tiles)
    top1_class_idx = top1_class_idx_flat.reshape(n_quadrats, n_tiles)

    # ---- 5. Threshold + persist ----
    tile_to_class_idx = apply_threshold(top1_sim, top1_class_idx, threshold)

    print()
    print("[5/5] Distribution + threshold summary:")
    summary = summarize(top1_sim_flat, threshold)
    print(summary)

    hist_path = output_dir / "tile_top1_sim_histogram.png"
    plot_histogram(top1_sim_flat, hist_path)
    print(f"\n      wrote histogram: {hist_path}")

    threshold_txt = output_dir / "faiss_threshold.txt"
    threshold_txt.write_text(summary + "\n")
    print(f"      wrote threshold summary: {threshold_txt}")

    out_path = output_dir / "sampler_b_lookup.npz"
    np.savez(
        out_path,
        image_names=image_names,
        test_cls=test_cls,
        top1_sim=top1_sim,
        top1_class_idx=top1_class_idx,
        tile_to_class_idx=tile_to_class_idx,
        threshold=np.float32(threshold),
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"      wrote {out_path} ({size_mb:.1f} MB)")
    print("Done.")


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-lookup",
        type=Path,
        default=Path(__file__).parent / "artifacts/train_cls_by_class_idx.npz",
    )
    p.add_argument(
        "--test-parquet",
        type=Path,
        default=Path(
            "/storage/project/ps-clef2026_plantclef-0/shared/inference/"
            "dinov2-vit-base/cls_grid_4x4.parquet"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Cosine-sim cutoff for per-tile species labelling (spec default 0.85).",
    )
    args = p.parse_args()
    main(
        train_lookup_path=args.train_lookup,
        test_parquet_path=args.test_parquet,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    cli()
