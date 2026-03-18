"""
SAM3 mask visualizer for PlantCLEF 2026.

Rsyncs images from the remote host, draws semi-transparent mask overlays +
bounding boxes, and saves PNG files per species/prompt.

Usage (local test):
    UV_PROJECT_ENVIRONMENT=~/scratch/plantclef-2026/.venv \\
      uv run --package 001-sam3-pcs python analysis/visualize_masks.py \\
        --n_species 5 --max_images 2 --prompt plant

Usage (specific species):
    UV_PROJECT_ENVIRONMENT=~/scratch/plantclef-2026/.venv \\
      uv run --package 001-sam3-pcs python analysis/visualize_masks.py \\
        --species 1356390 1743350 --max_images 5
"""

import argparse
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image, ImageDraw

# Resolve src/ relative to this file so rle_decode can be imported
_SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))
from run_sam3 import rle_decode  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLOURS = [
    (255,  80,  80, 120),
    ( 80, 200,  80, 120),
    ( 80, 120, 255, 120),
    (255, 200,  50, 120),
    (200,  80, 255, 120),
    ( 50, 220, 220, 120),
    (255, 140,   0, 120),
    (  0, 200, 150, 120),
    (220,  50,  50, 120),
    (100, 100, 255, 120),
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize SAM3 masks for PlantCLEF 2026")
    p.add_argument("--data_dir", type=Path,
                   default=Path("~/scratch/plantclef-2026/sam3_outputs").expanduser(),
                   help="Parquet root (default: ~/scratch/plantclef-2026/sam3_outputs)")
    p.add_argument("--remote", default="acheung46@login-phoenix.pace.gatech.edu",
                   help="SSH host for rsyncing images")
    p.add_argument("--remote_dir",
                   default="~/ps-clef2026_plantclef-0/train/PlantCLEF2024/train",
                   help="Base image directory on remote host")
    p.add_argument("--out_dir", type=Path,
                   default=Path("~/scratch/plantclef-2026/sam3_vis").expanduser(),
                   help="Output root (default: ~/scratch/plantclef-2026/sam3_vis)")
    p.add_argument("--stage_dir", type=Path,
                   default=Path(os.environ.get("TMPDIR", "/tmp")) / "vis_staging",
                   help="Temp dir for rsynced images (default: $TMPDIR/vis_staging)")

    species_group = p.add_mutually_exclusive_group(required=True)
    species_group.add_argument("--species", nargs="+", metavar="SPECIES_ID",
                               help="One or more species IDs to visualize")
    species_group.add_argument("--n_species", type=int, metavar="N",
                               help="Randomly sample this many species")

    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for species sampling (omit for fresh draw)")
    p.add_argument("--max_images", type=int, default=5,
                   help="Max images to visualize per species per prompt (default: 5)")
    p.add_argument("--prompt", default=None,
                   help="Restrict to a single prompt (default: all discovered prompts)")
    p.add_argument("--score_threshold", type=float, default=0.5,
                   help="Only draw masks with score >= this (default: 0.5)")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel species workers (default: 8)")

    args = p.parse_args()
    args.data_dir = args.data_dir.expanduser()
    args.out_dir  = args.out_dir.expanduser()
    args.stage_dir = args.stage_dir.expanduser()
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_prompts(data_dir: Path, prompt_filter: str | None) -> list[str]:
    prompts = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    if prompt_filter is not None:
        if prompt_filter not in prompts:
            raise ValueError(f"Prompt '{prompt_filter}' not found in {data_dir}. Available: {prompts}")
        prompts = [prompt_filter]
    return prompts


def parquet_paths(data_dir: Path, prompt: str) -> list[Path]:
    return sorted((data_dir / prompt).glob("task_*/results.parquet"))


def collect_all_species(data_dir: Path, prompts: list[str]) -> list[str]:
    """Collect all unique species IDs across all prompt parquets (lazy, no full load)."""
    all_ids: set[str] = set()
    for prompt in prompts:
        paths = parquet_paths(data_dir, prompt)
        if not paths:
            continue
        lf = pl.scan_parquet([str(p) for p in paths])
        ids = lf.select("species_id").unique().collect()["species_id"].to_list()
        all_ids.update(ids)
    return sorted(all_ids)


def load_species_data(data_dir: Path, prompt: str, species_set: set[str]) -> pl.DataFrame:
    """Lazy-load only rows for target species from all task parquets."""
    paths = parquet_paths(data_dir, prompt)
    if not paths:
        return pl.DataFrame()
    lf = pl.scan_parquet([str(p) for p in paths])
    return lf.filter(pl.col("species_id").is_in(species_set)).collect()


def select_images(df: pl.DataFrame, species_id: str, max_images: int) -> list[str]:
    """
    Pick up to max_images filenames for a species.
    Prefer images with detections; fall back to no-detection images if needed.
    Sorted by filename for determinism.
    """
    sp_df = df.filter(pl.col("species_id") == species_id)

    # Images with at least one detection
    det_files = (
        sp_df.filter(pl.col("mask_idx") >= 0)
        .select("filename").unique()
        .sort("filename")["filename"].to_list()
    )
    selected = det_files[:max_images]

    if len(selected) < max_images:
        # Fill remainder with no-detection images
        nodet_files = (
            sp_df.filter(pl.col("mask_idx") < 0)
            .select("filename").unique()
            .sort("filename")["filename"].to_list()
        )
        needed = max_images - len(selected)
        selected += nodet_files[:needed]

    return selected


def rsync_images(remote: str, remote_dir: str, species_id: str,
                 filenames: list[str], stage_dir: Path) -> set[str]:
    """Rsync all filenames for a species in one call. Returns set of successfully fetched filenames."""
    import tempfile
    species_stage = stage_dir / species_id
    species_stage.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        files_from = Path(f.name)
        f.write("\n".join(filenames))

    try:
        # Strip leading "~/" — rsync doesn't shell-expand ~ when using --files-from,
        # producing /home/user/~/path. A bare relative path is implicitly relative
        # to the remote home dir, which is equivalent and works correctly.
        remote_base = remote_dir[2:] if remote_dir.startswith("~/") else remote_dir
        src = f"{remote}:{remote_base}/{species_id}/"
        result = subprocess.run(
            ["rsync", "-aq", "--no-perms", f"--files-from={files_from}", src, str(species_stage) + "/"],
            capture_output=True, text=True,
        )
    finally:
        files_from.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  rsync failed for species {species_id}: {result.stderr.strip()}")
        return set()

    return {f for f in filenames if (species_stage / f).exists()}


def draw_masks(img: Image.Image, mask_rows: pl.DataFrame, score_threshold: float) -> Image.Image:
    """Alpha-composite mask overlays and bounding boxes onto img (RGBA)."""
    for i, row in enumerate(mask_rows.sort("mask_idx").iter_rows(named=True)):
        if row["score"] < score_threshold:
            continue
        rle   = np.frombuffer(row["rle"], dtype=np.int32)
        shape = (row["shape_h"], row["shape_w"])
        mask  = rle_decode(rle, shape)

        colour = COLOURS[i % len(COLOURS)]
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_arr = np.array(overlay)
        overlay_arr[mask] = colour
        img = Image.alpha_composite(img, Image.fromarray(overlay_arr, "RGBA"))

        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = row["box_x1"], row["box_y1"], row["box_x2"], row["box_y2"]
        box_colour = colour[:3] + (220,)
        draw.rectangle([x1, y1, x2, y2], outline=box_colour, width=2)
        draw.text((x1 + 3, y1 + 3), f"{row['score']:.2f}", fill=(255, 255, 255, 230))

    return img


def add_no_detection_watermark(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "NO DETECTION", fill=(255, 0, 0, 230))
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    prompts = discover_prompts(args.data_dir, args.prompt)
    print(f"Prompts: {prompts}")

    # Determine species set
    if args.species is not None:
        species_list = args.species
    else:
        print(f"Collecting all species IDs from {len(prompts)} prompt(s)...")
        all_species = collect_all_species(args.data_dir, prompts)
        if args.n_species > len(all_species):
            print(f"  WARNING: requested {args.n_species} but only {len(all_species)} available")
            args.n_species = len(all_species)
        rng = random.Random(args.seed)
        species_list = sorted(rng.sample(all_species, args.n_species))
        print(f"Sampled {len(species_list)} species (seed={args.seed})")

    species_set = set(species_list)

    args.stage_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Summary tracking: prompt -> species -> (saved, with_det)
    summary: dict[str, dict[str, list[int]]] = {}

    def process_species(species_id: str, df: pl.DataFrame, prompt: str) -> tuple[str, int, int]:
        """Worker: rsync + draw + save all images for one species/prompt. Returns (species_id, n_saved, n_with_det)."""
        filenames = select_images(df, species_id, args.max_images)
        if not filenames:
            return species_id, 0, 0

        out_species_dir = args.out_dir / species_id
        out_species_dir.mkdir(parents=True, exist_ok=True)

        fetched = rsync_images(args.remote, args.remote_dir, species_id, filenames, args.stage_dir)

        n_saved = n_with_det = 0
        for filename in filenames:
            if filename not in fetched:
                continue

            local_img = args.stage_dir / species_id / filename
            img = Image.open(local_img).convert("RGBA")

            stem = Path(filename).stem
            img.convert("RGB").save(out_species_dir / f"{stem}.png")

            mask_rows = df.filter(
                (pl.col("species_id") == species_id) &
                (pl.col("filename")   == filename)   &
                (pl.col("mask_idx")   >= 0)          &
                (pl.col("score")      >= args.score_threshold)
            ).sort("mask_idx")

            has_detection = len(mask_rows) > 0
            if has_detection:
                img = draw_masks(img, mask_rows, args.score_threshold)
                n_with_det += 1
            else:
                img = add_no_detection_watermark(img)

            img.convert("RGB").save(out_species_dir / f"{stem}_{prompt}.png")
            print(f"  {species_id}/{stem}_{prompt}.png  ({len(mask_rows)} mask(s))")
            n_saved += 1
            local_img.unlink(missing_ok=True)

        return species_id, n_saved, n_with_det

    for prompt in prompts:
        print(f"\n=== Prompt: {prompt} ===")
        df = load_species_data(args.data_dir, prompt, species_set)
        if len(df) == 0:
            print(f"  No data found for prompt '{prompt}'")
            continue

        found_species = set(df["species_id"].unique().to_list())
        missing = species_set - found_species
        if missing:
            print(f"  WARNING: {len(missing)} species not found in '{prompt}': {sorted(missing)[:5]}...")

        active_species = [s for s in sorted(species_list) if s in found_species]
        prompt_summary: dict[str, list[int]] = {}

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_species, sid, df, prompt): sid
                for sid in active_species
            }
            for future in as_completed(futures):
                sid, n_saved, n_with_det = future.result()
                prompt_summary[sid] = [n_saved, n_with_det]

        summary[prompt] = prompt_summary

    # Summary table
    print("\n=== Summary ===")
    header = f"{'prompt':<12} {'species':<12} {'saved':>8} {'w/det':>8}"
    print(header)
    print("-" * len(header))
    for prompt, sp_dict in summary.items():
        for species_id, (n_saved, n_with_det) in sp_dict.items():
            print(f"{prompt:<12} {species_id:<12} {n_saved:>8} {n_with_det:>8}")

    total_saved   = sum(v[0] for sp in summary.values() for v in sp.values())
    total_with_det = sum(v[1] for sp in summary.values() for v in sp.values())
    print("-" * len(header))
    print(f"{'TOTAL':<12} {'':<12} {total_saved:>8} {total_with_det:>8}")
    print(f"\nOutput: {args.out_dir}")


if __name__ == "__main__":
    main()
