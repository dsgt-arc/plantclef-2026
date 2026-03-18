#!/usr/bin/env python3
"""Download PlantCLEF 2026 competition data to shared project storage.

Downloads training data, test data, and pretrained model, then extracts
tar archives. Uses aria2c for fast, resumable, multi-threaded downloads.

Usage:
    python download_data.py [--dry-run] [--skip-extract]

Destinations:
    Training/test data -> /storage/project/ps-clef2026_plantclef-0/shared/data/raw/
    Pretrained model   -> /storage/project/ps-clef2026_plantclef-0/shared/pretrained_models/
"""

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path

# Shared project storage paths
SHARED_BASE = Path("/storage/project/ps-clef2026_plantclef-0/shared")
DATA_RAW_DIR = SHARED_BASE / "data" / "raw"
MODEL_DIR = SHARED_BASE / "pretrained_models"

DOWNLOADS = [
    {
        "name": "PlantCLEF 2024 Training Data",
        "url": "https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata.tar",
        "dest_dir": DATA_RAW_DIR,
        "filename": "PlantCLEF2024singleplanttrainingdata.tar",
        "expected_size_gb": 280.0,
        "extract": True,
    },
    {
        "name": "PlantCLEF 2025 Test Data",
        "url": "https://lab.plantnet.org/LifeCLEF/PlantCLEF2025/vegetation_plot_test_data/PlantCLEF2025test.tar",
        "dest_dir": DATA_RAW_DIR,
        "filename": "PlantCLEF2025test.tar",
        "expected_size_gb": 8.0,
        "extract": True,
    },
    {
        "name": "PlantNet Pretrained Models (SW Europe flora)",
        "url": "https://zenodo.org/records/10848263/files/PlantNet_PlantCLEF2024_pretrained_models_on_the_flora_of_south-western_europe.tar?download=1",
        "dest_dir": MODEL_DIR,
        "filename": "PlantNet_PlantCLEF2024_pretrained_models_on_the_flora_of_south-western_europe.tar",
        "expected_size_gb": 2.0,
        "extract": True,
    },
]


def fmt_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def verify_file_size(filepath: Path, expected_gb: float) -> bool:
    """Check that file size is within 10% of expected size."""
    if not filepath.exists():
        return False
    actual = filepath.stat().st_size
    expected = int(expected_gb * 1024**3)
    tolerance = 0.10
    lower = expected * (1 - tolerance)
    upper = expected * (1 + tolerance)
    if lower <= actual <= upper:
        print(f"  Size OK: {fmt_size(actual)} (expected ~{expected_gb} GB)")
        return True
    else:
        print(f"  WARNING: Size mismatch! Got {fmt_size(actual)}, expected ~{expected_gb} GB")
        return False


def download_file(url: str, dest_dir: Path, filename: str, dry_run: bool = False) -> Path:
    """Download a file using aria2c with resume and multi-threaded support."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    filepath = dest_dir / filename

    if filepath.exists():
        print(f"  File already exists: {filepath}")
        return filepath

    if dry_run:
        print(f"  [DRY RUN] Would download to: {filepath}")
        return filepath

    print(f"  Downloading to: {filepath}")
    cmd = [
        "aria2c",
        "--continue=true",          # Resume partial downloads
        "--check-certificate=false",
        "--split=4",                # Split into 4 connections
        "--max-connection-per-server=4",
        "--dir", str(dest_dir),
        "--out", filename,
        url,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR: aria2c exited with code {result.returncode}")
        sys.exit(1)

    return filepath


def extract_tar(filepath: Path, dry_run: bool = False) -> None:
    """Extract a tar archive into the same directory."""
    dest_dir = filepath.parent
    print(f"  Extracting to: {dest_dir}")

    if dry_run:
        print(f"  [DRY RUN] Would extract: {filepath}")
        return

    with tarfile.open(filepath) as tar:
        # Security: check for path traversal
        for member in tar.getmembers():
            member_path = os.path.join(dest_dir, member.name)
            if not os.path.realpath(member_path).startswith(os.path.realpath(str(dest_dir))):
                print(f"  ERROR: Tar member {member.name} escapes destination directory!")
                sys.exit(1)
        tar.extractall(path=dest_dir)

    print(f"  Extraction complete")


def main():
    parser = argparse.ArgumentParser(description="Download PlantCLEF 2026 competition data")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without downloading")
    parser.add_argument("--skip-extract", action="store_true", help="Download only, skip tar extraction")
    args = parser.parse_args()

    print("=" * 60)
    print("PlantCLEF 2026 Data Download")
    print("=" * 60)
    print()

    for i, dl in enumerate(DOWNLOADS, 1):
        print(f"[{i}/{len(DOWNLOADS)}] {dl['name']}")
        print(f"  URL: {dl['url']}")

        filepath = download_file(dl["url"], dl["dest_dir"], dl["filename"], dry_run=args.dry_run)

        if not args.dry_run and filepath.exists():
            verify_file_size(filepath, dl["expected_size_gb"])

        if dl["extract"] and not args.skip_extract:
            extract_tar(filepath, dry_run=args.dry_run)

        print()

    print("=" * 60)
    print("Done!")
    print("=" * 60)
    print()
    print("Data locations:")
    print(f"  Training data: {DATA_RAW_DIR}")
    print(f"  Test data:     {DATA_RAW_DIR}")
    print(f"  Models:        {MODEL_DIR}")


if __name__ == "__main__":
    main()
