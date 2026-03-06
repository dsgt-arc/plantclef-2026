#!/bin/bash
#SBATCH --job-name=delete-data
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#SBATCH --output=/storage/project/ps-clef2026_plantclef-0/mgustineli3/plantclef-2026/user/murilo/sbatch/logs/slurm_delete_%j.log
#
# Delete PlantCLEF2024 dataset (~280GB)
#
# Usage (run from anywhere):
#   sbatch /path/to/delete_data.sh
#

set -eu

TARGET_DIR="/storage/project/ps-clef2026_plantclef-0/shared/data/raw/PlantCLEF2024"
TARGET_TAR="/storage/project/ps-clef2026_plantclef-0/shared/data/raw/PlantCLEF2024singleplanttrainingdata.tar"

echo "=============================================="
echo "Job: Delete PlantCLEF2024 Dataset"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Target dir: ${TARGET_DIR}"
echo "Target tar: ${TARGET_TAR}"
echo ""

# --- Delete directory ---
if [ -d "$TARGET_DIR" ]; then
    echo "Directory size before deletion:"
    du -sh "$TARGET_DIR"
    echo ""

    echo "Step 1: Deleting files (8 parallel workers)..."
    find "$TARGET_DIR" -type f -print0 | xargs -0 -P 8 rm -f
    echo "Files deleted."
    echo ""

    echo "Step 2: Removing empty directories..."
    find "$TARGET_DIR" -type d -empty -delete
    echo "Empty directories removed."
    echo ""

    echo "Step 3: Removing top-level directory..."
    rmdir "$TARGET_DIR"
    echo "Top-level directory removed."
else
    echo "Directory does not exist: ${TARGET_DIR} (skipping)"
fi
echo ""

# --- Delete tar file ---
if [ -f "$TARGET_TAR" ]; then
    echo "Deleting tar file..."
    ls -lh "$TARGET_TAR"
    rm -f "$TARGET_TAR"
    echo "Tar file deleted."
else
    echo "Tar file does not exist: ${TARGET_TAR} (skipping)"
fi
echo ""

# --- Verify ---
echo "Verifying deletion..."
if [ ! -d "$TARGET_DIR" ] && [ ! -f "$TARGET_TAR" ]; then
    echo "SUCCESS: Both directory and tar file have been deleted."
else
    [ -d "$TARGET_DIR" ] && echo "WARNING: ${TARGET_DIR} still exists."
    [ -f "$TARGET_TAR" ] && echo "WARNING: ${TARGET_TAR} still exists."
fi

echo ""
echo "=============================================="
echo "End Time: $(date)"
echo "Job completed!"
echo "=============================================="
