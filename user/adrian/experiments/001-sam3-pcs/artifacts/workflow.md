# SAM3 PCS — End-to-End Workflow

## Overview

This experiment runs Meta's SAM3 model over the entire PlantCLEF 2026 training set
(~1.4M images, 7,806 species) to produce per-image segmentation masks. Two text prompts
were run: `"plant"` and `"flower"`. Results are stored as RLE-encoded masks in
zstd-compressed Parquet files — one file per SLURM task — to stay within scratch inode
limits and enable fast columnar access via `polars`.

**Status**: Both prompt runs complete as of 2026-03-11.

---

## Machines Involved

| Role | Host | Purpose |
|------|------|---------|
| **Data host** | `acheung46@login-phoenix.pace.gatech.edu` | Stores the raw PlantCLEF images |
| **Compute node** | PACE Phoenix H100 node (allocated by SLURM) | Runs SAM3 inference |
| **Login node** | `login-phoenix.pace.gatech.edu` | Job submission only |

---

## Directory Structure

### Data host (remote)
```
~/ps-clef2026_plantclef-0/train/PlantCLEF2024/train/
├── 1355868/          ← species ID (taxon ID)
│   ├── 0070793945bc6db2c597387006c5425751204baa.jpg
│   └── ...           ← ~167 images avg (up to ~800 for large species)
├── 1355869/
└── ...               ← 7,806 species total, ~1.4M images total
```

### Compute node (local, ephemeral)
```
$TMPDIR/                          ← fast local NVMe, exists only during the job
├── species_task_{task_id}.txt    ← this task's slice of species IDs (~390 species)
└── sam3_staging/
    └── {species_id}/             ← rsynced images, ONE species at a time
        └── ...                   ← deleted immediately after inference
```

### Scratch (persistent output)
```
~/scratch/plantclef-2026/
├── .venv/                          ← Python 3.12 venv with all dependencies
├── .cache/huggingface/hub/         ← SAM3 model weights cache (~3GB)
└── sam3_outputs/
    ├── plant/                      ← prompt="plant" run
    │   ├── task_0000/
    │   │   └── results.parquet     ← masks for species 1–390  (~600 MB)
    │   ├── task_0001/
    │   │   └── results.parquet
    │   └── ...                     ← 20 files, ~6.7 GB total
    └── flower/                     ← prompt="flower" run
        ├── task_0000/
        │   └── results.parquet     ← masks for species 1–390  (~107 MB)
        ├── task_0001/
        │   └── results.parquet
        └── ...                     ← 20 files, ~1.9 GB total
```

### Repo
```
~/arc/plantclef-2026/user/adrian/experiments/001-sam3-pcs/
├── species_list.txt              ← all 7,806 species IDs, one per line
├── species_test.txt              ← 5-species subset for validation runs
├── src/run_sam3.py               ← main inference script
├── submit.sbatch                 ← production SLURM array job
├── validate.sbatch               ← single-task test job
├── analysis/                     ← post-hoc analysis scripts
│   ├── summary_stats.py
│   ├── threshold_sensitivity.py
│   ├── species_detection_rate.py
│   └── prompt_agreement.py
├── artifacts/
│   ├── workflow.md               ← this file
│   └── sam3-analysis-results.md ← full analysis outputs & insights
└── logs/
    ├── sam3-{arrayid}_{taskid}.out
    └── validate-{jobid}.out
```

---

## Parquet Schema

One row per detected instance. Images with no detections get exactly one row with
`mask_idx = -1` and `NaN` for box/score fields.

| Column | Type | Description |
|--------|------|-------------|
| `species_id` | string | Taxon ID (directory name on data host) |
| `filename` | string | Image filename (e.g. `abc123.jpg`) |
| `prompt` | string | Text prompt used (`"plant"` or `"flower"`) |
| `shape_h` | int32 | Original image height in pixels |
| `shape_w` | int32 | Original image width in pixels |
| `mask_idx` | int16 | Instance index; `-1` = no detection |
| `rle` | binary | COCO column-major RLE as raw int32 bytes; `b""` if no detection |
| `box_x1/y1/x2/y2` | float32 | Bounding box in absolute pixel coords (xyxy) |
| `score` | float32 | Detection confidence score (≥ 0.5 at inference time) |

---

## Step-by-Step Workflow

### Step 1 — Job submission (login node)

```bash
cd ~/arc/plantclef-2026/user/adrian/experiments/001-sam3-pcs

# plant prompt
sbatch --export=ALL,PROMPT=plant submit.sbatch

# flower prompt (separate job)
sbatch --export=ALL,PROMPT=flower submit.sbatch
```

SLURM creates **20 array tasks** (IDs 0–19), each requesting 1 H100, 8 CPUs, 64GB RAM.
All 20 start simultaneously when GPUs are available.

---

### Step 2 — Species slice (per task, in `submit.sbatch`)

Each task computes its own contiguous slice of `species_list.txt`:

```
task 0  → lines 1–390    → species 1355868 … 1357xxx
task 1  → lines 391–780  → species 1357xxx … 1359xxx
...
task 19 → lines 7417–7806
```

The slice is written to `$TMPDIR/species_task_{task_id}.txt` and passed to `run_sam3.py`.

---

### Step 3 — Model load (once per task)

`run_sam3.py` loads SAM3 from the HuggingFace cache on scratch:

```
~/scratch/plantclef-2026/.cache/huggingface/hub/models--facebook--sam3/
```

Model is loaded in **bfloat16** and moved to the H100. This takes ~30–60s and happens
once per task, not once per species.

---

### Step 4 — Per-species loop (`run_sam3.py`, repeated ~390 times per task)

For each species ID in the task's slice:

#### 4a. Rsync (~5–20s)

```
rsync -aq acheung46@login-phoenix.pace.gatech.edu:
    ~/ps-clef2026_plantclef-0/train/PlantCLEF2024/train/{species_id}/
→  $TMPDIR/sam3_staging/{species_id}/
```

Images land on **local NVMe** — fast random access for DataLoader workers.
Only one species directory exists on disk at a time (~54MB typical).

#### 4b. Inference

Images in `$TMPDIR/sam3_staging/{species_id}/` are loaded by a PyTorch `DataLoader`
(8 worker threads, batch size 8) and fed through SAM3:

```
PIL images (batch of 8)
    → Sam3Processor                         # resize to 1008×1008, tokenize prompt
    → Sam3Model.forward()                   # ViT encoder → DETR → mask decoder
    → processor.post_process_instance_segmentation(threshold=0.5)
    → per-image: masks (bool H×W), boxes (N×4 float32 xyxy px), scores (N float32)
```

Measured throughput: **9.27 img/s** on H100 80GB.

#### 4c. RLE encoding

Each boolean mask `(H, W)` is compressed into a COCO-style column-major RLE array
(int32 run-length counts starting from the zero-pixel run). This is ~100–1000× smaller
than storing the raw mask. The RLE bytes are stored as a `binary` column in Parquet.

#### 4d. Cleanup

```bash
rm -rf $TMPDIR/sam3_staging/{species_id}/
```

Disk is freed before the next species is rsynced.

---

### Step 5 — Save output

After all ~390 species are processed, the task writes a single compressed Parquet file:

```
~/scratch/plantclef-2026/sam3_outputs/{prompt}/task_{NNNN}/results.parquet
```

Compressed with zstd. File sizes vary by prompt: ~600 MB/task for `plant`, ~100 MB/task
for `flower` (flower detects fewer images but more masks per image).

---

## Loading Results

### With polars (recommended)

```python
import polars as pl
import numpy as np

def rle_decode(rle_bytes: bytes, h: int, w: int) -> np.ndarray:
    rle = np.frombuffer(rle_bytes, dtype=np.int32)
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    for i, length in enumerate(rle):
        if i % 2 == 1:
            flat[pos:pos + length] = True
        pos += length
    return flat.reshape((h, w), order="F")

# Load one task
df = pl.read_parquet(
    "~/scratch/plantclef-2026/sam3_outputs/plant/task_0000/results.parquet"
)

# All detected instances for one image
row = df.filter(
    (pl.col("species_id") == "1355868") &
    (pl.col("filename") == "0070793945bc6db2c597387006c5425751204baa.jpg") &
    (pl.col("mask_idx") >= 0)
)

# Decode first mask
r = row.row(0, named=True)
mask = rle_decode(r["rle"], r["shape_h"], r["shape_w"])  # bool (H, W)
box  = (r["box_x1"], r["box_y1"], r["box_x2"], r["box_y2"])
score = r["score"]
```

### Load all tasks for a prompt

```python
import polars as pl
from pathlib import Path

data_dir = Path("~/scratch/plantclef-2026/sam3_outputs").expanduser()
df = pl.concat([
    pl.read_parquet(p)
    for p in sorted((data_dir / "plant").glob("task_*/results.parquet"))
])
```

---

## Scale & Timing

| Metric | plant | flower |
|--------|-------|--------|
| Total species | 7,806 | 7,806 |
| Total images | 1,308,899 | 1,308,632 |
| Detection rate (threshold=0.5) | 73.9% | 58.0% |
| Avg masks / detected image | 2.86 | 12.36 |
| Measured throughput | ~9.27 img/s | ~9.27 img/s |
| SLURM tasks | 20 | 20 |
| Species per task | ~390 | ~390 |
| Requested `--time` | 6:00:00 | 6:00:00 |
| Total output size | ~6.7 GB | ~1.9 GB |

---

## Analysis Scripts

All scripts in `analysis/` are runnable via:
```bash
cd user/adrian/experiments/001-sam3-pcs
UV_PROJECT_ENVIRONMENT=~/scratch/plantclef-2026/.venv uv run --package 001-sam3-pcs \
    python analysis/<script>.py [args]
```

| Script | Purpose |
|--------|---------|
| `summary_stats.py` | Per-task table + totals: images, detection rate, masks/img, size |
| `threshold_sensitivity.py` | How detection rate changes at score thresholds 0.5–0.9 |
| `species_detection_rate.py` | Top/bottom N species by detection rate; count of 0%-species |
| `prompt_agreement.py` | plant vs flower 2×2 contingency + per-species disagreement |

See `artifacts/sam3-analysis-results.md` for full outputs and insights.

---

## Running

### Validate first (5 species, single GPU)
```bash
sbatch validate.sbatch
tail -f logs/validate-{jobid}.out
```

### Full production run
```bash
sbatch submit.sbatch                    # runs with default PROMPT=plant
sbatch --export=ALL,PROMPT=flower submit.sbatch
squeue -u $USER
```
