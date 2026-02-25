# Proposal: SAM3 Promptable Concept Segmentation for PlantCLEF 2026

**Status**: in-progress
**Created**: 2026-02-18
**Author**: Adrian Cheung

## Hypothesis

Running Meta's SAM3 over the 1.4M PlantCLEF training images in a massively parallel SLURM
job array will produce high-quality per-image segmentation masks that can serve as spatial
priors for downstream plant species classification, improving mAP compared to a baseline
that uses the full uncropped image.

## Background

PlantCLEF 2026 training images contain plants photographed in diverse, cluttered
backgrounds. Instance segmentation isolates relevant plant structure, reducing background
noise for classifiers. SAM3 offers built-in open-vocabulary text prompting (no external
grounding model required), making it well-suited for zero-shot plant concept segmentation
at scale. See vendor doc: [`docs/vendor/sam3.md`](../../../../docs/vendor/sam3.md).

## Method

### Approach

1. Chunk the full image dataset into N equal-sized subsets (one per SLURM array job).
2. Each job: copy its chunk to local NVMe `$TMPDIR`, run `run_sam3.py` in bfloat16 on
   a single H100 GPU, write one `results.npz` file per chunk to scratch storage.
3. Downstream experiments load masks from the `.npz` files as cheap spatial priors.

### Setup

| Component | Details |
|-----------|---------|
| Data | PlantCLEF 2026 training set (~1.4M JPEGs on scratch) |
| Compute | PACE Phoenix, 1× H100 80GB per array job, `#SBATCH --gres=gpu:H100:1` |
| Model | `facebook/sam3` (gated; 848M params, bfloat16) via `sam3` pip package |
| Dependencies | `sam3` (from GitHub), `torch>=2.7`, `Pillow`, `tqdm`, `numpy` |
| venv | `~/scratch/plantclef-2026/.venv` (Python 3.12) |
| Code | `src/run_sam3.py` — main inference script |

### Procedure

1. Install `sam3` package into `~/scratch/plantclef-2026/.venv`.
2. Validate `run_sam3.py` on a 100-image test chunk on a compute node.
3. Confirm `.npz` output structure and RLE round-trip correctness.
4. Write and submit SLURM array job script over all chunks.
5. Spot-check decoded mask quality on 10 random images.
6. Feed masks downstream as ROI crops or attention priors.

### Variables

- **Independent**: Text prompt string (e.g. `"plant"`, `"leaf and flower"`)
- **Dependent**: Mask coverage of plant pixels; throughput (images/sec)
- **Controlled**: bfloat16 precision, H100 GPU, processor default image resolution

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Throughput | Images/second on H100 (from SLURM stdout) |
| Mask recall | Fraction of plant pixels covered vs. manual spot check |
| Downstream mAP | Classification mAP with vs. without mask-cropped inputs |

### Baseline

No segmentation — classification trained on full uncropped images.

### Success Criteria

- **Confirm if**: Throughput ≥ 5 images/sec AND mask recall ≥ 80% on spot check
- **Reject if**: Throughput < 1 images/sec OR systematic mask misses on plant objects

## Limitations

- `Sam3Processor` API is per-image; batching is implemented at the DataLoader level
  (parallelised workers feed the GPU sequentially).
- Masks may be over-segmented for cluttered multi-plant scenes; downstream classifier
  must handle multiple mask candidates per image.
- Inode limits on scratch require aggregating all per-image results into a single
  `.npz` per chunk (no per-image files).
