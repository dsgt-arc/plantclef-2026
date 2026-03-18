# Proposal: SAM3 Promptable Concept Segmentation for PlantCLEF 2026

**Status**: complete
**Created**: 2026-02-18
**Completed**: 2026-03-11
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
| Model | `facebook/sam3` (gated; 848M params, bfloat16) via 🤗 `transformers` |
| Dependencies | `transformers>=4.40`, `torch>=2.7`, `Pillow`, `tqdm`, `pyarrow`, `polars` |
| venv | `~/scratch/plantclef-2026/.venv` (Python 3.12) |
| Code | `src/run_sam3.py` — main inference script |
| Output format | Parquet (zstd-compressed), one `results.parquet` per task |
| Prompts run | `"plant"` and `"flower"` (separate array jobs, 20 tasks each) |

### Procedure

1. Install dependencies into `~/scratch/plantclef-2026/.venv` (`uv sync`).
2. Validate `run_sam3.py` on a 5-species test subset (`species_test.txt`) on a compute node.
3. Confirm Parquet output schema and RLE round-trip correctness.
4. Submit production SLURM array job (20 tasks) for `"plant"` prompt.
5. Submit second array job for `"flower"` prompt.
6. Run analysis scripts in `analysis/` to characterize output quality.
7. Feed masks downstream as ROI crops or attention priors.

### Variables

- **Independent**: Text prompt string — `"plant"` and `"flower"` both completed
- **Dependent**: Mask coverage of plant pixels; throughput (images/sec)
- **Controlled**: bfloat16 precision, H100 GPU, processor default image resolution (1008px), score threshold 0.5

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

### Results Summary (completed 2026-03-11)

| Metric | plant | flower |
|---|---|---|
| Total images | 1,308,899 | 1,308,632 |
| Detection rate (threshold=0.5) | 73.9% | 58.0% |
| Masks / detected image | 2.86 | 12.36 |
| Disk size | 6.7 GB | 1.9 GB |
| Throughput (H100 bfloat16) | ~9.27 img/s | ~9.27 img/s |

- Throughput criterion met (9.27 img/s >> 5 img/s). ✓
- 106 species with 0% detection under `plant` prompt — warrants downstream handling.
- Prompts are highly complementary: union covers ~93.2% of images.
- See `artifacts/sam3-analysis-results.md` for full analysis.

## Limitations

- `Sam3Processor` API is per-image; batching is implemented at the DataLoader level
  (parallelised workers feed the GPU sequentially).
- Masks may be over-segmented for cluttered multi-plant scenes; downstream classifier
  must handle multiple mask candidates per image.
- Inode limits on scratch require aggregating all per-image results into a single
  `.npz` per chunk (no per-image files).
