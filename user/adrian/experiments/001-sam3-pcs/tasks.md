# Tasks: 001-sam3-pcs

## 1. Setup
- [x] 1.1 Create venv at `~/scratch/plantclef-2026/.venv` (Python 3.12)
- [x] 1.2 Add `vendor/sam3` git submodule
- [x] 1.3 Document SAM3 transformers API in `docs/vendor/sam3.md`
- [x] 1.4 Install dependencies into venv (`uv sync`)
- [x] 1.5 Verify HuggingFace gated model access for `facebook/sam3`
- [x] 1.6 Prepare a 100-image test chunk for validation (`species_test.txt`, 5 species)

## 2. Implementation
- [x] 2.1 Write `src/run_sam3.py` — batched inference script (Parquet output, remote streaming mode)
- [x] 2.2 Write `submit.sbatch` — SLURM array job script (20 tasks, H100, 6h wall time)
- [x] 2.3 Validate on test chunk on a compute node (no crash, output produced)
- [x] 2.4 Verify Parquet output schema and RLE round-trip correctness
- [x] 2.5 Visually inspect decoded masks overlaid on source images

## 3. Scale-up
- [x] 3.1 Build `species_list.txt` (7,806 species IDs)
- [x] 3.2 Submit array job with `--array=0-19` for both `plant` and `flower` prompts
- [x] 3.3 Monitor and confirm all 40 tasks (20 plant + 20 flower) completed successfully

## 4. Analysis
- [x] 4.1 Write `analysis/summary_stats.py` — per-task table + totals per prompt
- [x] 4.2 Write `analysis/threshold_sensitivity.py` — detection rate vs score threshold
- [x] 4.3 Write `analysis/species_detection_rate.py` — per-species breakdown
- [x] 4.4 Write `analysis/prompt_agreement.py` — plant vs flower per-image comparison
- [x] 4.5 Run all analysis scripts; confirm TOTAL rows match expected image counts
- [x] 4.6 Document findings in `artifacts/sam3-analysis-results.md`
- [x] 4.7 Update workflow doc and proposal to reflect Parquet output and dual-prompt runs
