# Tasks: 001-sam3-pcs

## 1. Setup
- [x] 1.1 Create venv at `~/scratch/plantclef-2026/.venv` (Python 3.12)
- [x] 1.2 Add `vendor/sam3` git submodule
- [x] 1.3 Document SAM3 transformers API in `docs/vendor/sam3.md`
- [ ] 1.4 Install dependencies into venv (`uv sync`)
- [ ] 1.5 Verify HuggingFace gated model access for `facebook/sam3`
- [ ] 1.6 Prepare a 100-image test chunk for validation

## 2. Implementation
- [x] 2.1 Write `src/run_sam3.py` — batched inference script
- [x] 2.2 Write `submit.sbatch` — SLURM array job script
- [ ] 2.3 Validate on test chunk on a compute node (no crash, output produced)
- [ ] 2.4 Verify `.npz` output and RLE round-trip correctness
- [ ] 2.5 Visually inspect decoded masks overlaid on source images

## 3. Scale-up
- [ ] 3.1 Determine chunk count; pre-split image list into `~/scratch/plantclef-2026/chunks/`
- [ ] 3.2 Update `--array` bound in `submit.sbatch` to match chunk count
- [ ] 3.3 Submit array job; monitor with `squeue -u $USER`

## 4. Analysis
- [ ] 4.1 Spot-check mask quality on 10 images per biome/taxon
- [ ] 4.2 Log throughput numbers from SLURM stdout files
- [ ] 4.3 Document findings in `results.md`
