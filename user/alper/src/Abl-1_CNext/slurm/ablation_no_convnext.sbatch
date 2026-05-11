#!/bin/bash
# Abl-1 — ConvNeXt knockout ablation against the B_R250 private-LB champion.
#
# Reference run: exp16c_s9_B_R250 (Stage 9, β=0.10, R=250, n_near=40)
#   public  F1 = 0.48414
#   private F1 = 0.44649   ← highest private score across all submissions
#
# This ablation reproduces that pipeline EXACTLY except for ConvNeXt:
#   - Step A (Exp-14C apply_exp14_mechanisms): replaced with ensemble_vit_only.py
#     (same vit_rnnb030.json controller, same sim_w-on-RNNB aggregator,
#      same SIM_JSON, same THR=0.085, same topk=10; no ConvNeXt branch).
#   - Step B (Exp-16C stage9 B_R250): unchanged apply_habitat_fit.py with
#     β=0.10, R=250, n_near=40, --use-altitude, thr=0.085, k=10,
#     per-source admission {OPTMix: n_floor=2, LISAH: n_floor=2}.
#
# Methodology note (fixed-knob ablation): the ConvNeXt ensemble weight
# (W=0.88, i.e. ConvNeXt contribution 0.12) was tuned with ConvNeXt
# present. The downstream habitat-fit knobs (β=0.10, R=250, n=40) and the
# per-source admission floors were tuned on top of the W=0.88 ensemble.
# This ablation does NOT re-tune them — it reports the marginal cost of
# removing ConvNeXt with all other knobs held fixed. Per the memory,
# ConvNeXt's ensemble lift is sub-noise (~+0.001 to +0.003 at w=0.23 in
# the older ViT-B baseline; in Exp-14C it's 0.12 at w=0.88), so the
# fixed-knob gap should be small.
#
# Output:
#   abl1_simw_b0.70_w1.00_ensemble.parquet            (ViT-only ensemble)
#   abl1_simw_b0.70_w1.00_thr0.085_top10.csv          (bare ViT-only, no habitat-fit)
#   abl1_s9_B_R250.csv                                (full B_R250 stack minus ConvNeXt)
#
# Usage:
#   sbatch slurm/ablation_no_convnext.sbatch

#SBATCH --job-name=abl1-no-cnxt
#SBATCH --account=paceship-clef2026_plantclef
#SBATCH --partition=cpu-medium
#SBATCH --qos=embers
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aerten3@gatech.edu

set -euo pipefail

# B_R250-locked parameters (do not change without a separate re-tune ablation)
BETA_KNN="${BETA_KNN:-0.70}"        # kNN β baked into ViT prefusion file
THR="${THR:-0.085}"
TOPK="${TOPK:-10}"
HAB_BETA="${HAB_BETA:-0.10}"
HAB_R="${HAB_R:-250}"
HAB_N="${HAB_N:-40}"

PLANTCLEF_ROOT="/storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0"
export PLANTCLEF_ROOT

REPO_ROOT="/storage/home/hcoda1/2/aerten3/plantclef-2026"
EXP14_ROOT="${REPO_ROOT}/user/alper/src/Exp-14_Adaptive_Temporal_Fusion"
EXP16_ROOT="${REPO_ROOT}/user/alper/src/Exp-16_Multi_Species_Discovery"
ABL1_ROOT="${REPO_ROOT}/user/alper/src/Abl-1_CNext"
ABL1_SCRIPTS="${ABL1_ROOT}/scripts"

mkdir -p logs
source "${REPO_ROOT}/user/alper/experiments/001-embedding-extraction/.venv/bin/activate"

# Inputs (same as Exp-14C apply_exp14_mechanisms.sbatch)
PREFUSION_DIR="${PLANTCLEF_ROOT}/aerten3/exp14c_prefusion_vitL"
PREFUSION="${PREFUSION_DIR}/vitL_prefusion_b${BETA_KNN}.parquet"
SPECIES_MAP="${PLANTCLEF_ROOT}/data/raw/class_id_to_species_mapping/class_to_species_id.json"
GEO_MASK="${PLANTCLEF_ROOT}/aerten3/species_masking/sw_europe_mask_bool.npy"
HABITAT_MASK="${EXP14_ROOT}/outputs/signals/aquatic_blacklist_mask.npy"
BIOGEO_NPZ="${EXP14_ROOT}/outputs/signals/biogeo_per_source.npz"
VIT_CONTROLLER="${EXP14_ROOT}/stage_configs/controllers/vit_rnnb030.json"
SIM_JSON="${EXP14_ROOT}/outputs/signals/visit_similarity.json"
GEO_PROFILE="${EXP16_ROOT}/outputs/habitat_classifier/species_geo_profile.parquet"
REFINED_REF="${EXP16_ROOT}/outputs/habitat_classifier/refined_source_ref.json"

OUT_DIR="${PLANTCLEF_ROOT}/aerten3/submissions/abl1_no_cnxt"
PS_DIR="${OUT_DIR}/per_src_configs"
mkdir -p "${OUT_DIR}" "${PS_DIR}"

# Locked per-source admission from Stage 9 B_R250
cat > "${PS_DIR}/locked.json" <<'EOF'
{"OPTMix": {"n_floor": 2}, "LISAH": {"n_floor": 2}}
EOF
LOCKED_PS="${PS_DIR}/locked.json"

# Pre-flight sanity
for f in "${PREFUSION}" "${HABITAT_MASK}" "${BIOGEO_NPZ}" \
         "${VIT_CONTROLLER}" "${SIM_JSON}" "${GEO_PROFILE}" "${REFINED_REF}"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing required file: $f"; exit 1; fi
done

TAG="abl1_simw_b${BETA_KNN}"

echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_NAME}  id: ${SLURM_JOB_ID}"
echo "=== Abl-1 ConvNeXt knockout (reference: B_R250 private 0.44649 / public 0.48414) ==="
echo "kNN β: ${BETA_KNN}"
echo "Habitat-fit: β=${HAB_BETA}, R=${HAB_R}, n=${HAB_N}, alt=on, thr=${THR}, k=${TOPK}"
echo "Per-source admission: OPTMix n_floor=2, LISAH n_floor=2"

# ---------------------------------------------------------------
# Step A: ViT-only ensemble (no ConvNeXt branch)
#         Saves the fused ViT probabilities as the ensemble parquet that
#         apply_habitat_fit.py consumes.
# ---------------------------------------------------------------
echo ""
echo "--- Step A: ViT-only fusion + ensemble parquet ---"
python "${ABL1_SCRIPTS}/ensemble_vit_only.py" \
    --vit-prefusion-parquet "${PREFUSION}" \
    --species-map "${SPECIES_MAP}" \
    --class-mask-npy "${GEO_MASK}" \
    --out-dir "${OUT_DIR}" \
    --thresholds "${THR}" \
    --topk "${TOPK}" \
    --vit-controller "${VIT_CONTROLLER}" \
    --vit-aggregator max \
    --vit-aggregator-per-source '{"RNNB":"sim_w"}' \
    --sim-json "${SIM_JSON}" \
    --tag "${TAG}"

ENS_PARQUET="${OUT_DIR}/${TAG}_w1.00_ensemble.parquet"
if [ ! -f "${ENS_PARQUET}" ]; then
    echo "ERROR: ensemble parquet not saved at ${ENS_PARQUET}"
    exit 1
fi

# ---------------------------------------------------------------
# Step B: habitat-fit + per-source admission (B_R250 lock)
# ---------------------------------------------------------------
echo ""
echo "--- Step B: habitat-fit (β=${HAB_BETA}, R=${HAB_R}, n=${HAB_N}) + admission ---"
OUT_CSV="${OUT_DIR}/abl1_s9_B_R250.csv"
python "${EXP16_ROOT}/scripts/apply_habitat_fit.py" \
    --ensemble-parquet "${ENS_PARQUET}" \
    --geo-profile "${GEO_PROFILE}" \
    --species-map "${SPECIES_MAP}" \
    --geo-mask "${GEO_MASK}" \
    --habitat-mask "${HABITAT_MASK}" \
    --biogeo-npz "${BIOGEO_NPZ}" \
    --refined-ref-json "${REFINED_REF}" \
    --radius-km "${HAB_R}" \
    --beta "${HAB_BETA}" \
    --n-near-min "${HAB_N}" \
    --use-altitude \
    --threshold "${THR}" \
    --topk "${TOPK}" \
    --n-floor 3 \
    --ratio-floor 0.5 \
    --per-source-json "${LOCKED_PS}" \
    --out-csv "${OUT_CSV}"

echo ""
echo "=== Done. CSVs in ${OUT_DIR} ==="
echo "  bare ViT-only (no habitat-fit):     ${TAG}_w1.00_thr${THR}_top${TOPK}.csv"
echo "  full stack minus ConvNeXt:          abl1_s9_B_R250.csv"
echo ""
echo "LB submissions to record:"
echo "  abl1_s9_B_R250.csv -> compare private F1 against B_R250 0.44649"
echo "  (bare ViT-only is optional — informational, not the headline ablation)"
