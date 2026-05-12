# Light threshold sweep on Abl-3 (no temporal fusion) -- to check whether the
# 0.085 threshold (tuned for the full-pipeline distribution) is still optimal
# once temporal fusion, habitat-fit, masks, kNN, and admission have all been
# stripped.
#
# Anchor (already submitted): Abl-3 at thr=0.085 -> priv 0.38961 / pub 0.41178
#
# Sweep: thr in {0.075, 0.100, 0.115}.  3 new LB submissions.
#
# Uses the existing Abl-3 ensemble parquet (new_abl4_no_temporal_w1.00_ensemble.parquet)
# -- no re-fusion, no re-extraction. Pure apply_habitat_fit.py invocations with
# beta=1.0 + skip masks. CPU minutes per variant.
#
# Usage:  pwsh .\run_abl3_thr_sweep.ps1

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\alper\anaconda3\python.exe"

$REPO = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026"
$EXP14 = "$REPO\user\alper\src\Exp-14_Adaptive_Temporal_Fusion"
$EXP16 = "$REPO\user\alper\src\Exp-16_Multi_Species_Discovery"
$ABLN  = "$REPO\user\alper\src\Abl-N_Cumulative"
$PACE_DL = "C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\PACE_downloads"

$ENS_PARQUET = "$ABLN\outputs\new_abl4_no_temporal_w1.00_ensemble.parquet"
$SPECIES_MAP   = "$PACE_DL\class_id_to_species_mapping\class_to_species_id.json"
$GEO_MASK      = "$PACE_DL\sw_europe_mask_bool.npy"
$HABITAT_MASK  = "$EXP14\outputs\signals\aquatic_blacklist_mask.npy"
$BIOGEO_NPZ    = "$EXP14\outputs\signals\biogeo_per_source.npz"
$GEO_PROFILE   = "$EXP16\outputs\habitat_classifier\species_geo_profile.parquet"
$REFINED_REF   = "$EXP16\outputs\habitat_classifier\refined_source_ref.json"

$TOPK = "10"
$HAB_BETA = "1.0"   # identity -- no habitat-fit demotion
$HAB_R = "250"
$HAB_N = "40"

$OUT_DIR = "$ABLN\outputs"
New-Item -Force -ItemType Directory $OUT_DIR | Out-Null

foreach ($f in @($ENS_PARQUET, $SPECIES_MAP, $GEO_MASK, $HABITAT_MASK,
                  $BIOGEO_NPZ, $GEO_PROFILE, $REFINED_REF)) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: missing required file: $f" -ForegroundColor Red
        exit 1
    }
}

Write-Host "==========================================================="
Write-Host "Threshold sweep on Abl-3 (no temporal fusion stripped config)"
Write-Host "==========================================================="
Write-Host "Anchor: thr=0.085  priv 0.38961 / pub 0.41178"
Write-Host "Sweep:  thr in {0.075, 0.100, 0.115}"
Write-Host ""

$thresholds = @("0.075", "0.100", "0.115")
foreach ($thr in $thresholds) {
    $tag = "abl3_thrsweep_thr$($thr.Replace('.','p'))"
    $OUT_CSV = "$OUT_DIR\${tag}.csv"
    Write-Host "--- threshold = $thr ---"
    & $PYTHON "$EXP16\scripts\apply_habitat_fit.py" `
        --ensemble-parquet $ENS_PARQUET `
        --geo-profile $GEO_PROFILE `
        --species-map $SPECIES_MAP `
        --geo-mask $GEO_MASK `
        --habitat-mask $HABITAT_MASK `
        --biogeo-npz $BIOGEO_NPZ `
        --refined-ref-json $REFINED_REF `
        --radius-km $HAB_R `
        --beta $HAB_BETA `
        --n-near-min $HAB_N `
        --use-altitude `
        --skip-aquatic `
        --skip-biogeo `
        --threshold $thr `
        --topk $TOPK `
        --n-floor 3 `
        --ratio-floor 0.5 `
        --out-csv $OUT_CSV
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at thr=$thr" -ForegroundColor Red; exit 1 }
    Write-Host "  -> $OUT_CSV"
    Write-Host ""
}

Write-Host "==========================================================="
Write-Host "Done. Submit these to Kaggle and report priv/pub for each:"
Write-Host "  abl3_thrsweep_thr0p075.csv   (vs anchor 0.38961)"
Write-Host "  abl3_thrsweep_thr0p100.csv"
Write-Host "  abl3_thrsweep_thr0p115.csv"
Write-Host "==========================================================="
