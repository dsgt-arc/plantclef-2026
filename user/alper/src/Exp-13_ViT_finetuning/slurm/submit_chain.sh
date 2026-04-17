#!/bin/bash
# Submit Phase 1 + Phase 2 as a dependency chain for a given ViT variant.
# Phase 2 waits for Phase 1 to finish successfully (afterok).
#
# Usage:
#   ./submit_chain.sh dinov2_large
#   ./submit_chain.sh dinov3_large

set -euo pipefail

MODEL="${1:?usage: $0 <dinov2_large|dinov3_large>}"

if [[ "${MODEL}" != "dinov2_large" && "${MODEL}" != "dinov3_large" ]]; then
    echo "ERROR: MODEL must be 'dinov2_large' or 'dinov3_large' (got '${MODEL}')"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">> Submitting Phase 1 (${MODEL})"
P1=$(sbatch --parsable --export=MODEL="${MODEL}" "${SCRIPT_DIR}/phase1_train.sbatch")
echo "   Phase 1 jobid: ${P1}"

echo ">> Submitting Phase 2 (${MODEL}) with dependency afterok:${P1}"
P2=$(sbatch --parsable --dependency=afterok:"${P1}" --export=MODEL="${MODEL}" "${SCRIPT_DIR}/phase2_train.sbatch")
echo "   Phase 2 jobid: ${P2}"

echo ""
echo "Submitted chain for ${MODEL}:"
echo "  P1: ${P1}"
echo "  P2: ${P2} (waits for ${P1})"
