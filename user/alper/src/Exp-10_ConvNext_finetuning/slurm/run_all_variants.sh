#!/bin/bash
# Submit training for all 4 ConvNeXt variants in parallel.
# Each variant runs Phase 1 -> Phase 2 with SLURM dependencies.
# Usage: bash slurm/run_all_variants.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Submitting all ConvNeXt variants ==="
echo "Each variant: Phase 1 (head-only) -> Phase 2 (full fine-tune)"
echo "All Phase 1 jobs run simultaneously. Phase 2 starts after Phase 1 completes."
echo ""

# Start with base (most comparable to ViT), then others
for VARIANT in base tiny small large; do
    bash "${SCRIPT_DIR}/run_training.sh" "${VARIANT}"
    echo ""
done

echo "=== All jobs submitted. Monitor with: squeue -u \$USER ==="
