#!/bin/bash
# Submit direction preparation, then calibration with a Slurm dependency.
set -euo pipefail

RUN_SIZE=${1:-pilot}
case "${RUN_SIZE}" in
  pilot|full) ;;
  *) echo "Usage: bash jobs/submit_calibration.sh [pilot|full]" >&2; exit 2 ;;
esac

CONFIG="configs/calibration/${RUN_SIZE}.yaml"
mkdir -p logs

PREPARE_ID=$(sbatch --parsable jobs/calibration_prepare.sbatch "${CONFIG}")
CALIBRATION_ID=$(sbatch --parsable --dependency="afterok:${PREPARE_ID}" jobs/calibration_run.sbatch "${CONFIG}")

echo "Preparation job: ${PREPARE_ID}"
echo "Calibration job: ${CALIBRATION_ID} (afterok:${PREPARE_ID})"
