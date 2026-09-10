#!/bin/bash
# Submit the three Experiment 0 stages with afterok dependencies.
set -euo pipefail

RUN_NAME=${1:-development_full}
CONFIG="configs/calibration/experiment_0/${RUN_NAME}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing configuration: ${CONFIG}" >&2
  exit 2
fi
mkdir -p logs

VECTORS_ID=$(sbatch --parsable jobs/calibration/experiment_0/ensure_vectors.sbatch "${CONFIG}")
PLAN_ID=$(sbatch --parsable --dependency="afterok:${VECTORS_ID}" jobs/calibration/experiment_0/prepare_plan.sbatch "${CONFIG}")
RUN_ID=$(sbatch --parsable --dependency="afterok:${PLAN_ID}" jobs/calibration/experiment_0/run.sbatch "${CONFIG}")

echo "Experiment 0 concept vectors: ${VECTORS_ID}"
echo "Experiment 0 plan: ${PLAN_ID} (afterok:${VECTORS_ID})"
echo "Experiment 0 calibration: ${RUN_ID} (afterok:${PLAN_ID})"

