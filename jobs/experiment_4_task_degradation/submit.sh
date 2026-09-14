#!/bin/bash
# Use the same preparation -> afterok execution pattern as Experiment 0.
set -euo pipefail
CONFIG=${1:?usage: bash jobs/experiment_4_task_degradation/submit.sh <configuration.yaml>}
test -f "${CONFIG}"
mkdir -p logs
PLAN_ID=$(sbatch --parsable jobs/experiment_4_task_degradation/01_prepare_material_plan.sbatch "${CONFIG}")
RUN_ID=$(sbatch --parsable --dependency="afterok:${PLAN_ID}" jobs/experiment_4_task_degradation/02_run_experiment_4.sbatch "${CONFIG}")
echo "Experiment 4 plan: ${PLAN_ID}"
echo "Experiment 4 classification: ${RUN_ID} (afterok:${PLAN_ID})"
