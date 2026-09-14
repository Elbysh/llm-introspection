#!/bin/bash
# Follow the repository's prepare -> afterok run convention.
set -euo pipefail
CONFIG=${1:?usage: bash jobs/experiment_5_text_induction/submit.sh <configuration.yaml>}
test -f "${CONFIG}"
mkdir -p logs
PLAN_ID=$(sbatch --parsable jobs/experiment_5_text_induction/01_prepare_material_plan.sbatch "${CONFIG}")
RUN_ID=$(sbatch --parsable --dependency="afterok:${PLAN_ID}" jobs/experiment_5_text_induction/02_run_experiment_5.sbatch "${CONFIG}")
ANALYSIS_ID=$(sbatch --parsable --dependency="afterok:${RUN_ID}" jobs/experiment_5_text_induction/03_analyze_experiment_5.sbatch "${CONFIG}")
echo "Experiment 5 plan: ${PLAN_ID}"
echo "Experiment 5 run: ${RUN_ID} (afterok:${PLAN_ID})"
echo "Experiment 5 analysis: ${ANALYSIS_ID} (afterok:${RUN_ID})"
