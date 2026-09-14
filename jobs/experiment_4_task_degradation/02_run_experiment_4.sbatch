#!/bin/bash
# Run the semantic task under sham and the planned interventions.
#SBATCH --job-name=exp4-classify
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/experiment_4-run-%j.out
#SBATCH --error=logs/experiment_4-run-%j.err

set -euo pipefail
CONFIG=${1:?configuration path required}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"
python -m experiment_4_task_degradation.run_experiment_4 --config "${CONFIG}"
