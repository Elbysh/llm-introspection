#!/bin/bash
# Experiment 0 steps 1-7: natural activations, projections and bootstrap.
#SBATCH --job-name=exp0-calibration
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/experiment_0-run-%j.out
#SBATCH --error=logs/experiment_0-run-%j.err

set -euo pipefail
CONFIG=${1:?configuration path required}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"
python -m experiment_0_calibration.run_experiment_0 --config "${CONFIG}"
