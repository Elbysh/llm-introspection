#!/bin/bash
# Experiment 0 material: freeze contexts, positions, direction IDs and seeds.
#SBATCH --job-name=exp0-plan
#SBATCH --partition=cpu_short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/experiment_0-plan-%j.out
#SBATCH --error=logs/experiment_0-plan-%j.err

set -euo pipefail
CONFIG=${1:?configuration path required}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"
python -m experiment_0_calibration.prepare_plan --config "${CONFIG}"
