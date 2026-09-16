#!/bin/bash
# Experiment 0 prerequisite: add any missing l+1 concept artifacts (notably 32).
#SBATCH --job-name=exp0-vectors
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/experiment_0-vectors-%j.out
#SBATCH --error=logs/experiment_0-vectors-%j.err

set -euo pipefail
CONFIG=${1:?configuration path required}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"
python -m experiment_0_calibration.prepare_concept_vectors --config "${CONFIG}"
