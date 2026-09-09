#!/bin/bash
# Estimate s(layer, direction) from natural activations and create diagnostics.
#SBATCH --job-name=calibration-run
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/calibration-run-%j.out
#SBATCH --error=logs/calibration-run-%j.err

set -euo pipefail

CONFIG=${1:-configs/calibration/pilot.yaml}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"

python -m calibration.compute_scales --config "${CONFIG}"
