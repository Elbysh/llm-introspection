#!/bin/bash
# Build and normalize the concept, random and noise direction banks.
#SBATCH --job-name=calibration-prepare
#SBATCH --partition=cpu_short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/calibration-prepare-%j.out
#SBATCH --error=logs/calibration-prepare-%j.err

set -euo pipefail

CONFIG=${1:-configs/calibration/pilot.yaml}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"

python -m calibration.directions --config "${CONFIG}"
