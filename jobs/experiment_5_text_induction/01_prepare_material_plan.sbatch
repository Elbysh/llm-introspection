#!/bin/bash
# Freeze reviewed texts, prompts, counterbalancing and calibrated coefficients.
#SBATCH --job-name=exp5-plan
#SBATCH --partition=cpu_short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/experiment_5-plan-%j.out
#SBATCH --error=logs/experiment_5-plan-%j.err

set -euo pipefail
CONFIG=${1:?configuration path required}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"
python -m experiment_5_text_induction.prepare_material_plan --config "${CONFIG}"
