#!/bin/bash
# Prepare the labelled examples, both mappings, directions and token spans.
#SBATCH --job-name=exp4-plan
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/experiment_4-plan-%j.out
#SBATCH --error=logs/experiment_4-plan-%j.err

set -euo pipefail
CONFIG=${1:?configuration path required}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"
python -m experiment_4_task_degradation.prepare_material_plan --config "${CONFIG}"
