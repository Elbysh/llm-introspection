#!/bin/bash
# Execute every fresh-context sham and concept-injection trial.
#SBATCH --job-name=exp5-induction
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/experiment_5-run-%j.out
#SBATCH --error=logs/experiment_5-run-%j.err

set -euo pipefail
CONFIG=${1:?configuration path required}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"
python -m experiment_5_text_induction.run_experiment_5 --config "${CONFIG}"
