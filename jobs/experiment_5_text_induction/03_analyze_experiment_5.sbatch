#!/bin/bash
# Analyze only after a complete Experiment 5 result has been written.
#SBATCH --job-name=exp5-analysis
#SBATCH --partition=cpu_short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/experiment_5-analysis-%j.out
#SBATCH --error=logs/experiment_5-analysis-%j.err

set -euo pipefail
CONFIG=${1:?configuration path required}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"
python -m experiment_5_text_induction.analyze_experiment_5 --config "${CONFIG}"
