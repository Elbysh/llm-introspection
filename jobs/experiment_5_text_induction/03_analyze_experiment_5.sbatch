#!/bin/bash
# Analyze only after a complete Experiment 5 result has been written.
# CPU-only: reads the finished trials and writes CSVs and figures, so it runs on
# prod10 and leaves the prod40/prod80 QOS slots for GPU work. prod10 caps memory at
# 15G (MaxMemPerNode=15360) and CPUs at 4 per GPU.
#SBATCH --job-name=exp5-analysis
#SBATCH --partition=prod10
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --output=logs/experiment_5-analysis-%j.out
#SBATCH --error=logs/experiment_5-analysis-%j.err

set -euo pipefail
CONFIG=${1:?configuration path required}
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export PYTHONPATH="${PWD}/code${PYTHONPATH:+:${PYTHONPATH}}"
python -m experiment_5_text_induction.analyze_experiment_5 --config "${CONFIG}"
