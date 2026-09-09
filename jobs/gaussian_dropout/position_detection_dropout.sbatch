#!/bin/bash


#SBATCH --job-name=position-detection-dropout
#SBATCH --partition=prod80
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/position-detection-dropout-%j.out
#SBATCH --error=logs/position-detection-dropout-%j.err

set -euo pipefail


source ".venv/bin/activate"

# Localization sweep. --coeffs are dropout levels, not injection coefficients:
# dropout rate p in [0, 1), rescaled by 1/(1-p)
# Run from the repo root: main.py resolves data/saved_vectors/llama itself and writes
# to ./plots and ./new_results.
PYTHONPATH=code/utils uv run python code/experiments/main.py \
    --type localization \
    --perturbation_type dropout \
    --layers 0 1 2 3 4 5 8 11 14 17 20 \
    --coeffs 0.05 0.1 0.2 0.4 0.6 0.8 \
    --num_sentences 10 \
    --num_trials 50 \
    --output_suffix "_reproduction"
