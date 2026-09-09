#!/bin/bash


#SBATCH --job-name=position-detection-random
#SBATCH --partition=prod80
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/position-detection-random-%j.out
#SBATCH --error=logs/position-detection-random-%j.err

set -euo pipefail


source ".venv/bin/activate"

# Localization sweep over the 10 random-direction controls (random_s0 ... random_s9),
# same grid as the concept-vector run in jobs/position_detection.sbatch.
# Run from the repo root: main.py resolves data/saved_vectors/llama itself and writes
# to ./plots and ./new_results.
PYTHONPATH=code/utils uv run python code/experiments/main.py \
    --type localization \
    --concepts random \
    --num_vectors 10 \
    --layers 0 1 2 3 4 5 8 11 14 17 20 \
    --coeffs 2 5 8 11 14 17 20 \
    --num_sentences 10 \
    --num_trials 50 \
    --output_suffix "_random"
