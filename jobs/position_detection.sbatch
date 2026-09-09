#!/bin/bash


#SBATCH --job-name=position-detection
#SBATCH --partition=prod80
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/position-detection-%j.out
#SBATCH --error=logs/position-detection-%j.err

set -euo pipefail


source ".venv/bin/activate"

cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/main.py \
    --type localization \
    --layers 0 1 2 3 4 5 8 11 14 17 20 \
    --coeffs 2 5 8 11 14 17 20 \
    --num_sentences 10 \
    --num_trials 50 \
    --output_suffix "_reproduction"
