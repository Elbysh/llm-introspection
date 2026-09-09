#!/bin/bash


#SBATCH --job-name=strength-comparison
#SBATCH --partition=prod40
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/strength-comparison-%j.out
#SBATCH --error=logs/strength-comparison-%j.err

set -euo pipefail


source ".venv/bin/activate"


cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/strength_comparison.py \
  --concepts random \
  --num_vectors 10 \
  --layers 0 1 2 3 4 5 6 7 8 9 10 15 20 25 30 \
  --pairs best \
  --num_trials 30 \
  --output_dir ../plots \
  --output_suffix "_reproduction"