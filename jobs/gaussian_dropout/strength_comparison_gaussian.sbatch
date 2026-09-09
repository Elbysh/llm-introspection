#!/bin/bash


#SBATCH --job-name=strength-comparison-gaussian
#SBATCH --partition=prod40
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/strength-comparison-gaussian-%j.out
#SBATCH --error=logs/strength-comparison-gaussian-%j.err

set -euo pipefail


source ".venv/bin/activate"

# --custom_pairs are (weaker, stronger) gaussian levels, not injection coefficients:
# absolute sigma of the additive noise
cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/strength_comparison.py \
  --perturbation_type gaussian \
  --sigma_mode absolute \
  --custom_pairs 0.1,0.3 0.25,0.5 0.1,0.5 0.5,1 \
  --layers 0 1 2 3 4 5 6 7 8 9 10 15 20 25 30 \
  --num_trials 30 \
  --output_dir ../plots \
  --output_suffix "_reproduction"
