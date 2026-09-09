#!/bin/bash


#SBATCH --job-name=binary-detection-gaussian
#SBATCH --partition=prod40
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/binary-detection-gaussian-%j.out
#SBATCH --error=logs/binary-detection-gaussian-%j.err

set -euo pipefail


source ".venv/bin/activate"

# --strengths are gaussian levels, not injection coefficients:
# absolute sigma of the additive noise
cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/position_detection.py \
  --perturbation_type gaussian \
  --sigma_mode absolute \
  --layers 0 4 8 12 16 20 24 30 \
  --strengths 0.05 0.1 0.25 0.5 1 \
  --num_trials 10 \
  --output_dir ../plots
