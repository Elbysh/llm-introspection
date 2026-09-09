#!/bin/bash


#SBATCH --job-name=calibration
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/calibration-%j.out
#SBATCH --error=logs/calibration-%j.err

set -euo pipefail


source ".venv/bin/activate"


cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/calibration.py \
  --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 \
  --alphas 0 1 2 4 6 8 12 16 \
  --num_trials 5 \
  --vec_type avg \
  --output_dir ../plots
