#!/bin/bash


#SBATCH --job-name=multi-detection
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/multi-detection-%j.out
#SBATCH --error=logs/multi-detection-%j.err

set -euo pipefail


source ".venv/bin/activate"


cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/multi_detection.py \
  --k_values 0 1 2 3 4 \
  --conditions real sham \
  --prompt_styles count scale \
  --alpha 8 \
  --num_trials 20 \
  --vec_type avg \
  --output_dir ../plots
