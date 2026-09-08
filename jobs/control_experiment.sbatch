#!/bin/bash


#SBATCH --job-name=binary-detection
#SBATCH --partition=prod80
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/binary-detection-%j.out
#SBATCH --error=logs/binary-detection-%j.err

set -euo pipefail


source ".venv/bin/activate"

cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/position_detection.py \
  --concept all \
  --layers 0 4 8 12 16 20 24 30 \
  --strengths 1 2 3 4 5 \
  --num_trials 10 \
  --vec_type avg \
  --control_question "Can humans breathe underwater without equipment?" \
  --output_dir ../plots