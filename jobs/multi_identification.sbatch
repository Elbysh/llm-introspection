#!/bin/bash


#SBATCH --job-name=multi-identification
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/multi-identification-%j.out
#SBATCH --error=logs/multi-identification-%j.err

set -euo pipefail


source ".venv/bin/activate"


cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/multi_identification.py \
  --run_all_conditions \
  --alpha 8 \
  --num_trials 30 \
  --vec_type avg \
  --output_dir ../plots
