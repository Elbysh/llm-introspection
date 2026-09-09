#!/bin/bash


#SBATCH --job-name=layer-ordering
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/layer-ordering-%j.out
#SBATCH --error=logs/layer-ordering-%j.err

set -euo pipefail


source ".venv/bin/activate"


cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/layer_ordering.py \
  --alpha 8 \
  --num_trials 40 \
  --presentation_orders shallow_first deep_first \
  --vec_type avg \
  --output_dir ../plots
