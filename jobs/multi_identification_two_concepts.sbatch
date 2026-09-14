#!/bin/bash


#SBATCH --job-name=multi-id-two-concepts
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/multi-id-two-concepts-%j.out
#SBATCH --error=logs/multi-id-two-concepts-%j.err

set -euo pipefail

source ".venv/bin/activate"

export HF_HOME=/gpfs/workdir/dainl/hf_cache
export HF_HUB_OFFLINE=1

cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/multi_identification.py \
  --n_real 2 \
  --n_random 0 \
  --condition_name two_concepts \
  --alphas 1 2 3 4 5 6 7 \
  --num_trials 30 \
  --vec_type avg \
  --output_dir ../plots
