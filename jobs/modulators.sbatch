#!/bin/bash


#SBATCH --job-name=modulators
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/modulators-%j.out
#SBATCH --error=logs/modulators-%j.err

set -euo pipefail


source ".venv/bin/activate"


cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/modulators.py \
  --experiments e4_distance e5_alpha_ratio e6_similarity \
  --alpha 8 \
  --num_trials 20 \
  --distances 1 2 4 8 16 31 \
  --placements early late \
  --layers 8 24 \
  --alpha_ratios 1 2 4 8 \
  --similarity_reference_layer 16 \
  --num_similarity_buckets 4 \
  --vec_type avg \
  --output_dir ../plots
