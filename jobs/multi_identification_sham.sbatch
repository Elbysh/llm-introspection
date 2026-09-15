#!/bin/bash


#SBATCH --job-name=multi-id-sham
#SBATCH --partition=prod80
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/multi-id-sham-%j.out
#SBATCH --error=logs/multi-id-sham-%j.err

set -euo pipefail

source ".venv/bin/activate"

export HF_HOME=/raid/home/projects/llm-injection-2026/hf_cache
export HF_HUB_OFFLINE=1

cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/multi_identification.py \
  --n_real 0 \
  --n_random 0 \
  --condition_name sham \
  --alphas 1 2 3 4 5 6 7 \
  --num_trials 30 \
  --vec_type avg \
  --output_dir ../plots
