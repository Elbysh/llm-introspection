#!/bin/bash


#SBATCH --job-name=multi-detection-budget
#SBATCH --partition=prod80
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/multi-detection-budget-%j.out
#SBATCH --error=logs/multi-detection-budget-%j.err

set -euo pipefail


source ".venv/bin/activate"

export HF_HOME=/raid/home/projects/llm-injection-2026/hf_cache
export HF_HUB_OFFLINE=1


# Narrowed to the z_total 2-5 "sweet spot", more trials per cell -- see
# multi_detection.sbatch for the rationale.
cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/multi_detection.py \
  --k_values 0 1 2 3 4 \
  --conditions real random \
  --prompt_styles count \
  --dose_regime budget \
  --z_totals 2 3 4 5 \
  --num_trials 40 \
  --vec_type avg \
  --output_dir ../plots
