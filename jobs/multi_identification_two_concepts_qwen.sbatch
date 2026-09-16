#!/bin/bash

# Requires jobs/vector_save_qwen.sbatch to have completed first.
# GPU sizing is a conservative, unverified assumption -- see
# calibration_qwen.sbatch's comment for the rationale; adjust once known.

#SBATCH --job-name=multi-id-two-concepts-qwen
#SBATCH --partition=prod80
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:2
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/multi-id-two-concepts-qwen-%j.out
#SBATCH --error=logs/multi-id-two-concepts-qwen-%j.err

set -euo pipefail

source ".venv/bin/activate"

export HF_HOME=/raid/home/projects/llm-injection-2026/hf_cache
export HF_HUB_OFFLINE=1

cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/multi_identification.py \
  --model qwen \
  --n_real 2 \
  --n_random 0 \
  --condition_name two_concepts \
  --alphas 1 2 3 4 5 6 7 \
  --num_trials 30 \
  --vec_type avg \
  --output_dir ../plots
