#!/bin/bash

# Requires jobs/vector_save_qwen.sbatch to have completed first.
# GPU sizing is a conservative, unverified assumption -- see
# calibration_qwen.sbatch's comment for the rationale; adjust once known.

#SBATCH --job-name=layer-ordering-qwen
#SBATCH --partition=prod80
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:2
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/layer-ordering-qwen-%j.out
#SBATCH --error=logs/layer-ordering-qwen-%j.err

set -euo pipefail


source ".venv/bin/activate"

export HF_HOME=/raid/home/projects/llm-injection-2026/hf_cache
export HF_HUB_OFFLINE=1


cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/layer_ordering.py \
  --model qwen \
  --alphas 2 3 4 5 \
  --num_trials 60 \
  --label_assignments shallow_is_a shallow_is_b \
  --vec_type avg \
  --output_dir ../plots
