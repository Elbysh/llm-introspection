#!/bin/bash

# Requires jobs/vector_save_qwen.sbatch to have completed first.
# GPU sizing is a conservative, unverified assumption -- see
# calibration_qwen.sbatch's comment for the rationale; adjust once known.

#SBATCH --job-name=multi-detection-budget-qwen
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/multi-detection-budget-qwen-%j.out
#SBATCH --error=logs/multi-detection-budget-qwen-%j.err

set -euo pipefail


source ".venv/bin/activate"

export HF_HOME=/gpfs/workdir/dainl/hf_cache
export HF_HUB_OFFLINE=1


cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/multi_detection.py \
  --model qwen \
  --k_values 0 1 2 3 4 \
  --conditions real random \
  --prompt_styles count \
  --dose_regime budget \
  --z_totals 2 3 4 5 \
  --num_trials 40 \
  --vec_type avg \
  --output_dir ../plots
